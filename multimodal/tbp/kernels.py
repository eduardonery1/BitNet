"""CUDA kernels for Ternary Backpropagation (TBP), compiled at runtime with NVRTC through CuPy.

Weights are stored packed: 2 bits per weight (code = w + 1 in {0, 1, 2}), 16 weights per uint32, row-major, each row
padded to `stride = ceil(cols / 16)` words (padding code 1 = weight 0). Torch tensors are passed zero-copy.

Kernels
- tern_pack / tern_unpack:        int8 [R, C] <-> packed uint32 [R, stride]; tern_unpack_f32 -> float32 (for cuBLAS)
- tern_mm_nt:  Y[M, N] = scale * A[M, K] @ W[N, K]^T      (forward, W packed [N, K])
- tern_mm_nn:  Y[M, N] = scale * A[M, K] @ W[K, N]        (error sent back, W packed [K, N])
- tbp_update:  per output row: direction -sign(G), movability, exact top-k of |G| (ties -> lowest index), int8 counter
               votes, flips written into the packed weights; returns the number of flips
"""

from __future__ import annotations

import math

import cupy as cp
import torch

_SRC = r"""

__device__ __forceinline__ int tget(const unsigned* __restrict__ Wp, int stride, int r, int c) {
    unsigned w = Wp[(long long)r * stride + (c >> 4)];
    return (int)((w >> ((c & 15) << 1)) & 3u) - 1;
}

extern "C" __global__ void tern_pack(const signed char* __restrict__ W, unsigned* __restrict__ Wp, int R, int C, int stride) {
    int r = blockIdx.y;
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R || wi >= stride) return;
    unsigned word = 0;
    for (int j = 0; j < 16; ++j) {
        int c = wi * 16 + j;
        unsigned code = 1u;
        if (c < C) code = (unsigned)(W[(long long)r * C + c] + 1);
        word |= (code & 3u) << (2 * j);
    }
    Wp[(long long)r * stride + wi] = word;
}

extern "C" __global__ void tern_unpack(const unsigned* __restrict__ Wp, signed char* __restrict__ W, int R, int C, int stride) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)R * C) return;
    int r = (int)(i / C), c = (int)(i % C);
    W[i] = (signed char)tget(Wp, stride, r, c);
}

extern "C" __global__ void tern_unpack_f32(const unsigned* __restrict__ Wp, float* __restrict__ W, int R, int C, int stride) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)R * C) return;
    int r = (int)(i / C), c = (int)(i % C);
    W[i] = (float)tget(Wp, stride, r, c);
}

// 64x64 output tile, BK = 32, 16x16 threads, 4x4 outputs per thread (rows ty + 16 i, cols tx + 16 j).
// Each packed word is read once per tile and decoded into 16 shared-memory weights.
// TRANS=false: B(k, n) = w[n][k] (2 words per n-row of the tile); TRANS=true: B(k, n) = w[k][n] (4 words per k-row).
#define MM_BM 64
#define MM_BN 64
#define MM_BK 32
template <bool TRANS>
__device__ __forceinline__ void tern_mm(const float* __restrict__ A, const unsigned* __restrict__ Wp,
                                        float* __restrict__ Y, int M, int N, int K, int stride, float scale) {
    __shared__ float As[MM_BK][MM_BM + 4];
    __shared__ float Bs[MM_BK][MM_BN + 4];
    int tx = threadIdx.x, ty = threadIdx.y, t = ty * 16 + tx;
    int n0 = blockIdx.x * MM_BN, m0 = blockIdx.y * MM_BM;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += MM_BK) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {   // A tile: 64 x 32, coalesced along k, stored transposed
            int idx = t + 256 * i, m = idx >> 5, k = idx & 31;
            As[k][m] = (m0 + m < M && k0 + k < K) ? A[(long long)(m0 + m) * K + k0 + k] : 0.f;
        }
        if (t < 128) {                  // B tile: 128 packed words, each decoded once
            int row, wi, base;
            long long grow;
            bool ok;
            if (!TRANS) { row = t >> 1; wi = (k0 >> 4) + (t & 1); base = (t & 1) * 16; grow = n0 + row; ok = grow < N; }
            else        { row = t >> 2; wi = (n0 >> 4) + (t & 3); base = (t & 3) * 16; grow = k0 + row; ok = grow < K; }
            unsigned word = (ok && wi < stride) ? Wp[grow * stride + wi] : 0x55555555u;   // code 1 = weight 0
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                float v = (float)((int)((word >> (2 * j)) & 3u) - 1);
                if (!TRANS) { int k = base + j; Bs[k][row] = (k0 + k < K) ? v : 0.f; }
                else        { int n = base + j; Bs[row][n] = (n0 + n < N) ? v : 0.f; }
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < MM_BK; ++k) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[k][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int m = m0 + ty + 16 * i, n = n0 + tx + 16 * j;
            if (m < M && n < N) Y[(long long)m * N + n] = acc[i][j] * scale;
        }
}

extern "C" __global__ void tern_mm_nt(const float* A, const unsigned* Wp, float* Y, int M, int N, int K, int stride, float scale) {
    tern_mm<false>(A, Wp, Y, M, N, K, stride, scale);
}

extern "C" __global__ void tern_mm_nn(const float* A, const unsigned* Wp, float* Y, int M, int N, int K, int stride, float scale) {
    tern_mm<true>(A, Wp, Y, M, N, K, stride, scale);
}

// Eligible-score bits of column c: float bits of |G| if the step -sign(G) is allowed, else 0. Positive floats order
// like their bit patterns, so the radix select below works on the raw bits.
__device__ __forceinline__ unsigned score_bits(const float* __restrict__ g, const unsigned* __restrict__ Wp,
                                               int stride, int r, int c) {
    float v = g[c];
    if (!(v != 0.f) || v != v) return 0u;   // zero or NaN
    int d = v > 0.f ? -1 : 1;
    int w = tget(Wp, stride, r, c);
    bool movable = (d > 0 && w < 1) || (d < 0 && w > -1);
    return movable ? __float_as_uint(fabsf(v)) : 0u;
}

#define UPD_THREADS 256

// Deterministic per-(seed, row, col) hash for stochastic counter increments (mirrored in Python for tests).
__device__ __forceinline__ unsigned vote_hash(unsigned seed, unsigned r, unsigned c) {
    unsigned h = seed * 0x9E3779B1u ^ (r + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}

// Counters: BITS=8 -> int8 [R, C]; BITS=3 or 4 -> packed fields, 32/BITS per uint32 (3: 10 per word, |c| <= 3;
// 4: 8 per word, |c| <= 7), code = c + 2^(BITS-1) - 1, row stride cstride words. A same-direction vote increments the
// counter only if (vote_hash >> 8) < acc_thr24 (acc_thr24 = 1 << 24 -> always); an opposite vote restarts it at +-1.
template <int BITS>
__device__ __forceinline__ void tbp_update_impl(const float* __restrict__ G, unsigned* __restrict__ Wp,
                                                signed char* __restrict__ cnt, unsigned* __restrict__ Cp, int cstride,
                                                int R, int C, int stride, int k, int T, int* __restrict__ flips,
                                                int use_cache, unsigned acc_thr24, unsigned seed,
                                                const int* __restrict__ krows, int use_krows) {
    extern __shared__ unsigned sb[];   // C entries when use_cache (row fits in shared memory)
    int r = blockIdx.x;
    if (r >= R) return;
    const float* g = G + (long long)r * C;
    int tid = threadIdx.x;
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix, s_mask, s_rem, s_total;
    __shared__ unsigned tiecnt[UPD_THREADS];

#define SB(c) (use_cache ? sb[c] : score_bits(g, Wp, stride, r, (c)))
    if (use_cache) {
        for (int c = tid; c < C; c += UPD_THREADS) sb[c] = score_bits(g, Wp, stride, r, c);
    }
    // number of eligible columns
    if (tid == 0) s_total = 0u;
    __syncthreads();
    unsigned local = 0;
    for (int c = tid; c < C; c += UPD_THREADS) local += SB(c) != 0u;
    atomicAdd(&s_total, local);
    __syncthreads();
    int k_r = use_krows ? max(krows[r], 0) : k;   // optional per-row count (fractional k sampled on the host side)
    unsigned kk = min((unsigned)k_r, s_total);
    if (kk == 0u) return;

    // radix select (8 bits per pass, MSB first) of the kk-th largest eligible score
    if (tid == 0) { s_prefix = 0u; s_mask = 0u; s_rem = kk; }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += UPD_THREADS) hist[i] = 0u;
        __syncthreads();
        unsigned prefix = s_prefix, mask = s_mask;
        for (int c = tid; c < C; c += UPD_THREADS) {
            unsigned b = SB(c);
            if (b != 0u && (b & mask) == prefix) atomicAdd(&hist[(b >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid < 32) {   // warp 0 finds the bin holding the rem-th largest: lane L owns bins 255-8L .. 248-8L
            unsigned rem = s_rem, cnts[8], lsum = 0u;
#pragma unroll
            for (int j = 0; j < 8; ++j) { cnts[j] = hist[255 - 8 * tid - j]; lsum += cnts[j]; }
            unsigned incl = lsum;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                unsigned v = __shfl_up_sync(0xffffffffu, incl, o);
                if (tid >= o) incl += v;
            }
            unsigned ball = __ballot_sync(0xffffffffu, incl >= rem);   // nonzero: matching entries >= rem
            int L = __ffs(ball) - 1;
            if (tid == L) {
                unsigned cum = incl - lsum;
                for (int j = 0; j < 8; ++j) {
                    if (cum + cnts[j] >= rem) {
                        s_prefix = prefix | ((unsigned)(255 - 8 * tid - j) << shift);
                        s_rem = rem - cum;
                        break;
                    }
                    cum += cnts[j];
                }
                s_mask = mask | (255u << shift);
            }
        }
        __syncthreads();
    }
    unsigned thr = s_prefix, ties_needed = s_rem;   // kk-th largest bits; how many equal-to-thr entries to take

    // ties are taken in column order: contiguous chunk per thread + exclusive scan of per-thread tie counts
    int chunk = (C + UPD_THREADS - 1) / UPD_THREADS;
    int c0 = tid * chunk, c1 = min(C, c0 + chunk);
    unsigned myties = 0u;
    for (int c = c0; c < c1; ++c) myties += SB(c) == thr;
    // block exclusive scan of myties (warp shuffles, then the 8 warp totals)
    int lane = tid & 31, wid = tid >> 5;
    unsigned incl = myties;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        unsigned v = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += v;
    }
    if (lane == 31) tiecnt[wid] = incl;
    __syncthreads();
    if (wid == 0) {
        unsigned w = lane < UPD_THREADS / 32 ? tiecnt[lane] : 0u, wi = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            unsigned v = __shfl_up_sync(0xffffffffu, wi, o);
            if (lane >= o) wi += v;
        }
        if (lane < UPD_THREADS / 32) tiecnt[lane] = wi - w;   // exclusive warp offsets
    }
    __syncthreads();
    unsigned tie_rank = incl - myties + tiecnt[wid];
    __syncthreads();   // every thread has its rank before any packed weight changes below

    unsigned* wrow = Wp + (long long)r * stride;
    int nflip = 0;
    for (int c = c0; c < c1; ++c) {
        unsigned b = SB(c);
        bool sel = false;
        if (b != 0u) {
            if (b > thr) sel = true;
            else if (b == thr) { sel = tie_rank < ties_needed; ++tie_rank; }
        }
        if (!sel) continue;
        int d = g[c] > 0.f ? -1 : 1;
        int cv, old;
        const int FPW = 32 / BITS, OFF = (1 << (BITS - 1)) - 1;
        const unsigned FMASK = (1u << BITS) - 1u;
        if (BITS != 8) old = (int)((Cp[(long long)r * cstride + c / FPW] >> (BITS * (c % FPW))) & FMASK) - OFF;
        else           old = cnt[(long long)r * C + c];
        int sc = (old > 0) - (old < 0);
        if (sc == d) cv = ((vote_hash(seed, (unsigned)r, (unsigned)c) >> 8) < acc_thr24) ? old + d : old;
        else         cv = d;
        if (cv >= T || cv <= -T) {
            int w = tget(Wp, stride, r, c);
            int wn = w + (cv > 0 ? 1 : -1);
            unsigned x = ((unsigned)(w + 1) ^ (unsigned)(wn + 1)) << ((c & 15) << 1);
            atomicXor(&wrow[c >> 4], x);
            cv = 0;
            ++nflip;
        }
        if (BITS != 8) {
            if (cv != old)
                atomicXor(&Cp[(long long)r * cstride + c / FPW],
                          ((unsigned)(old + OFF) ^ (unsigned)(cv + OFF)) << (BITS * (c % FPW)));
        } else {
            cnt[(long long)r * C + c] = (signed char)cv;
        }
    }
    if (nflip) atomicAdd(flips, nflip);
#undef SB
}

extern "C" __global__ void tbp_update(const float* G, unsigned* Wp, signed char* cnt, int R, int C, int stride, int k,
                                      int T, int* flips, int use_cache, unsigned acc_thr24, unsigned seed,
                                      const int* krows, int use_krows) {
    tbp_update_impl<8>(G, Wp, cnt, nullptr, 0, R, C, stride, k, T, flips, use_cache, acc_thr24, seed, krows,
                       use_krows);
}

extern "C" __global__ void tbp_update3(const float* G, unsigned* Wp, unsigned* Cp, int cstride, int R, int C,
                                       int stride, int k, int T, int* flips, int use_cache, unsigned acc_thr24,
                                       unsigned seed, const int* krows, int use_krows) {
    tbp_update_impl<3>(G, Wp, nullptr, Cp, cstride, R, C, stride, k, T, flips, use_cache, acc_thr24, seed, krows,
                       use_krows);
}

extern "C" __global__ void tbp_update4(const float* G, unsigned* Wp, unsigned* Cp, int cstride, int R, int C,
                                       int stride, int k, int T, int* flips, int use_cache, unsigned acc_thr24,
                                       unsigned seed, const int* krows, int use_krows) {
    tbp_update_impl<4>(G, Wp, nullptr, Cp, cstride, R, C, stride, k, T, flips, use_cache, acc_thr24, seed, krows,
                       use_krows);
}

"""

_mod = None


def _module():
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC, options=("-std=c++14",))
    return _mod


def _k(name):
    return _module().get_function(name)


def _cp(t: torch.Tensor):
    assert t.is_cuda and t.is_contiguous(), "expected a contiguous CUDA tensor"
    return cp.asarray(t)


def stride_of(cols: int) -> int:
    return (cols + 15) // 16


def pack(W: torch.Tensor) -> torch.Tensor:
    """int8 [R, C] in {-1, 0, 1} (CUDA) -> packed int32 [R, ceil(C/16)] (bit pattern of uint32)."""
    W = W.to(torch.int8).contiguous()
    R, C = W.shape
    s = stride_of(C)
    Wp = torch.empty((R, s), dtype=torch.int32, device=W.device)
    _k("tern_pack")(((s + 127) // 128, R), (128,), (_cp(W), _cp(Wp), cp.int32(R), cp.int32(C), cp.int32(s)))
    return Wp


def unpack(Wp: torch.Tensor, cols: int, dtype=torch.int8) -> torch.Tensor:
    R, s = Wp.shape
    assert s == stride_of(cols)
    out = torch.empty((R, cols), dtype=dtype, device=Wp.device)
    name = "tern_unpack" if dtype == torch.int8 else "tern_unpack_f32"
    assert dtype in (torch.int8, torch.float32)
    n = R * cols
    _k(name)(((n + 255) // 256,), (256,), (_cp(Wp), _cp(out), cp.int32(R), cp.int32(cols), cp.int32(s)))
    return out


def mm_nt(A: torch.Tensor, Wp: torch.Tensor, n_out: int, scale: float = 1.0) -> torch.Tensor:
    """scale * A[M, K] @ W[n_out, K]^T with W packed [n_out, ceil(K/16)]."""
    A = A.float().contiguous()
    M, K = A.shape
    assert Wp.shape == (n_out, stride_of(K))
    Y = torch.empty((M, n_out), dtype=torch.float32, device=A.device)
    _k("tern_mm_nt")(((n_out + 63) // 64, (M + 63) // 64), (16, 16),
                     (_cp(A), _cp(Wp), _cp(Y), cp.int32(M), cp.int32(n_out), cp.int32(K), cp.int32(Wp.shape[1]),
                      cp.float32(scale)))
    return Y


def mm_nn(A: torch.Tensor, Wp: torch.Tensor, n_cols: int, scale: float = 1.0) -> torch.Tensor:
    """scale * A[M, K] @ W[K, n_cols] with W packed [K, ceil(n_cols/16)] (the error sent back through W)."""
    A = A.float().contiguous()
    M, K = A.shape
    assert Wp.shape == (K, stride_of(n_cols))
    Y = torch.empty((M, n_cols), dtype=torch.float32, device=A.device)
    _k("tern_mm_nn")(((n_cols + 63) // 64, (M + 63) // 64), (16, 16),
                     (_cp(A), _cp(Wp), _cp(Y), cp.int32(M), cp.int32(n_cols), cp.int32(K), cp.int32(Wp.shape[1]),
                      cp.float32(scale)))
    return Y


CACHE_MAX_COLS = 10240  # 40 KB of dynamic shared memory + the kernel's static arrays stay under the 48 KB default


def _fpw(bits: int) -> int:
    assert bits in (3, 4)
    return 32 // bits  # fields per word: 10 (3-bit) or 8 (4-bit)


def cstride_of(cols: int, bits: int = 3) -> int:
    f = _fpw(bits)
    return (cols + f - 1) // f


def pack_counters(c: torch.Tensor, bits: int) -> torch.Tensor:
    """int8 counters [R, C] with |c| <= 2^(bits-1) - 1 -> packed fields (code = c + off) -> int32 [R, cstride]."""
    R, C = c.shape
    f, off = _fpw(bits), (1 << (bits - 1)) - 1
    assert int(c.min()) >= -off and int(c.max()) <= off, f"{bits}-bit counters hold -{off}..{off}"
    s = cstride_of(C, bits)
    codes = torch.full((R, s * f), off, dtype=torch.int64, device=c.device)
    codes[:, :C] = c.to(torch.int64) + off
    shifts = torch.arange(f, device=c.device, dtype=torch.int64) * bits
    words = (codes.view(R, s, f) << shifts).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words)  # 4-bit fields fill all 32 bits: keep the bit pattern
    return words.to(torch.int32)


def unpack_counters(Cp: torch.Tensor, cols: int, bits: int) -> torch.Tensor:
    R, s = Cp.shape
    f, off = _fpw(bits), (1 << (bits - 1)) - 1
    shifts = torch.arange(f, device=Cp.device, dtype=torch.int64) * bits
    codes = ((Cp.to(torch.int64) & 0xFFFFFFFF).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return (codes.view(R, s * f)[:, :cols] - off).to(torch.int8)


def pack_counters3(c: torch.Tensor) -> torch.Tensor:
    return pack_counters(c, 3)


def unpack_counters3(Cp: torch.Tensor, cols: int) -> torch.Tensor:
    return unpack_counters(Cp, cols, 3)


def acc_threshold(p_inc: float) -> int:
    """Integer threshold so that a vote increments when (hash >> 8) < thr, i.e. with probability p_inc."""
    assert 0.0 < p_inc <= 1.0
    return min(1 << 24, int(round(p_inc * (1 << 24))))


def sample_krows(k_frac: float, rows: int, fan_in: int, device="cuda", gen: torch.Generator | None = None):
    """Per-row pick counts with mean k_frac * fan_in: floor(e) + Bernoulli(frac(e)). Allows less than one weight per
    row on average (the plain `k` argument floors at 1)."""
    e = k_frac * fan_in
    base = math.floor(e)
    extra = (torch.rand(rows, device=device, generator=gen) < (e - base)).to(torch.int32)
    return (extra + base).to(torch.int32).contiguous()


def update(G: torch.Tensor, Wp: torch.Tensor, cnt: torch.Tensor, k: int, T: int, cache: bool | None = None,
           p_inc: float = 1.0, seed: int = 0, counter_bits: int | None = None,
           flips_out: torch.Tensor | None = None, krows: torch.Tensor | None = None) -> int | None:
    """In place: TBP top-k vote + flip-counter step on packed weights. G float32 [R, C].
    cnt: int8 [R, C] counters, or int32 packed counters with counter_bits = 3 (T <= 4) or 4 (T <= 8);
    an int32 cnt without counter_bits is read as 3-bit.
    p_inc < 1: a same-direction vote increments with probability p_inc (hash of seed, row, col).
    Returns the number of weights flipped, or, with flips_out (an int32 CUDA tensor of 1 element), adds the count
    into it on the GPU and returns None (no host sync).
    krows: optional int32 CUDA tensor [R], the number of weights to pick in each row (overrides k; see sample_krows)."""
    G = G.float().contiguous()
    R, C = G.shape
    assert Wp.shape == (R, stride_of(C))
    flips = torch.zeros(1, dtype=torch.int32, device=G.device) if flips_out is None else flips_out
    assert flips.dtype == torch.int32 and flips.numel() == 1 and flips.is_contiguous()
    use_cache = C <= CACHE_MAX_COLS and (cache is None or bool(cache))  # rows too long for shared memory: global path
    if krows is not None:
        assert krows.dtype == torch.int32 and krows.shape == (R,) and krows.is_cuda and krows.is_contiguous()
    kr = krows if krows is not None else flips  # dummy pointer when unused
    common = (cp.int32(R), cp.int32(C), cp.int32(Wp.shape[1]), cp.int32(k), cp.int32(T), _cp(flips),
              cp.int32(int(use_cache)), cp.uint32(acc_threshold(p_inc)), cp.uint32(seed & 0xFFFFFFFF),
              _cp(kr), cp.int32(int(krows is not None)))
    if cnt.dtype == torch.int8:
        assert cnt.shape == (R, C)
        _k("tbp_update")((R,), (256,), (_cp(G), _cp(Wp), _cp(cnt)) + common, shared_mem=4 * C if use_cache else 0)
    else:
        bits = counter_bits or 3
        assert cnt.dtype == torch.int32 and cnt.shape == (R, cstride_of(C, bits)), (cnt.shape, bits)
        assert 1 <= T <= (1 << (bits - 1)), f"{bits}-bit counters support T <= {1 << (bits - 1)}"
        _k(f"tbp_update{bits}")((R,), (256,), (_cp(G), _cp(Wp), _cp(cnt), cp.int32(cnt.shape[1])) + common,
                                shared_mem=4 * C if use_cache else 0)
    return None if flips_out is not None else int(flips.item())
