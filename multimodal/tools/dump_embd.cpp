// dump mtmd image embeddings: dump_embd <llm.gguf> <mmproj.gguf> <image> <out.bin> [threads]
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <cstdio>
#include <cstdlib>
#include <string>
int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage\n"); return 1; }
    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    auto cp = mtmd_context_params_default();
    cp.use_gpu = false; cp.n_threads = argc > 5 ? atoi(argv[5]) : 4; cp.warmup = false; cp.print_timings = false;
    mtmd_context * ctx = mtmd_init_from_file(argv[2], model, cp);
    if (!ctx) return 3;
    auto bw = mtmd_helper_bitmap_init_from_file(ctx, argv[3], false);
    if (!bw.bitmap) return 4;
    std::string prompt = std::string(mtmd_default_marker());
    mtmd_input_text txt = { prompt.c_str(), false, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bms[1] = { bw.bitmap };
    if (mtmd_tokenize(ctx, chunks, &txt, bms, 1) != 0) return 5;
    const int n_embd = llama_model_n_embd_inp(model);
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        if (mtmd_encode_chunk(ctx, ch) != 0) return 6;
        size_t nt = mtmd_input_chunk_get_n_tokens(ch);
        float * e = mtmd_get_output_embd(ctx);
        FILE * f = fopen(argv[4], "wb");
        int32_t hdr[2] = { (int32_t) nt, n_embd };
        fwrite(hdr, sizeof(hdr), 1, f);
        fwrite(e, sizeof(float), nt * n_embd, f);
        fclose(f);
        fprintf(stderr, "dumped %zu x %d\n", nt, n_embd);
    }
    return 0;
}
