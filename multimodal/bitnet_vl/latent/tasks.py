"""Synthetic variable-tracking programs.

Example (hops=3, distractors=2)::

    q = 7
    x = 4
    m = q + 5
    t = m * 3
    k = x - 1
    p = t - q
    print(p)

* The *chain* is ``root = c; v1 = root op o1; v2 = v1 op o2; ...``; ``hops`` = number of computed chain assignments
  (= number of trace steps). Operands are a constant 1..9 or (sometimes) an earlier chain variable.
* *Distractors* are assignments not on the chain (constants or expressions of earlier variables), interleaved at
  random positions (always after the variables they use).
* All intermediate values stay in [0, 99] (operations that would leave the range are re-sampled), so the program is
  plain Python and answers have at most 2 digits; ``exec`` of the program gives the same values (tested).
* ``trace_steps`` are the evaluated chain assignments, e.g. ``["m = 12", "t = 36", "p = 29"]``.

Everything is deterministic given the seed. `TaskSampler` streams training examples; `make_test_set` builds a
fixed test set from a disjoint seed space.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

OPS = ("+", "-", "*")
MAX_VAL = 99
TEST_SEED_OFFSET = 1_000_003  # test streams use seeds disjoint from the training stream


def _apply(a: int, op: str, b: int) -> int:
    return a + b if op == "+" else a - b if op == "-" else a * b


def _sample_op(rng: random.Random, value: int, operands: list[tuple[str, int]]) -> tuple[str, str, int]:
    """Pick (op, operand_text, result) with result in [0, MAX_VAL]. operands: available (name, value) vars."""
    for _ in range(50):
        op = rng.choice(OPS)
        if operands and rng.random() < 0.3:
            name, val = rng.choice(operands)
        else:
            val = rng.randint(1, 9) if op != "*" else rng.randint(2, 5)
            name = str(val)
        r = _apply(value, op, val)
        if 0 <= r <= MAX_VAL:
            return op, name, r
    return ("+", "1", value + 1) if value < MAX_VAL else ("-", "1", value - 1)


def make_example(rng: random.Random, hops: int, distractors: int = 0) -> dict:
    """One program. Returns dict(prompt, answer, trace_steps, hops, program_lines, query)."""
    if hops < 1:
        raise ValueError("hops >= 1")
    n_vars = hops + 1 + distractors
    if n_vars > 26:
        raise ValueError("too many variables for single-letter names")
    names = list(string.ascii_lowercase)
    rng.shuffle(names)
    chain_names, dist_names = names[:hops + 1], names[hops + 1:n_vars]

    # chain
    root_val = rng.randint(0, 20)
    chain_lines = [f"{chain_names[0]} = {root_val}"]
    values = {chain_names[0]: root_val}
    trace = []
    cur = root_val
    for k in range(1, hops + 1):
        prev = chain_names[k - 1]
        earlier = [(n, values[n]) for n in chain_names[:k - 1]]
        op, operand, cur = _sample_op(rng, cur, earlier)
        line = f"{chain_names[k]} = {prev} {op} {operand}"
        values[chain_names[k]] = cur
        chain_lines.append(line)
        trace.append(f"{chain_names[k]} = {cur}")

    # interleave distractors: choose insertion slots among the chain lines (slot i = before chain line i, or end)
    lines: list[str] = list(chain_lines)
    defined_at = {n: i for i, n in enumerate(chain_names)}  # index in `lines` of definition
    dvals: dict[str, int] = {}
    for d in dist_names:
        pos = rng.randint(0, len(lines))  # insert before lines[pos]
        avail = [(n, values.get(n, dvals.get(n))) for n, i in defined_at.items() if i < pos]
        if avail and rng.random() < 0.6:
            src, sval = rng.choice(avail)
            others = [a for a in avail if a[0] != src]
            op, operand, val = _sample_op(rng, sval, others)
            line = f"{d} = {src} {op} {operand}"
        else:
            val = rng.randint(0, 20)
            line = f"{d} = {val}"
        lines.insert(pos, line)
        for n in defined_at:
            if defined_at[n] >= pos:
                defined_at[n] += 1
        defined_at[d] = pos
        dvals[d] = val
    query = chain_names[-1]
    lines.append(f"print({query})")
    prompt = "\n".join(lines) + "\n"
    return dict(prompt=prompt, answer=values[query], trace_steps=trace, hops=hops, program_lines=lines, query=query)


def exec_program(prompt: str) -> int:
    """Reference answer by executing the program with print captured."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(prompt, {"__builtins__": {"print": print}}, {})  # noqa: S102 (trusted generated code)
    return int(buf.getvalue().strip())


def trace_values(prompt: str, trace_steps: list[str]) -> bool:
    """Check that each trace line 'v = n' matches the value of v after executing the program."""
    env: dict = {}
    exec(prompt.replace("print(", "_q = ("), {"__builtins__": {}}, env)  # noqa: S102
    for t in trace_steps:
        name, val = (s.strip() for s in t.split("="))
        if env[name] != int(val):
            return False
    return True


def parse_hops(spec: str | list | tuple) -> list[int]:
    """'2-6' -> [2..6]; '2,4,8' -> [2,4,8]; list passthrough."""
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    spec = str(spec).strip()
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


@dataclass
class TaskSampler:
    """Infinite deterministic stream of training examples with hops uniform over `hops`."""

    hops: list[int]
    distractors: int = 0
    seed: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def sample(self) -> dict:
        return make_example(self.rng, self.rng.choice(self.hops), self.distractors)

    def batch(self, n: int) -> list[dict]:
        return [self.sample() for _ in range(n)]


def make_test_set(hops: list[int], n_per_hops: int, distractors: int = 0, seed: int = 0) -> dict[int, list[dict]]:
    """Fixed test examples per hop count, from a seed space disjoint from the training stream."""
    out = {}
    for h in hops:
        rng = random.Random(TEST_SEED_OFFSET + 7919 * h + seed)
        out[h] = [make_example(rng, h, distractors) for _ in range(n_per_hops)]
    return out
