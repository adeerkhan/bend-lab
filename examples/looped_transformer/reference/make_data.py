#!/usr/bin/env python3
"""Deterministic 16-note chain-puzzle batches for the Bend looped transformer.

The generator is a Python port of `burn-lab/looped/src/puzzle.rs`: the same
SplitMix64 stream, the same shuffle and sample rules, and the same token ids.
A training sequence is

    sixteen notes ("a=b," x 16) + one question ("q?") + the answer digit

so the model learns the answer as the next token after `?`; the last target of
the shifted sequence is the answer. Evaluation files force a chain length.

  python3 reference/make_data.py --out data --steps 500 --batch 16
"""
from __future__ import annotations

import argparse
import os

N_NOTES = 16
N_CHAINS = 4
LETTERS = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
NOTES_LEN = 4 * N_NOTES
SEQ_LEN = NOTES_LEN + 3          # notes + question letter + '?' + answer
INPUT_LEN = SEQ_LEN - 1          # shifted next-token input


class Rng:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

    def below(self, bound: int) -> int:
        return 0 if bound <= 0 else self.next_u64() % bound

    def shuffle(self, values: list) -> None:
        for index in range(len(values) - 1, 0, -1):
            other = self.below(index + 1)
            values[index], values[other] = values[other], values[index]

    def sample(self, values: list, count: int) -> list:
        pool = list(values)
        for index in range(count):
            other = index + self.below(len(pool) - index)
            pool[index], pool[other] = pool[other], pool[index]
        return pool[:count]


def split(rng: Rng, n: int, k: int) -> list[int]:
    cuts = rng.sample(list(range(1, n)), k - 1)
    cuts = [0] + cuts + [n]
    cuts.sort()
    return [b - a for a, b in zip(cuts, cuts[1:])]


def make_puzzle(rng: Rng, max_hops: int, first: int | None = None):
    notes, chains = N_NOTES, N_CHAINS
    while True:
        if first is None:
            lens = split(rng, notes, chains)
            check = lens
        else:
            lens = [first] + split(rng, notes - first, chains - 1)
            check = lens[1:]
        if max(check) <= max_hops:
            break

    names = rng.sample(list(LETTERS), notes)
    digits = rng.sample(list(DIGITS), chains)
    note_pairs = []
    qa = []
    for length, value in zip(lens, digits):
        chain = names[:length]
        names = names[length:]
        note_pairs.append((chain[0], value))
        qa.append((chain[0], value, 1))
        for i, name in enumerate(chain[1:], start=1):
            note_pairs.append((name, chain[i - 1]))
            qa.append((name, value, i + 1))
    rng.shuffle(note_pairs)
    rng.shuffle(qa)
    text = "".join(f"{a}={b}," for a, b in note_pairs)
    return text, qa


def token_id(ch: str) -> int:
    if ch in LETTERS:
        return LETTERS.index(ch)
    if ch in DIGITS:
        return 26 + DIGITS.index(ch)
    return {"=": 36, ",": 37, "?": 38}[ch]


def sequence(rng: Rng, max_hops: int, first: int | None = None) -> bytes:
    text, qa = make_puzzle(rng, max_hops, first)
    if first is None:
        question, answer, _ = qa[rng.below(len(qa))]
    else:
        eligible = [q for q in qa if q[2] == first]
        question, answer, _ = eligible[rng.below(len(eligible))]
    full = text + question + "?" + answer
    assert len(full) == SEQ_LEN, (len(full), full)
    return bytes(token_id(ch) for ch in full)


def write_step(f, seqs: list[bytes]) -> None:
    """All inputs for the step, then all next-token targets."""
    f.write(b"".join(seq[:INPUT_LEN] for seq in seqs))
    f.write(b"".join(seq[1:] for seq in seqs))


def write_train(path: str, steps: int, batch: int, seed: int, max_hops: int) -> None:
    rng = Rng(seed)
    with open(path, "wb") as f:
        for _ in range(steps):
            write_step(f, [sequence(rng, max_hops) for _ in range(batch)])
    print(f"wrote {path}: {steps} steps x {batch} sequences, {2 * INPUT_LEN} bytes each")


def write_eval(path: str, hop: int, count: int, seed: int, max_hops: int) -> None:
    rng = Rng(seed)
    with open(path, "wb") as f:
        write_step(f, [sequence(rng, max_hops=max_hops, first=hop) for _ in range(count)])
    print(f"wrote {path}: {count} sequences at hop {hop}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-count", type=int, default=8)
    ap.add_argument("--max-hops", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    write_train(os.path.join(args.out, "train.bin"), args.steps, args.batch,
                args.seed, args.max_hops)
    for hop in range(1, 13):
        write_eval(os.path.join(args.out, f"eval-hop-{hop}.bin"), hop,
                   args.eval_count, args.seed + hop, args.max_hops)


if __name__ == "__main__":
    main()
