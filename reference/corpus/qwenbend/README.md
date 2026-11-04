# The QwenBend corpus

200 small Bend programs written by Qwen3.5-0.8B for arithmetic tasks (`sum_const`,
`gcd`, `digit_sum`, `triangle`, `count_multiples`, `digit_count`, ...), each one
checked with **real Bend2** by its author, together with

- the task it was written for,
- the expected accept/reject verdict under Bend2's static rules,
- the value the accepted ones compute,
- the first diagnostic line the rejected ones get.

`reference/test_corpus.py` runs this file through the Bend this repository pins
and compares all four.

## Where it comes from

- Source: `https://github.com/voidash/qwenbend`, `bendtune/logs/tuned400_test.jsonl`,
  cloned 2026-11-03 (`main`, depth 1). The file here is that file's 200 rows with
  `raw`, `gen_seconds` and the duplicated `correct` flag... kept as they were and
  only the fields above retained, in the original order.
- Pinned by `programs.jsonl`'s sha256, `cf97d8a3128778eda286c7b53a173c02a55d3f72221610233f9403c4f206df72`: re-projecting the upstream
  file with the snippet below reproduces these bytes (checked 2026-11-03).
- Licence: **Apache-2.0** (the upstream repository). This repository is MIT; this
  directory is Apache-2.0 and says so here, which is what the licence asks for.
  The programs are data, not code from that project; nothing else of QwenBend is
  vendored.
- What the corpus is for, upstream: measuring a LoRA writer against real Bend2.
  What it is for here: an **outside** conformance corpus for our compiler and its
  diagnostics. We did not write these programs, which is the point.

## What it agrees with, measured

On 2026-11-03, `bend 2.0.38`:

| check | result |
|---|---|
| verdict: accept exactly the programs the author's Bend2 accepted | 116 / 116 |
| verdict: reject exactly the ones it rejected | 84 / 84 |
| value: the accepted programs run to the same number | 116 / 116 |
| diagnostic: a rejection's first rule line is the same | 84 / 84 |

That agreement is why the file is worth keeping: it turns "our checker probably
behaves like the language" into a gate that fails when a Bend bump changes the
static rules or the diagnostics.

## Refreshing it

```sh
git clone --depth 1 https://github.com/voidash/qwenbend /tmp/qwenbend
python3 - <<'EOF'
import json
rows = [json.loads(l) for l in open('/tmp/qwenbend/bendtune/logs/tuned400_test.jsonl')]
keep = ('i', 'family', 'held_out', 'task', 'expected', 'program', 'valid', 'output', 'error', 'correct')
with open('reference/corpus/qwenbend/programs.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) + '\n')
EOF
```

If a refresh disagrees with `reference/test_corpus.py`, the disagreement is the
finding: either our Bend moved or theirs did, and the row says which program.
