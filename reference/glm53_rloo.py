"""RLOO post-training for the vendored GLM-5.3-Flash reference.

  reference/.venv-rf/bin/python reference/glm53_rloo.py \
      --checkpoint /tmp/opencode/glm53/ref-pretrain-3e4/checkpoint-0100 \
      --output results/glm53-reference/rloo

The reference's report describes the setup: 96 prompts across `increment`, `double` and `even`,
16 sampled completions each, executable reward (1 passed, 0 valid but wrong, -0.1 invalid), and
RLOO (each sample against the mean of the other fifteen). Its RL code is not in the vendored tree,
so this is an implementation from that description, starting from the selected pretraining
checkpoint. One optimizer step per prompt; a prompt whose sixteen rewards are all equal carries no
signal and is skipped.

`--updates 96` is the reference's 1,536 rollouts. Dev (three families, greedy) is scored every
`--eval-every` updates and the final checkpoint is evaluated on confirm, greedy and pass@8.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/glm53_flash"))

from glm53_flash.evaluator import evaluate_source  # noqa: E402
from glm53_flash.model import GLM53FlashFromScratch  # noqa: E402
from glm53_flash.runtime import generate_group, load_checkpoint, save_checkpoint  # noqa: E402
from glm53_flash.tasks import FAMILY_BY_NAME, make_task  # noqa: E402
from glm53_flash.tokenizer import ByteTokenizer  # noqa: E402

FAMILIES = ("increment", "double", "even")
RL_SEED = 3907


def rl_tasks(families, per_family=32):
    return [make_task(FAMILY_BY_NAME[family], split="rl", index=index, seed=RL_SEED)
            for family in families for index in range(per_family)]


def dev_tasks(families, per_family=8):
    return [make_task(FAMILY_BY_NAME[family], split="dev", index=index, seed=1701)
            for family in families for index in range(per_family)]


def rewards(task, tokenizer, completions):
    out = []
    for item in completions:
        evaluation = evaluate_source(task, task.prompt + item["completion"])
        if evaluation.passed:
            out.append(1.0)
        elif evaluation.status == "invalid":
            out.append(-0.1)
        else:
            out.append(0.0)
    return out


def log_probs(model, tokenizer, prompt_ids, completions, temperature, device):
    """Sum of log p(token) for each completion under the tempered policy."""
    sequences = [prompt_ids + item["token_ids"] for item in completions]
    width = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), width), tokenizer.pad_id, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    ids = ids.to(device)
    logits, _ = model(ids)
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    totals = []
    for row, sequence in enumerate(sequences):
        start = len(prompt_ids)
        positions = torch.arange(start - 1, len(sequence) - 1, device=device)
        tokens = torch.tensor(sequence[start:], dtype=torch.long, device=device)
        totals.append(logp[row, positions, tokens].sum())
    return torch.stack(totals)


def rloo_advantages(reward_values):
    values = torch.tensor(reward_values, dtype=torch.float32)
    if float(values.max() - values.min()) == 0.0:
        return None
    leave_one_out = (values.sum() - values) / (len(values) - 1)
    return values - leave_one_out


@torch.no_grad()
def score_split(model, tokenizer, tasks, device, max_new, sample, group, temperature, seed):
    model.eval()
    solved = tests_passed = tests_total = valid = 0
    for index, task in enumerate(tasks):
        generated = generate_group(model, tokenizer, task, group_size=group, max_new_tokens=max_new,
                                   temperature=temperature if sample else 1.0, sample=sample,
                                   seed=seed + index)
        passed_any = False
        for item in generated:
            evaluation = evaluate_source(task, task.prompt + item["completion"])
            passed_any = passed_any or evaluation.passed
            valid += int(evaluation.status != "invalid")
            tests_passed += evaluation.tests_passed
            tests_total += evaluation.tests_total
        solved += int(passed_any)
    return {"solved": solved, "tasks": len(tasks), "tests_passed": tests_passed,
            "tests_total": tests_total, "valid": valid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=96)
    parser.add_argument("--group", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--eval-pass-at", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = ByteTokenizer()
    model = load_checkpoint(args.checkpoint, device)
    model.train()
    prompts = rl_tasks(FAMILIES)
    dev = dev_tasks(FAMILIES)
    confirm = dev_tasks(FAMILIES)
    all_eight = [make_task(FAMILY_BY_NAME[family], split="confirm", index=index, seed=8123)
                 for family in FAMILY_BY_NAME for index in range(8)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    history = []
    reward_updates = 0
    started = time.perf_counter()
    for step, task in enumerate(prompts[:args.updates], 1):
        model.eval()
        generated = generate_group(model, tokenizer, task, group_size=args.group,
                                   max_new_tokens=args.max_new, temperature=args.temperature,
                                   sample=True, seed=args.seed + step * 1000)
        reward_values = rewards(task, tokenizer, generated)
        advantages = rloo_advantages(reward_values)
        gained = sum(1 for value in reward_values if value > 0)
        row = {"step": step, "task": task.task_id, "solved": gained,
               "reward_mean": sum(reward_values) / len(reward_values),
               "reward_updates": int(advantages is not None)}
        if advantages is not None:
            reward_updates += row["reward_updates"]
            model.train()
            prompt_ids = tokenizer.encode(task.prompt, bos=True)
            totals = log_probs(model, tokenizer, prompt_ids, generated, args.temperature, device)
            loss = -(advantages.to(device) * totals).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            row["loss"] = float(loss.detach())
        row["seconds"] = round(time.perf_counter() - started, 1)
        history.append(row)
        print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == len(prompts[:args.updates]):
            model.eval()
            dev_score = score_split(model, tokenizer, dev, device, args.max_new, False, 1, 1.0,
                                    args.seed)
            row["dev"] = dev_score
            history[-1] = row
            print(json.dumps({"step": step, "dev": dev_score}), flush=True)
            if step % args.save_every == 0 or step == len(prompts[:args.updates]):
                save_checkpoint(model, args.output / f"checkpoint-{step:04d}",
                                {"stage": "rloo", "step": step, "dev": dev_score, "args": vars(args) | {"checkpoint": str(args.checkpoint), "output": str(args.output)}})

    model.eval()
    final = args.output / f"checkpoint-{args.updates:04d}"
    if not final.exists():
        save_checkpoint(model, final, {"stage": "rloo", "step": args.updates, "args": vars(args) | {"checkpoint": str(args.checkpoint), "output": str(args.output)}})
    summary = {
        "updates": args.updates, "rollouts": args.updates * args.group,
        "reward_updates": reward_updates, "lr": args.lr, "temperature": args.temperature,
        "group": args.group, "start_checkpoint": str(args.checkpoint),
        "dev_three_families_greedy": score_split(model, tokenizer, dev, device, args.max_new,
                                                 False, 1, 1.0, args.seed),
        "confirm_three_families_greedy": score_split(model, tokenizer, confirm, device, args.max_new,
                                                     False, 1, 1.0, args.seed),
        "confirm_all_eight_greedy": score_split(model, tokenizer, all_eight, device, args.max_new,
                                                False, 1, 1.0, args.seed),
        "confirm_all_eight_pass8": score_split(model, tokenizer, all_eight, device, args.max_new,
                                               True, args.eval_pass_at, 0.8, args.seed),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "history": history,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in summary if key != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
