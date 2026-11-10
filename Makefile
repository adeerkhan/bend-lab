# MLBend: setup and verification shortcuts. Run `make setup` once, then `make check`.
export BEND_NO_TELEMETRY := 1
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python
BENDENV := $(HOME)/.bend/bin/bendenv
BEND := $(or $(BEND_BIN),$(HOME)/.bend/bin/bend)

BEND_VERSION := 2.0.38
BEND_REF := ../bend-ref

.PHONY: setup check laws sync-bend clean lint env env-check release experiment experiments results report trace dash peaks bounds bench plumbing flow flow64 doctor figures select plots sweep sweep-agent

setup:            ## Bend, Lean and the Python venv
	scripts/setup.sh

check:            ## the proofs, the kernel (--verdict), the rejected programs, the NumPy comparison
	$(PY) reference/check_all.py

check-fast:       ## the proofs and the frozen references only: the cheap gate, under a minute
	$(PY) reference/check_all.py --fast

validate:         ## one gate: env-check, the cheap gate, the full gate, the smoke (QUICK=1, LONG=1)
	QUICK=$(QUICK) LONG=$(LONG) scripts/validate.sh $(ARGS)

laws:             ## print every law statement of every package
	$(PY) reference/list_laws.py

env:              ## the environment: one lock, one env, for every package here
	$(BENDENV) lock
	$(BENDENV) sync

env-check:        ## the environment against the lock, and the example over it
	$(BENDENV) check
	$(BENDENV) run -- $(BEND) examples/transformers/attention/main.bend

experiment:       ## run one task and write a report card: make experiment TASK=iris-crossval
	@test -n "$(TASK)" || (echo "usage: make experiment TASK=<task> (see reference/experiment.py --list)" && false)
	$(PY) reference/experiment.py --task $(TASK) $(if $(TRAINER),--trainer $(TRAINER),) $(if $(REFERENCES),--references $(REFERENCES),)

experiments:      ## every task, with its report cards, saved under results/
	$(PY) reference/experiment.py --all --repeat 2

results:          ## regenerate results/RESULTS.md from the latest reports
	$(PY) reference/experiment.py --summary

report:           ## the page of a task's latest record; AGAINST=<record> compares two runs
	@test -n "$(TASK)" || (echo "usage: make report TASK=<task> [AGAINST=<record>]" && false)
	$(PY) reference/report_run.py --task $(TASK) $(if $(AGAINST),--against $(AGAINST),)

trace:            ## one task's program with --trace: one JSON line per step, written as it arrives
	@test -n "$(TASK)" || (echo "usage: make trace TASK=<task> [STEPS=<n>]" && false)
	$(PY) reference/trace.py --task $(TASK) $(if $(STEPS),--steps $(STEPS),)

dash:             ## the live page of a task's trace (WATCH=1, PID=<pid> for memory)
	@test -n "$(TASK)" || (echo "usage: make dash TASK=<task> [WATCH=1] [PID=<pid>]" && false)
	$(PY) reference/dash.py --task $(TASK) $(if $(WATCH),--watch,) $(if $(PID),--sample-pid $(PID),)

references:       ## freeze every task's reference numbers under results/reference/
	$(PY) reference/references.py --freeze --all $(if $(REFRESH),--refresh-references,)

references-refresh: ## recompute the frozen references with the live functions (runs the corpus)
	$(PY) reference/experiment.py --all --freeze-references --refresh-references

prune:            ## keep the newest records per task and everything RESULTS.md cites (PRUNE_DRY=1 to look)
	$(PY) reference/prune.py $(if $(PRUNE_DRY),--dry-run,) $(if $(KEEP),--keep $(KEEP),)

matrix:           ## run the declared cells (task, lane, seed), one record each, and a summary
	$(PY) reference/matrix.py $(if $(LIVE),--live,) $(if $(TASK),--task $(TASK),)

call-cost:        ## what a small matmul costs, and what its call costs, at the ViT's shapes
	@mkdir -p .scratch
	BEND_MEM=$(or $(MEM),6G) scripts/bend-guard.sh $(BEND) -o .scratch/call_cost bench/call_cost/main.bend
	.scratch/call_cost

peaks:            ## the machine's measured ceilings (read, write, copy, flops), saved under results/
	$(PY) reference/bench_peaks.py

bounds:           ## the floor of a task's stages, saved as results/bounds-<task>.json
	@test -n "$(TASK)" || (echo "usage: make bounds TASK=<task> (see reference/bounds.py --list)" && false)
	$(PY) reference/bounds.py --task $(TASK)

bench:            ## a compute-only kernel benchmark, saved under results/
	$(PY) reference/bench_matmul.py

plumbing:         ## the time split of one ViT training step, saved under results/
	$(PY) reference/bench_plumbing.py

figures:          ## rebuild the README figures from results/, saved under docs/assets/figures/
	$(PY) reference/make_readme_figures.py

flow:             ## train the 2D flow matching example, saved under results/flow-matching/
	@mkdir -p .scratch
	BEND_LIB=$(CURDIR)/.bend/env $(BEND) -o .scratch/flow.bin examples/diffusion/flow_matching/main.bend
	.scratch/flow.bin > .scratch/flow.log
	$(PY) reference/flow_results.py .scratch/flow.log

doctor:           ## what this machine can compile: RAM, cores, GPU, build budget, flow preset
	$(PY) scripts/system-profile.py

# The 64x64 flow: pick the preset the machine can pay for, build it under a
# memory cgroup, run it, and report the loss and the samples. MEM is the
# cgroup's ceiling: the build dies at it and the desktop does not.
flow64:           ## build and run the 64x64 flow (PRESET=tiny MEM=6G)
	@mkdir -p .scratch
	$(PY) scripts/flow-config.py --preset $(or $(PRESET),tiny)
	BEND_MEM=$(or $(MEM),6G) scripts/bend-guard.sh env BEND_LIB=$(CURDIR)/.bend/env $(BEND) -o .scratch/flow64 examples/diffusion/flowers_flow/main.bend
	.scratch/flow64 > .scratch/flow64.log
	$(PY) reference/flowers_report.py .scratch/flow64.log


select:           ## model selection and a classification report (kNN)
	$(PY) reference/select_models.py

plots:            ## write examples/<name>/README.md and plot.svg for every example
	$(PY) reference/make_example_readmes.py

issues:          ## render .local/issues-hit.md as .local/issues-hit.html (bend-lang theme)
	$(PY) reference/make_issues_html.py

sweep:            ## start the W&B sweep defined in reference/wandb_sweep.yaml
	$(PY) -m wandb sweep reference/wandb_sweep.yaml

sweep-agent:      ## run an agent for a started sweep: make sweep-agent SWEEP=<id>
	@test -n "$(SWEEP)" || (echo "usage: make sweep-agent SWEEP=<id>" && false)
	$(PY) -m wandb agent $(SWEEP)

release:          ## release one package to the hub and tag it here: make release PKG=hf
	@test -n "$(PKG)" || (echo "usage: make release PKG=<package directory>" && false)
	scripts/release.sh $(PKG)

sync-bend:       ## a read-only copy of the pinned Bend source, for reading the compiler
	rm -rf $(BEND_REF)
	git clone --depth 1 --branch v$(BEND_VERSION) https://github.com/bendlang/bend.git $(BEND_REF)

clean:
	rm -rf reference/__pycache__ .scratch

lint:             ## bend-lint's performance rules over the Bend code: a report, not a gate
	bunx bend-lint --jobs max
