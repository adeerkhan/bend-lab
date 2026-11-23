# MLBend: setup and verification shortcuts. Run `make setup` once, then `make check`.
export BEND_NO_TELEMETRY := 1
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python
BENDENV := $(HOME)/.bend/bin/bendenv
BEND := $(or $(BEND_BIN),$(HOME)/.bend/bin/bend)

BEND_VERSION := 2.0.38
BEND_REF := ../bend-ref

.PHONY: setup check check-dit check-dit-denoiser laws sync-bend clean lint env env-check release experiment experiments results report trace dash dashboard session unet-metric unet-record unet-record-four unet-sample mamba3-bench mamba3-layer mamba3-step mamba3-train-curve mamba3-varlen mamba3-gqa mamba3-outnorm mamba3-deep mamba3-lanes mamba-bf16-curve mamba3-cases mamba3-fd mamba3-state mamba3-wrap mamba3-clamps mamba3-canfail mamba3-det record-compat oracle-boundary lane-matrix pages mamba3-tiny-data mamba-vit-emb mamba-vit-fwd mamba-vit-bwd mamba-vit-train glm-step glm-block-step glm-lin-block glm-decode glm-h192-step batch peaks bounds bench plumbing bench-mamba-tiny bench-ab bend-smoke bend-side bend-latest flow flow64 doctor figures select plots sweep sweep-agent

setup:            ## Bend, Lean and the Python venv
	scripts/setup.sh

check:            ## the proofs, the kernel (--verdict), the rejected programs, the NumPy comparison
	$(PY) reference/check_all.py

check-fast:       ## the proofs and the frozen references only: the cheap gate, under a minute
	$(PY) reference/check_all.py --fast

check-dit:        ## the Bendlang DiT block: attention and LN parity, frozen full-block gap (minutes)
	$(PY) reference/test_dit_ref.py

check-dit-denoiser: ## the Bendlang DiT denoiser trains: loss trajectory against the torch twin
	$(PY) reference/test_dit_denoiser.py

versions:         ## the Bend pin, bend.toml, .bend/env and the imports, side by side
	$(PY) reference/versions.py

mamba3:           ## the Mamba-3 SISO gates: forward, backward, training, generation
	$(PY) reference/check_mamba3_bend.py

mamba3-bench:     ## twenty forward passes at seqlen 8: the scan benchmark
	@mkdir -p .scratch
	scripts/bend-guard.sh $(BEND) -o .scratch/mamba3_fwd.bin examples/language/mamba3/bench_mamba3_fwd_n.bend
	/usr/bin/time -f "bench mamba3: %e s wall, %U s user" .scratch/mamba3_fwd.bin

mamba3-layer:     ## the SISO block as a BendTorch T.Layer, trained with T.Train
	$(PY) reference/test_mamba3_layer.py

mamba3-step:       ## the Mamba-3 decode step: one token at a time, parity with the full-prefix pass
	$(PY) reference/test_mamba3_step.py

mamba3-train-curve: ## the ten-step tiny train, recorded curve, custom vs T.Train
	$(PY) reference/test_mamba3_train_curve.py

mamba3-varlen:     ## packed sequences: the state resets at each segment boundary
	$(PY) reference/test_mamba3_varlen.py

mamba3-gqa:        ## ngroups 2 + rope_fraction 1.0: forward and backward
	$(PY) reference/test_mamba3_gqa.py

mamba3-outnorm:    ## is_outproj_norm: the gated grouped RMSNorm after the block
	$(PY) reference/test_mamba3_outnorm.py

mamba3-deep:       ## depth-4 stack: forward, ten-step backward and generation
	$(PY) reference/test_mamba3_deep.py

mamba3-lanes:      ## numerical lanes: the SISO block through F32/TF32/bf16, vs the twin
	$(PY) reference/test_mamba3_lanes.py

mamba-bf16-curve:  ## the tiny-LM 20-step curve in the F32 and bf16 lanes, side by side
	$(PY) reference/test_mamba3_bf16_curve.py

mamba3-cases:      ## the kernel's explanatory cases: T = 1 skip, T = 2 state, T = 64 chunk
	$(PY) reference/test_mamba3_cases.py

mamba3-fd:         ## finite differences: the scan's y-gradient at seqlen 4 vs the bumped-input twin
	$(PY) reference/test_mamba3_fd.py

mamba3-state:      ## the exact-step decoder's state round-trip: T + k vs one 30-step run
	$(PY) reference/test_mamba3_state.py

mamba3-wrap:       ## angle wrap: a synthetic sequence crossing 2pi, exact loop and full-prefix vs the twin
	$(PY) reference/test_mamba3_wrap.py

mamba3-clamps:     ## clamp boundaries: A at -A_floor, trap at 0/1, softplus at large/small dd_dt
	$(PY) reference/test_mamba3_clamps.py

mamba3-canfail:    ## can-fails: missing shifted, wrong rope sign, zeroed D — each must FAIL its gate
	$(PY) reference/test_mamba3_canfail.py

mamba3-det:        ## determinism: two builds, identical seqlen-8 forward checksums
	$(PY) reference/test_mamba3_det.py

record-compat:     ## old and new run records both render the reader's page
	$(PY) reference/test_record_compat.py

oracle-boundary:   ## no oracle in the run path (static check + committed records)
	$(PY) reference/test_oracle_boundary.py

lane-matrix:       ## the task/lane compliance page, from committed records only
	$(PY) reference/lane_matrix.py

pages:             ## render every task's latest run page from its committed record
	@mkdir -p results/pages
	@for f in results/latest/*.json; do t=$${f##*/}; t=$${t%.json}; \
	  $(PY) reference/report_run.py --task $$t > results/pages/$$t.txt || exit 1; \
	done
	@echo "rendered results/pages/*.txt"

mamba3-tiny-data:  ## the tiny-LM corpus subset and its Bend read-back (trainer pending)
	$(PY) reference/test_mamba3_tiny_lm.py

mamba3-lm-tgt:     ## Mamba-3 tiny-LM B1: next-token loss/dy/dWh vs the NumPy twin
	$(PY) reference/test_mamba3_lm_tgt.py

mamba3-tiny-opt:   ## Mamba-3 tiny-LM B2: AdamW + clipping + warmup/cosine vs the twin
	$(PY) reference/test_mamba3_tiny_opt.py

mamba3-tiny-lm:    ## Mamba-3 tiny-LM B3: the held-out run — bits/byte vs the twin and the bigram
	$(PY) reference/test_mamba3_tiny_lm_run.py

mamba3-step-varlen: ## Mamba-3 decode breadth C1: the step at the packed-prompt boundary
	$(PY) reference/test_mamba3_step_varlen.py

mamba3-step-gqa:    ## Mamba-3 decode breadth C2: the step at ngroups 2, rope_fraction 1.0
	$(PY) reference/test_mamba3_step_gqa.py

mamba3-step-outnorm: ## Mamba-3 decode breadth C3: the step with is_outproj_norm
	$(PY) reference/test_mamba3_step_on.py


mamba-vit-emb:     ## MambaViT A1a: patch embedding + tokens vs the reference dump
	$(PY) reference/test_mamba_vit.py

mamba-vit-fwd:     ## MambaViT A1b: the whole forward, eight blocks to the logits
	$(PY) reference/test_mamba_vit.py

mamba-vit-bwd:     ## MambaViT A2: the hand-written backward, every gradient at 1e-4
	$(PY) reference/test_mamba_vit_bwd.py

mamba-vit-train:   ## MambaViT A3: subset training parity, Bend vs the twin (~31 min measured, 1869.8 s wall)
	$(PY) reference/test_mamba_vit_train.py

glm-step:         ## the 192-position attention step: state and cache against the full prefix
	@mkdir -p .scratch
	scripts/bend-guard.sh $(BEND) -o .scratch/glm_step.bin examples/language/glm53_flash/probe_glm_step.bend
	@.scratch/glm_step.bin | tee .scratch/glm_step.log
	@grep -q "GLM STEP PASS" .scratch/glm_step.log

glm-block-step:   ## the fourth block's step (attention and MLP) on the trained weights
	@mkdir -p .scratch
	scripts/bend-guard.sh $(BEND) -o .scratch/glm_block.bin examples/language/glm53_flash/probe_glm_block.bend
	@.scratch/glm_block.bin | tee .scratch/glm_block.log
	@grep -q "GLM BLOCK STEP PASS" .scratch/glm_block.log

glm-lin-block:    ## the linear block's step (attention and MLP) on the trained weights
	@mkdir -p .scratch
	scripts/bend-guard.sh $(BEND) -o .scratch/glm_lin_block.bin examples/language/glm53_flash/probe_glm_block_lin.bend
	@.scratch/glm_lin_block.bin | tee .scratch/glm_lin_block.log
	@grep -q "GLM LIN BLOCK PASS" .scratch/glm_lin_block.log

glm-decode:       ## the four-block step decoder: episode 0 tokens against the full prefix
	@mkdir -p .scratch
	scripts/bend-guard.sh $(BEND) -o .scratch/step_h192.bin examples/language/glm53_flash/step_h192.bend
	@.scratch/step_h192.bin --episodes 1 | tee .scratch/step_h192.log
	@$(PY) reference/glm53_step_gen_check.py .scratch/step_h192.log

batch:            ## the streamlined matrix: scripts/batch.sh (ROWS=... for a subset)
	scripts/batch.sh

glm-h192-step:    ## the full hybrid 192 dev and confirm evaluation with the step decoder
	@mkdir -p .scratch
	scripts/bend-guard.sh $(BEND) -o .scratch/step_h192.bin examples/language/glm53_flash/step_h192.bend
	$(PY) reference/glm53_dense_eval.py --weights results/glm53-hybrid128/weights/final_weights.bin --out results/glm53-hybrid192-step --splits dev,confirm --episodes 64 --layout hybrid --binary .scratch/step_h192.bin --golden ''

unet-metric:      ## the UNet held-out gate: legibility and spectrum, with can-fail arms
	$(PY) reference/test_unet_flowers_metric.py

unet-record:      ## register the trained UNet's gate result as a run record with a manifest
	$(PY) reference/unet_record.py

unet-record-four: ## register a --four run (UNET_RUN=... UNET_TASK=...)
	$(PY) reference/unet_record.py

unet-sample:    ## inference only: 64 Euler draws from a checkpoint (CKPT=...), then the gate
	bash scripts/unet_sample.sh

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
	@test -n "$(TASK)" || (echo "usage: make trace TASK=<task> [STEPS=<n>] [MAX_SECONDS=3600]" && false)
	$(PY) reference/trace.py --task $(TASK) $(if $(STEPS),--steps $(STEPS),) $(if $(MAX_SECONDS),--max-seconds $(MAX_SECONDS),)

dashboard:        ## the browser dashboard (PORT=8787, TASK=<fallback>, OPEN=1)
	$(PY) reference/dashboard.py --port $(or $(PORT),8787) $(if $(TASK),--task $(TASK),) $(if $(OPEN),--open,)

session:          ## run any command as a dashboard session (TASK=<name> CMD="...")
	@test -n "$(TASK)" -a -n "$(CMD)" || (echo 'usage: make session TASK=<name> CMD="make check-fast"' && false)
	$(PY) reference/session.py --task $(TASK) -- $(CMD)

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
	BEND_MEM=$(or $(MEM),4G) scripts/bend-guard.sh $(BEND) -o .scratch/call_cost bench/call_cost/main.bend
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

bench-mamba-tiny: ## one line: ms per step by the program's own clock, failing on changed outputs
	scripts/bench_mamba_tiny.sh

bend-smoke:       ## is the installed Bend whole? compiles a live F32.from_nat
	scripts/bend_smoke.sh

bend-side:        ## install a Bend release beside the pin, V=<version> [SHA=<sha256>]
	scripts/bend-side.sh $(V) $(SHA)

bend-latest:      ## is there a Bend newer than the pin? INSTALL=1 tests it side by side
	scripts/bend-latest.sh $(if $(INSTALL),--install,)

bench-ab:         ## A/B two plumbing binaries, shuffled rounds, refused under load (A= B= ROUNDS=)
	scripts/bench_ab.sh $(A) $(B) $(or $(ROUNDS),5)

figures:          ## rebuild the README figures from results/, saved under docs/assets/figures/
	$(PY) reference/make_readme_figures.py

flow:             ## train the 2D flow matching example, saved under results/flow-matching/
	@mkdir -p .scratch
	scripts/bend-guard.sh env BEND_LIB=$(CURDIR)/.bend/env $(BEND) -o .scratch/flow.bin examples/diffusion/flow_matching/main.bend
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
	BEND_MEM=$(or $(MEM),4G) scripts/bend-guard.sh env BEND_LIB=$(CURDIR)/.bend/env $(BEND) -o .scratch/flow64 examples/diffusion/flowers_flow/main.bend
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
