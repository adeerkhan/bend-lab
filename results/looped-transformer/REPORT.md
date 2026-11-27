# Looped transformer, Bend v0

Local measurement. The model is `examples/looped_transformer/main.bend`; the
generator is `examples/looped_transformer/reference/make_data.py`.

## Build

    BEND_MEM=6G TMPDIR=/tmp/opencode scripts/bend-guard.sh make session \
      TASK=looped-build \
      CMD="env PATH=$HOME/.local/bin:$HOME/.bend/bin:$HOME/.elan/bin:$PATH \
      LD_LIBRARY_PATH=$HOME/.local/lib/llvm22 \
      bend -o .scratch/looped examples/looped_transformer/main.bend"

Run 320, exit 0, 82.6 s (the final tree; an earlier 88 s build was run 313).

## Data

    python3 examples/looped_transformer/reference/make_data.py \
      --out /tmp/opencode/loopdata2k --steps 2000 --batch 8 \
      --eval-count 8 --max-hops 8

## Result

    make session TASK=looped-plain-2k-fix \
      CMD=".scratch/looped train_eval 2000 1 \
      /tmp/opencode/loopdata2k/train.bin \
      /tmp/opencode/loopdata2k/eval-hop-1.bin"

Run 318, `accuracy 0.25`, 99 s.

    make session TASK=looped-loop4-2k-fix \
      CMD=".scratch/looped train_eval 2000 4 \
      /tmp/opencode/loopdata2k/train.bin \
      /tmp/opencode/loopdata2k/eval-hop-1.bin"

Run 319, `accuracy 0.125`, 234 s.

Both arms use the same 4 stored blocks; plain runs 4 layer applications,
looped runs 10. The eval file has 8 puzzles, so one correct answer is 12.5%.
The result is a smoke result, not an architecture comparison.

The v0 lacks the learned gate and the four-token depthwise K/V convolution.
It is not the burn-lab model. Burn-lab's recorded 2,000-step 16-note result is
plain 35.4%, looped 34.8%, deep 37.3% over 128 puzzles per hop; reproducing that
needs the gate, the short convolution and a larger eval batch.
