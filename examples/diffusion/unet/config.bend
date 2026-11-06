# The 64x64 UNet's shape, fixed before the compile: every number here is a
# type-level constant, so a bigger model is a different program.
#
#   bend examples/diffusion/unet/train.bend          the record's run
#   bend examples/diffusion/unet/train.bend --steps 50 --no-sample   a smoke run
#
# The compile is what costs memory, not the step: choose B and CH before the
# build. B is images per step, CH the width of both levels.
#
# B is not free. A batch of B 64x64 images is B*4096 rows in the model's types,
# and the checker cannot equate a type-level Nat past about 16,384 on this
# machine (bendlang/bend#1071, measured: 8,192 checks, 16,384 only on an empty
# stack, 32,768 overflows). B = 2 is therefore the largest batch at 64x64; a
# larger batch has to buy its rows with resolution (16x16 allows 26).

def B() -> Nat:
  2n

def CH() -> Nat:
  64n

def RES() -> Nat:
  64n

def HW() -> Nat:
  Nat.mul(RES(), RES())

def HW2() -> Nat:
  1024n

def HW4() -> Nat:
  256n

# the export: the directory is the runner's MLBEND_DATA when it is set and the
# machine's export cache otherwise
def DATA() -> String:
  "flowers64-train-gray.u8"

# images in the file (3,874 at 64x64, one grayscale byte a pixel)
def NIMG() -> Nat:
  3874n

def STEPS() -> Nat:
  4000n

def EULER() -> Nat:
  32n

# draws printed by the sample (DRAWS/B rounds of B images)
def DRAWS() -> Nat:
  64n
