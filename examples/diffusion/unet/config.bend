# The 64x64 UNet's shape, fixed before the compile: every number here is a
# type-level constant, so a bigger model is a different program.
#
#   bend examples/diffusion/unet/train.bend          the record's run
#   bend examples/diffusion/unet/train.bend --steps 50 --no-sample   a smoke run
#
# The compile is what costs memory, not the step: choose B and CH before the
# build. B is images per step, CH the width of both levels.

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
  "flowers64-trainfull-gray.u8"

# images in the file (512 at 64x64, one grayscale byte a pixel)
def NIMG() -> Nat:
  7066n

def STEPS() -> Nat:
  6000n

def EULER() -> Nat:
  32n

# draws printed by the sample (eight rounds of B images)
def DRAWS() -> Nat:
  64n
