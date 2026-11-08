# The 16x16 toy scale for the lucidrains reference port.
#
# The reference's own Oxford recipe is 64x64, dim 64, 70k steps; that is an
# 18-hour run on the 8 GB card and is not the target. This example keeps the
# reference's layers and shrinks the image and width until one training run
# is under an hour:
#
#   images 16x16, batch 2, dim 8, two levels (64 -> 16 -> 8 -> 16 in the U).

def B() -> Nat:
  2n

def HW() -> Nat:
  16n

def HW2() -> Nat:
  8n

def CH() -> Nat:
  8n

def CH4() -> Nat:
  32n
