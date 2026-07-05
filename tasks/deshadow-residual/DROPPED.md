# DROPPED surface: deshadow-residual (NOT shipped)

This editable surface (direct clean-image regression vs residual-correction learning) was
DESIGNED and GPU-VALIDATED on the proven image-deshadow harness but DROPPED because it is NOT
CROSS-SEED ROBUST on this synthetic cast-shadow data.

Reason: with the MASK-GUIDED backbone + composite L1+SSIM loss and only a few hundred iters,
DIRECT clean-image regression is already competitive (the mask tells the net exactly where to
brighten), so the residual formulation does NOT reliably win. The ordering flips between seeds:
  seed 42:  direct(weak) light 33.01 / medium 31.31 / heavy 27.88  BEATS
            residual(strong) light 32.38 / medium 30.70 / heavy 26.09   -> weak>strong (inverted)
  seed 1 :  residual(strong) light 34.10 / medium 31.89 / heavy 29.20  BEATS
            direct(weak) light 33.02 / medium 29.19 / heavy 28.45       -> strong>weak (expected)
The ordering is seed-dependent -> dropped.

(Residual learning IS the FIXED strong formulation used by all the shipped configurable
surfaces; it is simply not a robustly-discriminating LEVER on its own here.) The surface code
remains in vendor/image-deshadow/harness.py + solution/residual.py for provenance.
