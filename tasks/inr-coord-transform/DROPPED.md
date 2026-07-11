# Dropped

The current candidates do not isolate an independent design question.

The `unit` candidate maps `x` to `0.5 * x + 0.5`; it is not merely a
translation and therefore is not phase-equivalent to `identity` as previously
claimed. For a paired sine/cosine random Fourier encoding, the additive term
rotates each feature pair by a fixed phase that a following linear layer can
absorb, while the factor of `0.5` halves the effective Fourier frequency. Thus
`unit` is equivalent, up to the absorbable phase rotation, to changing the
frequency scale already exposed by `inr-fourier-frequency`.

Multiplying coordinates before the same Fourier matrix likewise changes the
effective frequency scale. The `inflate` candidate therefore creates a large
gap by duplicating another sibling rather than testing a separate axis.

Reactivation requires a transform whose effect is not reducible to the Fourier frequency surface, followed by new same-protocol measurements on all settings.
