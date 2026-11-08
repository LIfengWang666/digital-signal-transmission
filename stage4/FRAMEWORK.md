# Complete-system experiment plan

The system comparison fixes the reference, sample rate, quantiser, FIR and baseline code before measuring held-out performance. Filtering and coding form four configurations; each complete-record bundle shares the intended noisy input across paired methods.

Full-record NMSE is the primary metric and MAE is the main alternative. No fitted gain, fitted alignment or endpoint trimming is used. The two severity slices contain 22 unique conditions, with twelve equally weighted main conditions and zero-weight diagnostics/refinements.

The pilot plans 1,400 independent validation bundles for a 92-statistic approximate t/CLT family. Metric interpretation, planning and held-out validation remain separate operations. Unmet precision targets and unresolved effects are retained in the results.

The executable choices are recorded in `r41/design.json`, `r42/design.json` and `r43/design.json`. Source-bound freezes identify the implementation used by the current publication edition. See [the architecture](../docs/ARCHITECTURE.md) and [experiment map](../docs/EXPERIMENTS.md) for interpretation.
