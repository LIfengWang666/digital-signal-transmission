# Design tradeoffs and constrained transmission

This stage tests one-decision alternatives using a separate pilot and confirmation study. The executable protocol is `design.json`; `counterfactual.py` runs the variants and `assessment.py` evaluates the declared objectives and uncertainty.

Twenty pilot bundles plan 350 confirmation bundles. Nine variants are evaluated at 15 conditions: twelve main conditions and three zero-weight diagnostics. The confirmation run therefore contains 47,250 paths. Earlier validation data support diagnosis but are not imported as new confirmation observations.

`constraint.py` reuses the prespecified anchor subset to assess a logical 320 kbit/s cap. Exact cost accounting includes final-block padding, while perfect recovery tests only real information bits. The feasible Hamming(15,11) choice trades exact-payload reliability for the retained mean-waveform-quality rule and lower traffic.

`packing_diagnostic.py` separately enumerates Hamming(7,4) error patterns and illustrates how bit significance and grouping affect waveform error. It does not add observations to the confirmation means.

Read [the experiment guide](../docs/EXPERIMENTS.md) and use `run_all.py` from the repository root for dependency-ordered reproduction.
