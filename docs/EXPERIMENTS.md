# Experiment map

Run dependency-sensitive regeneration through `run_all.py`. The stable internal labels below map each investigation to implementation and saved evidence.

| Study | Purpose | Main implementation |
|---|---|---|
| R1.1 | Reference spectrum, windows and power | `stage1/r11/characterise_reference.py` |
| R1.2 | Sample-rate selection | `stage1/r12/sampling_rate.py` |
| R1.3 | Predicted and measured aliasing | `stage1/r13/aliasing_experiment.py` |
| R1.4 | Joined sampling evidence | `stage1/r14/sampling_report.py` |
| R2.1 | Filtering approach | `stage2/r21/filtering_approach.py` |
| R2.2 | FIR parameter search | `stage2/r22/filter_parameters.py` |
| R2.3 | Held-out filter validation | `stage2/r23/validation.py` |
| R3.1 | Quantisation resolution | `stage3/r31/quantisation.py` |
| R3.2 | Code selection and capability | `stage3/r32/pilot_comparison.py` |
| R3.3 | Error-probability range | `stage3/r33/error_range.py` |
| R3.4 | Held-out coding gain | `stage3/r34/coding_gain.py` |
| R4.1 | Metric interpretation | `stage4/r41/evidence.py` |
| R4.2 | Pilot and record-count planning | `stage4/r42/pilot.py` |
| R4.3 | Four-configuration evaluation | `stage4/r43/evaluation.py` |
| R5.1 | One-decision counterfactuals | `stage5/counterfactual.py` |
| R5.2 | Rate-constrained decision | `stage5/constraint.py` |
| Packing | Significance and codeword grouping | `stage5/packing_diagnostic.py` |

Measured CSV, arrays, freezes and seed manifests live in the corresponding `results/` directory. The final packing diagnostic uses `stage5/packing_diagnostic/`. Plots are generated from those measured files, with report copies under `report/images/`.

## Reading the main studies

The reference has coherent components at 440, 1320, 2200 and 6000 Hz, with amplitudes 1, 0.6, 0.3 and 0.1. Mean square is 0.73. The 8 kHz aliasing experiment has full-record MSE approximately 0.01 and demonstrates an information loss that sample fitting cannot reverse.

The FIR study evaluates 372 candidates and validates the selected `M385-B70-hann` using 400 held-out records. The quantiser scan contains 1,040 bit-depth measurements; 14 bits is the smallest tested width meeting the declared component budget across both branches and all design records.

The coding studies distinguish design cases from new draws and compare exact model predictions with measurements. Held-out coding validation uses 300 independent sampled inputs. Bit errors, waveform quality and perfect payload recovery remain distinct outcomes.

The main four-configuration study uses 1,400 bundles, 22 unique conditions and 123,200 paths. `stage4/r43/results/trials.csv` retains record-level measurements; `decomposition.csv` retains components and signed cross terms. The central plot contains two severity slices, not a full response surface.

The separate counterfactual study uses 20 pilot and 350 confirmation bundles. Nine variants at 15 conditions give 47,250 confirmation paths. Only twelve main conditions enter the equal-weight objective; three diagnostics have zero objective weight.

The constrained scenario reuses a prespecified anchor subset. Hamming(15,11) fits the 320 kbit/s cap at 305.46 kbit/s and meets the mean-quality rule, with reduced perfect-payload reliability. It is a conditional design tradeoff, not a replacement for Hamming(7,4) throughout the original main domain.

## Interpretation rules

- Design, held-out validation, deterministic controls and replay checks serve different purposes.
- Replaying a random leaf creates no new independent observation.
- Mean-NMSE intervals do not describe individual-record spread.
- Exact-payload proportion intervals have a different family and interpretation from mean-waveform intervals.
- Seven of the 92 main-study means miss the planned relative precision target; some high-SNR contrasts remain unresolved.
- Logical bit costs do not establish physical-link throughput, energy or latency.

Full numerical tables and derivations are in `report/main.pdf`.
