# Architecture and numerical conventions

The system is a batch simulation with two distinct impairments: Gaussian noise before sample filtering and independent bit flips after encoding. Its outputs are reconstructed waveforms, fidelity measurements, bit costs and uncertainty estimates.

## Data and evaluation boundary

The NPY reference contains 96,000 float64 samples, representing two seconds at 48 kHz. The WAV is a signed 16-bit PCM listening representation with `round(20000 * NPY)` mapping. Numerical work retains the original amplitude units.

`reproduction.verify_inputs` checks hashes, shape, finite values and this exact relationship. The clean waveform and true noise/error masks are available to measurements only; the receiver uses received bits and public parameters.

## Sampling and reconstruction

The selected sequence is `u[3*n]` at 16 kHz. An 8 kHz diagnostic deliberately aliases the 6 kHz component to 2 kHz. Reconstruction evaluates the finite periodic Fourier representation on the original 48 kHz grid, including the even-length Nyquist correction. SciPy resampling provides a numerical cross-check. Nonperiodic signals require reconsidering this boundary assumption.

## Noise and filtering

Gaussian variance is `mean(x**2) / 10**(snr_db/10)`. Draws are neither recentered nor rescaled after generation. Finite realised SNR therefore differs from target SNR.

The symmetric multiband FIR uses the four known component frequencies. The baseline has 385 taps, a symmetric Hann window and 70 Hz half-width per band. FFT convolution pads to at least `N+M-1` and uses an explicit delay-compensating crop. Direct convolution and controlled examples verify its linear-convolution semantics.

Filter choices use design data; held-out records validate the frozen choices. The four-configuration study retains sampling, quantisation and reconstruction while switching FIR and coding independently.

## Quantisation and packing

The endpoint codebook spans `[-A,A]` with `2**B` levels and spacing `2*A/(2**B-1)`. The implemented epsilon-shifted scale/round/clip expression and inverse map are documented in the report. The baseline is `B=14`, `A=18`. Overloads are counted before clipping; a Gaussian headroom bound does not imply an absolute guarantee for an unbounded distribution.

Unsigned labels are packed sample-major and MSB-first. Bit significance and codeword grouping affect waveform damage. A separate exhaustive packing diagnostic demonstrates that small residual BER and small reconstructed error are different claims.

## Coding and rate accounting

Fixed systematic matrices define Hamming(7,4) and Hamming(15,11). Syndrome correction, repetition(5,1) and uncoded transmission are implemented explicitly. Error-pattern enumeration checks capability and miscorrection behaviour.

Padding fills the final message block before encoding. It contributes to transmitted bits, but its correctness is excluded from the perfect-real-payload event. The baseline Hamming(15,11) record uses eight pad bits and 610,920 transmitted bits.

## Random streams

The root seed is 2470964, using `SeedSequence(pool_size=4)` and PCG64.

| Purpose | Explicit path |
|---|---|
| Gaussian anchor parameter | `(0, 0)` |
| BSC anchor parameter | `(0, 1)` |
| Gaussian vector | `(1, study, phase, condition, repeat)` |
| BSC mask | `(2, study, phase, condition, repeat, layout)` |

Explicit paths avoid dependence on loop or worker order. Distinct repeats use distinct leaves. Pairing intentionally shares inputs across methods. `metadata/randomness_design.json` registers the tree; stage manifests record actual calls and fingerprints.

## Metrics and inference

Full-record NMSE divides reconstruction MSE by reference mean square. MAE retains amplitude units. Neither uses fitted gain, fitted delay or trimmed endpoints.

The main study has a declared 92-mean approximate Bonferroni t/CLT family. Each twelve-condition objective is calculated within a bundle before estimating uncertainty across bundles. Counterfactual confirmation uses a distinct 19-statistic family. Pointwise Clopper-Pearson intervals for perfect payloads are separate from mean-waveform intervals.

## Execution layers

`run_all.py` orders numerical stages and independent checks. Plotting modules read measured CSV. Report exporters generate tables/macros and copy generated figures. The six final vector figures are regenerated automatically.

The scientific-content checker compares fixed targets without feeding them into the simulations. The PDF builder copies source inputs to a new temporary workspace and fixes timestamps. Numerical replay and source-only report compilation retain separate evidence.
