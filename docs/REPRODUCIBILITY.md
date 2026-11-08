# Reproduction guide

## Recorded environment

Exact comparisons use 64-bit Windows, Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3 and Matplotlib 3.10.0. The explicit lock includes native BLAS and rendering dependencies.

```bash
conda create -n signal-transmission --file conda-win-64-explicit.txt
conda activate signal-transmission
python -c "from reproduction.check_reproducibility import require_environment; require_environment(); print('Environment OK')"
```

The lock targets Windows. Other operating systems may support exploratory execution, but exact frozen-result agreement is not validated there. The entry point checks the recorded stack.

## Available commands

Check saved scientific products without simulating new records:

```bash
python -m reproduction.check_reproducibility
```

Regenerate measurements, checks, figures and report fragments:

```bash
python run_all.py --data-only
```

Run serially when desired:

```bash
python run_all.py --data-only --r4-workers 1 --r5-workers 1
```

For the complete workflow and PDF compilation, install Tectonic 0.17.0:

```bash
python run_all.py --tectonic "C:/Tools/tectonic/tectonic.exe"
```

Compile only the saved report inputs:

```bash
python -m reproduction.build_pdf --tectonic "C:/Tools/tectonic/tectonic.exe" --report report
```

The full run includes independent numerical replay and is substantially slower than inspecting saved results. Use a disposable clone to preserve pristine outputs. A requested full run discards prior generated execution freezes and disposable caches, then binds current upstream fingerprints before each downstream study; scientific targets and experiment settings remain fixed. Six worker processes are used by default in the large studies; the FIR parameter scan separately uses its recorded four-thread BLAS setting.

## Scientific comparison policy

`stage6/baseline_products.json` fixes 316 main and seven packing-diagnostic targets. These are comparison references, not simulation inputs, and are not refreshed after a failure.

| Artifact | Compared content |
|---|---|
| CSV | Every field and row, including exact numerical strings; line endings ignored |
| NPY / binary | Exact bytes |
| Generated LaTeX | Exact text with normalised line endings |
| PNG | Exact decoded RGBA pixels |
| SVG | Geometry, visible text and styling; nonvisual metadata and opaque identifiers excluded |
| Packing JSON | Values excluding source-fingerprint fields |

The public-edition targets were fixed before packaging. Cosmetic metadata exclusions do not introduce a numerical tolerance for measured CSV or arrays. This release uses the existing validated measurements; publication does not claim a new complete numerical replay.

## Report builds

The builder copies source inputs into a new temporary TeX directory. Prior PDF and auxiliary files are not inherited. `SOURCE_DATE_EPOCH` fixes output timestamps. Initial compilation can download TeX dependencies; the dependency cache is separate from experiment caches.

Keep the full `report/` tree: `main.tex` alone lacks its included sections, bibliography, generated measurements and figure assets. The six final vector figures are generated automatically from measured CSV by `reproduction.prepare_publication_figures`.

## Troubleshooting

**Missing module:** activate the environment and run from the repository root.

**Version mismatch:** compare imported versions with the lock. Changing expected version strings does not validate a different environment.

**Frozen protocol differs:** inspect the changed source in a fresh disposable clone. Do not silently replace a freeze in a directory containing prior measurements.

**Tectonic unavailable:** supply its real executable path or use `--data-only`.

**Scientific mismatch:** inspect the failing path, generator and environment. A successful script exit alone is not evidence that the reported content agrees.

**Repository size:** complete trial CSV and seed manifests are included for inspection. `.work/`, `.build-cache/`, bytecode and TeX intermediates are ignored by Git.

## Evidence scope

Records under `stage6/results/` distinguish numerical execution from source-only report compilation. Replaying fixed streams does not increase the independent sample size. Local verification does not establish execution on a second machine or generalisation to a different signal family.
