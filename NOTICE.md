# Authorship, input provenance and references

The implementation, experiment design, analysis and project documentation are maintained by Lifeng Wang. Standard algorithms and published methods are attributed in the technical report and bibliography.

## Reference waveform

The bundled `reference_signal.npy` and `reference_signal.wav` were supplied externally by Xi'an Jiaotong-Liverpool University and are used as fixed numerical test inputs. The project does not claim authorship of these input files. Their original bytes are retained for reproducibility.

| File | SHA-256 |
|---|---|
| `reference_signal.npy` | `2f62c84fb0e11586373b2dc0d92f180206d15d4a77098046c9c43de290657095` |
| `reference_signal.wav` | `0ffec41d7773f8a5fda064aecfec3affe956b6955227ab0c5774e4a4349c9454` |

The NPY defines the numerical amplitude units. The WAV is signed 16-bit PCM with the exact relation `round(20000 * NPY)`.

## Published methods and dependencies

Published references retain their original titles and authors. NumPy, SciPy, Matplotlib and the report compiler remain the work of their respective developers and are governed by their own licenses. Articles and dependency installations are not redistributed here; links and citations identify technical sources.

No blanket license for the externally supplied reference waveform is asserted. Public repository availability does not transfer third-party rights.
