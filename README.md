# Tavis–Cummings lattice simulation and Qualtran resources

A compact, self-contained companion to **Simulating closed- and open-system dynamics of Tavis–Cummings lattices on fault-tolerant quantum computers**. The notebooks distinguish classical physics validation, executable circuit validation, arithmetic subtotals, and full logical resource estimates.

## Layout

```text
validation/
  closed_system.ipynb              Classical closed dynamics and signal checks
  open_system.ipynb                Two-cavity Lindblad/dilation validation
  validation.py                   Shared validators, references and gate tests
  results/                        Reproducible data, figures and test record
resource_estimation/
  manuscript_tables.ipynb          Tables II, III/IV, VI, Appendix H and Figure 7
  closed_system_resources.ipynb    Closed-system construction and resources
  open_system_resources.ipynb      Appendix H example and resource heatmaps
  block_encoding.py               Shared arithmetic, PREP, SELECT and encodings
  walk.py                         Shared controlled walks and rotation modules
  gqsp.py                         Precision, GQSP schedules, compilation and tables
  results/                        Counted results, tables, figures and manifests
requirements.txt
source_consolidation.json          Origin of consolidated source definitions
```

There are **three resource Python files and one validation Python file**. Related implementations are merged as ordinary readable definitions, not embedded modules or dynamically loaded source strings. Internal prefixes distinguish helpers merged from the development files; the notebook-facing API is at the end of `gqsp.py`. No import reads the parent development bundle. Historical notebooks, duplicate protocol scripts and standalone cost-ledger scripts are omitted.

## Environment

The recorded environment uses Python **3.12** and Qualtran **0.7.0**. `requirements.txt` records the tested dependency versions, including notebook execution and the consolidated gate tests.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name tcl-qualtran --display-name "TCL / Qualtran 0.7"
```

Open the notebooks in Jupyter or VS Code and select this environment. They locate this folder from their own working directory; no machine-specific paths need editing. For small numerical problems, setting `OPENBLAS_NUM_THREADS=1` before starting Python can avoid excessive BLAS thread overhead.

## What to run

From this folder:

```sh
# Independent small-system physics validation
python validation/validation.py closed
python validation/validation.py open

# Actual gate execution, projected blocks, controls, adjoints and catalysts
python -m pytest -q validation/validation.py

# Compile all manuscript resource cases and arithmetic primitive audits
python resource_estimation/gqsp.py tables

# Compile Figure 7's 60-row controlled-walk resource table and plot
python resource_estimation/gqsp.py figure7

# Generate the fixed-cutoff 3x3 closed-system heatmap grid
python resource_estimation/gqsp.py closed-grid

# Regenerate resource plots from the compiled tables
python resource_estimation/gqsp.py plots
```

`python resource_estimation/gqsp.py compile` compiles the coherent closed/open cases without the table audit. Compilers use source-hashed caches and isolated processes to limit memory growth. Editing any resource source invalidates resource caches; calculations rerun on the next notebook/command execution. Cached output is never used to provide a circuit's measured gate count.

The notebooks ship with executed outputs and supporting results. Gate-validation records also contain the validation/resource source hashes. After editing validation code, run `python validation/validation.py gates` to rerun the tests and refresh `validation/results/gate_validation.json` automatically. The validation notebooks reject a stale record until refreshed.

## Validation

**Closed system.** `closed_system.ipynb` begins with a recalculation of manuscript Figure 8: momentum-resolved vacuum dispersion and photon weights, the Hann-window single-polariton-reference spectrum at times 20/60/200 with both addition and removal, and the hard-window random-phase error-propagation experiment (seed 917). CSV data and PNG/SVG figures are supplied, with analytic-transform versus direct-quadrature checks. Panel (c) illustrates random-error cancellation rather than a measurement-shot concentration guarantee. The notebook then builds independent number-sector matrices for a periodic 4x4 lattice with one emitter per cavity and cutoff 2. It checks the analytic vacuum polariton dispersion, the addition/removal commutator sum rule, exact matrix-exponential versus eigenvector evolution, excitation conservation, and Chebyshev convergence. Its separate signal-processing check generates the first Table VI case's numerical GQSP phases and checks ideal/rounded spectral responses. These are small classical references and signal-layer checks, not large-lattice statevector simulations.

**Open system.** `open_system.ipynb` retains the two-cavity benchmark with two emitters per cavity, cutoff 2 and at most two excitations. Independent tensor-product and reduced-basis constructions are compared. Tests cover state trace, Hermiticity and positivity, Kraus completeness, local dissipative channels, analytic singlet/imbalance slopes, and first-order sampled-dilation versus second-order symmetric-splitting convergence. Averaging over jumps is deterministic, with no Monte Carlo sampling noise. The dynamics curves exclude circuit-synthesis errors.

**Circuit tests.** The tests in `validation.py` execute decomposed gates, check clean workspace, compare direct/paired/sparse projected encodings with ladder references, check the Appendix H jump and Hermitian-dilation blocks, inactive controls, exact adjoints, two persistent catalysts, and fresh timestep environments. They do not certify arbitrary-input full-channel accuracy for the large resource cases.

## Manuscript tables and conventions

| Output | What is reproduced |
|---|---|
| Table II | Complete coherent encodings and shared controlled walks, with matched precision and actual peak allocations. |
| Tables III/IV | Fixed-precision arithmetic subtotals and an independent primitive audit, including complete XOR row oracles. |
| Table VI | Generic GQSP evolution schedules, supplied catalysts and seven-T coherent Toffolis, without an external Hadamard-test control. |
| Appendix H | The specified open-system jump branches, shared dilation walk, precision plan, timestep T gates and qubit allocation. |
| Table VII-style extension | A new open-family summary over 4x4, 4x8 and 8x8 lattices with 1, 2 and 4 emitters per cavity. This is not a printed manuscript Table VII. |

The Table III primitive convention permits measurement-assisted AND erasure and is **not** the convention of the complete coherent queries. Its analytical and measured subtotals are labelled separately. Destination-address slack explains the primitive-subtotal differences; unused gates are not added to force equality. The arithmetic formulas remain only where required to reproduce these tables, within the shared `gqsp.py` implementation.

All Table II and Table VI integer resource entries match the supplied manuscript. Two printed Table VI Bessel-tail decimals round downward; the exported `bound_rounded_up` values give safe upward displays `2.3637e-8` and `2.9339e-8`. This does not alter the degree or resource counts.

Closed resource notebooks reproduce the current uniform charge-removed manuscript cases and a separate 3x3 heatmap grid: lattices 4x4/4x8/8x8 by N=1/2/4 at fixed cutoff 3, time 1 and target error 0.001. The heatmap panels show direct evolution T gates, paired evolution T gates and direct logical qubits; the first Table VI case is highlighted. Existing Table II/Figure 7 counts are reused, and only missing shared walks are compiled. T totals aggregate counted walk and fixed-angle rotation modules in the specified GQSP schedule; the supplied rotation catalyst is included and its adder fits in the clean bank. These heatmaps have the open notebook's layout, but retain closed-system time/error targets. The original nonuniform/historical development notebooks are not included in this curated folder. General coefficient-dependent block-encoding construction remains available through `block_encoding.model_build_pair`; its precision must be planned for those coefficients rather than reusing the uniform tables.

### Figure 7 reproduction

The table notebook reuses the previously validated Qualtran sweep, whose original source hashes are recorded and whose 60 count/qubit pairs were independently regenerated and matched. Its saved table contains 60 measured resource rows: five geometries (4x4, 4x8, 8x8, 8x16, 16x16), four photon cutoffs (3, 7, 15, 31), N=2 and three implementations. Every point is counted from the complete coherent shared walk, with precision derived per instance. The figure renderer reads `resource_estimation/results/figure7_resource_table.csv`; no resource coordinates are hard coded. Per-instance source hashes and the CSV checksum are recorded in `results/figure7/manifest.json`. The three Table II cases alone are not substituted for this sweep.

## Appendix H benchmark

For the uniform 4x4, N=1, cutoff-3 example, with coupling 1, hopping 0.05, photon loss and relaxation rates 0.1, zero dephasing and T=10:

| Quantity | Reproduced value |
|---|---:|
| b, B, K, m | 33, 24, 6, 48 |
| Loss / relaxation / dephasing T gates per timestep | 429,786 / 406,602 / 406,098 |
| Mean T gates per timestep | 423,990 |
| Logical qubits, including both catalysts | 434 |
| Timesteps per complete run | 251,658,240 |
| Independent measurement runs | 1,794,879 |
| Mean T gates per complete run | 1.067005771776e14 |
| Mean T gates across measurement runs | 1.915146252640e20 |

The measurement count uses the tighter correlated-moment bound of Eqs. (126)–(129), with target g(2) error 0.1, assumed occupation lower bound 0.5, failure probability 0.01 and one reported time. There is no additional trajectory multiplier. Dephasing is compiled as a circuit check but has zero sampling weight in this example.

**Scope:** Eq. (103), evaluated at channel tolerance e/3, assigns e/12 to sampling alone. The stated timestep prescription does not yet budget the dilation remainder. Consequently, neither the manuscript workload nor its heatmap extensions are presented as certified complete observable costs. The occupation lower bound is assumed, not established by dynamics. Input-state/catalyst preparation and physical error correction are excluded. The generic large-instance GQSP schedule counts executable fixed-angle modules but does not generate a numerical phase list.

## Reproduction data

CSV tables, numerical summaries, PNG plots and compact per-instance compilation records are included. Resource manifests hash all three implementation files. The source-consolidation record identifies which development definitions were retained and where they were merged; those original development files are not needed to run this folder.

The manuscript PDF is not bundled. Table numbering and transcribed reference values refer to the supplied revision containing Appendix H. Compare with that revision when assessing agreement, rather than older development PDFs.
