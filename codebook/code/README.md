# Analysis modules

Shared utilities imported by the analysis and robustness scripts.

| File | Scope |
|---|---|
| `_common.py` | Main analysis (85-channel working sample) |
| `_common_fr47.py` | FR47 final analytic sample used in the manuscript |
| `_common_grp.py` | Participation-frequency subsample analyses |

Requires Python 3.10+, `pandas`, `numpy`, `linearmodels`.

**These modules cannot be run as-is.** They read
`final_analysis_data_clean_85w.csv` and `results/dual_reclassified.csv`,
which are not redistributed here (see Data availability in the root README).
They are published so that the variable construction and model
specifications can be inspected.

## Correction in v1.0.1

`parent_like_count` is populated only on parent-comment rows in the source
data. Earlier code carried only the two parent labels over to child rows,
leaving `log_parent_like` constant at zero, so `PanelOLS(drop_absorbed=True)`
silently dropped it from every model. The parent like count is now merged on
`thread_id` alongside the labels. Sample size is unaffected
(N = 356,201 children); coefficients shift in the third decimal place.
