# Week 8 — Replication Laboratory #3

Completed analysis based on the course `replication-lab/replication-lab.qmd` template.
The original template remains unchanged. Starter commit:
`2326687d77e54ddc30ae34367e48cec1a6069027`.

## Reports

- `replication-lab.html`: self-contained rendered Quarto report, as requested in the assignment body.
- `replication-lab.docx`: Word version, provided because the Canvas upload header and general template specify DOCX.
- `replication-lab.qmd`: executable editable source; every render reruns the complete analysis.

Author: **Urmi Rani Biswas** (included in both rendered reports and the Quarto source).
Read and verify the interpretation before submission. No upload, push, or Canvas submission has been performed.

## Research design

Partial replication using original CivilComments-WILDS data, a custom annotation-defined environment split,
and a fixed TF-IDF/logistic regression model. This is not the paper's official benchmark protocol.
Development uses official train rows with Muslim annotation <0.5; ID and shifted evaluation use
disjoint official test rows with annotation <0.5 and >=0.5 respectively. This refers to identities mentioned
in comments, not the identities of their authors. Original validation rows and shifted training rows are unused.
The protocol is recorded in `protocol.json`; there is no model or threshold search.
The original starter commit is recorded in that file, so reproduction also works from an extracted
submission ZIP without Git history or a Git installation.

## Reproduce

Use Python 3.14 (run tested on 3.14.4) and Quarto (run tested on 1.10.19).
From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python download_data.py
.venv/bin/python analysis.py
QUARTO_PYTHON="$PWD/.venv/bin/python" quarto render replication-lab.qmd --to html
QUARTO_PYTHON="$PWD/.venv/bin/python" quarto render replication-lab.qmd --to docx
```

The original working virtual environment is outside this cloned repository at `../../.venv`.
For that environment, replace `.venv/bin/python` above with `../../.venv/bin/python`.
Quarto needs local Jupyter socket access. Install Quarto separately from its official release if unavailable.
Portable Quarto for the original render was extracted to `/tmp/week8-quarto`; that temporary tool path is
not a requirement for subsequent users.

`requirements-lock.txt` captures the full Python environment. If an operating system cannot install these
exact versions, use a compatible Python environment and document any changed versions rather than assume
identical numerical output.

## Data and provenance

- Source study: https://proceedings.mlr.press/v139/koh21a.html
- WILDS: https://github.com/p-lambda/wilds
- Official dataset schema: https://github.com/p-lambda/wilds/blob/main/wilds/datasets/civilcomments_dataset.py
- Pinned mirror: https://huggingface.co/datasets/shlomihod/civil-comments-wilds/blob/3fbfeca/all_data_with_identities.csv
- SHA-256: `403e638c83a225d738a937ff98b61fd0631e30f710d57928c7766d413526b77f`
- License: CC0, as documented by WILDS.

The original CodaLab endpoint failed TLS certificate validation. The downloaded mirror was verified
against its published checksum; no TLS verification was disabled. The mirror was not independently
compared against a newly downloaded original because the original endpoint was unavailable.

The raw CSV is approximately 273 MB and is intentionally excluded from Git and the deliverable ZIP.
The download script obtains it without credentials. Do not commit the raw comments.
Two rows with missing/empty required text were removed. Normalized exact-text deduplication removed
3,448 additional rows, prioritizing train, then validation, then test. The script asserts no normalized
text overlap between development and either evaluation set or between evaluation sets.

## Evidence files

- `analysis.py`: complete analysis; only one model fit per execution.
- `results/performance.csv`: both environments, class counts, and probability/error metrics.
- `results/subgroups.csv`: overlapping positive-identity subgroup metrics and denominators.
- `results/identity_label_accuracy.csv`: identity-by-label cells, including small cells.
- `results/calibration.csv`: ten-bin reliability evidence.
- `results/environments.csv`: composition, prevalence, comment lengths, and identity rates.
- `results/feature_shift.csv`: 15 largest standardized TF-IDF coordinate differences.
- `results/split_manifest.csv`: original row IDs and assigned roles, without comment text.
- `results/provenance.json`: checksum, package versions, seed, integrity checks, and conditional bootstrap interval.

Bootstrap intervals assume independent comments and condition on one fitted model; no training-seed
or article-cluster uncertainty is included. Results do not support causal or author-level demographic claims.
AI assistance and student review responsibilities are disclosed in the report.
