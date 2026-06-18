# Healthcare Dataset

Sample patient records dataset for testing the `healthcare` package on a Brane instance.
Follows the Brane dataset conventions from the [official documentation](https://braneframework.github.io/manual/software-engineers/data.html).

## Contents

```
healthcare/
├── patients.csv           ← 5 patient records (the actual data file)
├── data.yml               ← Brane dataset descriptor — build this with brane
└── workflows/
    ├── analyze_single.bs      ← Single patient, data defined inline (no dataset needed)
    ├── batch_analysis.bs      ← Load dataset via Data reference, commit IntermediateResult
    └── parallel_analysis.bs   ← Same dataset analysed on two sites in parallel
```

## Patient Records

| ID     | Age | Gender | BP  | HR | Cholesterol | History               | Expected Risk |
|--------|-----|--------|-----|----|-------------|-----------------------|---------------|
| PAT001 | 67  | M      | 165 | 88 | 245         | hypertension, smoking | CRITICAL      |
| PAT002 | 45  | F      | 135 | 72 | 210         | —                     | MODERATE      |
| PAT003 | 32  | M      | 118 | 65 | 175         | —                     | LOW           |
| PAT004 | 71  | F      | 155 | 90 | 260         | diabetes, hypertension| HIGH          |
| PAT005 | 52  | M      | 148 | 78 | 230         | smoking               | HIGH          |

## How it works (Brane Data model)

Brane distinguishes between **variables** (in-memory, manipulable in BraneScript) and
**data** (files on disk, opaque to BraneScript, transferred by the framework).

- **`Data`** — a reference to a registered dataset. Created with `new Data{ name := "..." }`.
  Brane mounts the file inside the container and passes the file path via an env var.
- **`IntermediateResult`** — the output of a package function that produces file data.
  The package writes to `/result/`; Brane wraps that directory automatically.
  Scoped to the current workflow unless persisted with `commit_result`.
- **`commit_result("name", result)`** — promotes an `IntermediateResult` to a
  persistent named dataset available to future workflows.

The `analyze_patients_file` action in `container.yml` implements this pattern:

```yaml
'analyze_patients_file':
  input:
    - name: patients_file
      type: Data          # Brane passes the file path via PATIENTS_FILE env var
  output:
    - name: output
      type: IntermediateResult  # package writes to /result/analysis_report.json
```

## Setup

### 1. Register the dataset with Brane

```bash
brane data build submodules/datasets/healthcare/data.yml

# Verify it appears:
brane data list
```

To make a permanent copy instead of a symlink:

```bash
brane data build --no-links submodules/datasets/healthcare/data.yml
```

### 2. Run the workflows

```bash
# Inline single patient — no dataset file required
brane workflow run submodules/datasets/healthcare/workflows/analyze_single.bs

# Batch: load dataset → analyze all records → commit result
brane workflow run submodules/datasets/healthcare/workflows/batch_analysis.bs

# Parallel: same dataset on two sites (edit site names first)
brane workflow run submodules/datasets/healthcare/workflows/parallel_analysis.bs
```

## Data format

`patients.csv` is a flat CSV file with one row per patient. The `medical_history`
column is pipe-separated (`|`) to avoid conflicts with the CSV comma delimiter.

```csv
patient_id,age,gender,blood_pressure,heart_rate,total_cholesterol,medical_history
PAT001,67,M,165,88,245,hypertension|smoking
PAT002,45,F,135,72,210,
PAT003,32,M,118,65,175,
PAT004,71,F,155,90,260,diabetes|hypertension
PAT005,52,M,148,78,230,smoking
```

Valid values:
- `gender`: `M`, `F`, or `O`
- `medical_history` pipe-separated entries: `hypertension`, `diabetes`, `smoking` (or empty)
