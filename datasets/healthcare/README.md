# Healthcare Dataset

Sample patient records dataset for testing the `healthcare` package on a Brane instance.
Follows the Brane dataset conventions from the [official documentation](https://braneframework.github.io/manual/software-engineers/data.html).

## Contents

```
healthcare/
├── patients.csv           ← 15 patient records with vitals + lab columns
├── data.yml               ← Brane dataset descriptor — build this with brane
└── workflows/
    ├── analyze_single.bs      ← Single patient, data defined inline (no dataset needed)
    ├── batch_analysis.bs      ← Load dataset via Data reference, commit IntermediateResult
    ├── batch_diabetes.bs      ← Batch diabetes risk analysis
    ├── cohort_stats.bs        ← Cohort-level descriptive statistics
    ├── filter_diabetics.bs    ← Filter dataset by diabetes history
    ├── parallel_analysis.bs   ← Same dataset analysed on two sites in parallel
    └── triage_all.bs          ← Batch triage ordered by urgency
```

## Patient Records

The dataset now contains 15 patients and includes additional clinical fields:

- demographics: `patient_id`, `age`, `gender`
- vitals: `blood_pressure`, `heart_rate`, `temperature`, `spo2`
- labs: `total_cholesterol`, `glucose`, `hba1c`
- anthropometrics: `weight_kg`, `height_cm`
- history: `medical_history`

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

# Batch diabetes scoring
brane workflow run submodules/datasets/healthcare/workflows/batch_diabetes.bs

# Cohort statistics
brane workflow run submodules/datasets/healthcare/workflows/cohort_stats.bs

# Filter diabetics
brane workflow run submodules/datasets/healthcare/workflows/filter_diabetics.bs

# Triage all patients
brane workflow run submodules/datasets/healthcare/workflows/triage_all.bs

# Parallel: same dataset on two sites (edit site names first)
brane workflow run submodules/datasets/healthcare/workflows/parallel_analysis.bs
```

## Data format

`patients.csv` is a flat CSV file with one row per patient. The `medical_history`
column is pipe-separated (`|`) to avoid conflicts with the CSV comma delimiter.

```csv
patient_id,age,gender,blood_pressure,heart_rate,total_cholesterol,glucose,hba1c,temperature,spo2,weight_kg,height_cm,medical_history
PAT001,67,M,165,88,245,112,6.1,37.2,96,92,175,hypertension|smoking
PAT002,45,F,135,72,210,95,5.5,36.8,98,68,165,
PAT003,32,M,118,65,175,88,5.2,37.0,99,75,180,
```

Valid values:
- `gender`: `M`, `F`, or `O`
- `medical_history` pipe-separated entries: `hypertension`, `diabetes`, `smoking` (or empty)
