# Healthcare Package

Brane package for patient health analysis: cardiovascular risk, diabetes risk,
BMI, triage, and batch processing of CSV datasets.

---

## How Brane invokes this package

- **Action dispatch** — the action name is passed as `argv[1]` via `command.args` in `container.yml`. `healthcare.py` reads `sys.argv[1]` to dispatch to the correct handler.
- **Input arguments** — branelet JSON-serialises each input and sets it as an uppercase env var (e.g. input `dataset` → env var `DATASET` containing `"{\\"age\\":55,...}"`).
- **String outputs** — printed to stdout as `output: "<json-string>"`. The surrounding quotes are mandatory to prevent serde_yaml from mistyping numbers or timestamps.
- **Class outputs** — printed as a YAML 2-element list `output: ["ClassName", {fields}]`, which serde deserialises as `FullValue::Instance`.
- **Data / IntermediateResult outputs** — written to `/result/` on disk; nothing is printed to stdout. Brane captures the directory and wraps it as `IntermediateResult`.

---

## Custom classes

Classes are declared in `container.yml` under `types:` and exposed to BraneScript.

### `PatientSummary`

| Field | Type | Description |
|---|---|---|
| `patient_id` | string | Unique patient identifier |
| `age` | integer | Patient age in years |
| `gender` | string | M / F / O |
| `risk_level` | string | `low` / `moderate` / `high` |
| `risk_score` | real | 0–100 composite CVD risk score |

**BraneScript usage:**
```
import healthcare;
let p := get_patient_summary("{\"patient_id\":\"PAT001\",\"age\":65,...}");
println(p.risk_level);
println(p.risk_score);
```

### `RiskAssessment`

| Field | Type | Description |
|---|---|---|
| `patient_id` | string | Unique patient identifier |
| `risk_score` | real | 0–100 composite risk score |
| `risk_level` | string | `low` / `moderate` / `high` |
| `top_factor` | string | Highest-weight risk factor |

---

## Functions

### String-based (single patient, inline JSON)

All functions in this group accept a `dataset` argument — a JSON string
encoding one patient record — and return a JSON string.

#### `validate_patient_data(dataset)`

Validate required fields and value ranges.

**Returns** JSON string:
```json
{ "is_valid": true, "error": "" }
```

Required fields: `patient_id`, `age` (0–150), `gender` (M/F/O), `vital_signs`, `lab_results`.

---

#### `analyze_heart_disease(dataset)`

Cardiovascular risk scoring based on age, gender, vitals, labs, and history.

**Risk factors and weights:**

| Factor | Points |
|---|---|
| Age ≥ 55 | +20 |
| Male gender | +15 |
| Blood pressure > 140 mmHg | +25 |
| Abnormal heart rate (< 60 or > 100 bpm) | +15 |
| Total cholesterol > 200 mg/dL | +20 |
| Elevated fasting glucose > 100 mg/dL | +10 |
| Diabetes history | +15 |
| Hypertension history | +20 |
| Smoking history | +25 |

Score is capped at 100. Risk levels: low (< 20), moderate (20–79), high (≥ 80).

**Returns** JSON string:
```json
{
  "patient_id": "PAT001",
  "risk_score": 75.0,
  "risk_level": "high",
  "risk_factors": ["Advanced age (>=55)", "Elevated blood pressure"],
  "recommendations": ["Schedule immediate specialist consultation", "..."]
}
```

---

#### `generate_report(dataset)`

Combines validation + CVD analysis into a single structured report.

**Returns** JSON string:
```json
{
  "status": "success",
  "patient_id": "PAT001",
  "timestamp": "2026-06-15T21:00:00Z",
  "analysis": { "...": "..." },
  "overall_risk_level": "high"
}
```

---

#### `assess_diabetes_risk(dataset)`

Type-2 diabetes risk scoring based on age, glucose, HbA1c, BMI (from vitals), and history.

**Key thresholds:** fasting glucose ≥ 126 mg/dL (+35), HbA1c ≥ 6.5% (+35), obesity BMI ≥ 30 (+20).

**Returns** JSON string:
```json
{
  "patient_id": "PAT001",
  "risk_score": 55.0,
  "risk_level": "high",
  "risk_factors": ["High fasting glucose (>=126 mg/dL)", "Obesity (BMI 31.2)"]
}
```

The patient JSON may include `lab_results.glucose`, `lab_results.hba1c`, and `vital_signs.weight_kg` / `vital_signs.height_cm` for full scoring.

---

#### `triage_patient(dataset)`

Urgency triage from vital signs (blood pressure, heart rate, temperature, SpO2).

| Level | Criteria |
|---|---|
| `immediate` | Score ≥ 6 (critical BP/HR/temp/SpO2) |
| `urgent` | Score 3–5 |
| `standard` | Score < 3 |

**Returns** JSON string:
```json
{
  "patient_id": "PAT001",
  "triage_level": "urgent",
  "flags": ["Elevated blood pressure (160 mmHg)"],
  "next_steps": ["Seek emergency care within 1 hour", "Monitor vitals continuously"]
}
```

Vital signs keys used: `blood_pressure` (mmHg), `heart_rate` (bpm), `temperature` (°C), `spo2` (%).

---

#### `compute_bmi(weight_kg, height_cm)`

Inputs are `real` (not a JSON patient record).

**Returns** JSON string:
```json
{ "bmi": 27.8, "category": "overweight", "weight_kg": 85.0, "height_cm": 175.0 }
```

BMI categories: `underweight` (< 18.5), `normal` (18.5–24.9), `overweight` (25–29.9),
`obese_class_1` (30–34.9), `obese_class_2` (35–39.9), `obese_class_3` (≥ 40).

**BraneScript usage:**
```
import healthcare;
let result := compute_bmi(85.0, 175.0);
println(result);
```

---

### Class-returning function

#### `get_patient_summary(dataset)`

Returns a `PatientSummary` class instance directly accessible in BraneScript.

**BraneScript usage:**
```
import healthcare;
let p := get_patient_summary("{\"patient_id\":\"PAT001\",\"age\":65,\"gender\":\"M\",...}");
if (p.risk_level == "high") {
    println("High risk patient: " + p.patient_id);
}
```

---

### Data-based functions (batch CSV processing)

These functions accept a `Data` reference pointing to a registered Brane dataset.
Brane mounts the dataset file at `/data/<name>` inside the container and passes
the path via the uppercase env var (e.g. `PATIENTS_FILE`).
Output is written to `/result/` and returned as `IntermediateResult`.

#### `analyze_patients_file(patients_file)`

Reads a CSV dataset, runs CVD analysis on all records.

**Output files in `/result/`:**
- `analysis_report.json` — full batch analysis with per-patient results

**BraneScript usage:**
```
import healthcare;
let patients := new Data{ name := "hospital_patients" };
let analysis := analyze_patients_file(patients);
```

---

#### `filter_high_risk_patients(patients_file)`

Filters the CSV dataset to retain only patients with CVD `risk_level == "high"`.

**Output files in `/result/`:**
- `high_risk.csv` — filtered CSV with an extra `risk_score` column
- `filter_summary.json` — count and timestamp

---

### IntermediateResult chaining

#### `generate_reports_from_results(analysis)`

Reads the `IntermediateResult` produced by `analyze_patients_file` and generates
per-patient JSON reports plus an HTML summary.

**Output files in `/result/`:**
- `reports/<patient_id>_report.json` — per-patient report
- `summary.html` — HTML table of all patients with risk level and score

**BraneScript usage:**
```
import healthcare;
let patients  := new Data{ name := "hospital_patients" };
let analysis  := analyze_patients_file(patients);
let reports   := generate_reports_from_results(analysis);
commit_result("hospital_reports", reports);
```

---

## Patient JSON format

```json
{
  "patient_id": "PAT001",
  "age": 65,
  "gender": "M",
  "vital_signs": {
    "blood_pressure": 155,
    "heart_rate": 88,
    "temperature": 37.2,
    "spo2": 97,
    "weight_kg": 90,
    "height_cm": 175
  },
  "lab_results": {
    "total_cholesterol": 230,
    "glucose": 112,
    "hba1c": 6.1
  },
  "medical_history": ["hypertension", "smoking"]
}
```

**Required fields:** `patient_id`, `age`, `gender`, `vital_signs`, `lab_results`.  
`medical_history` values: `diabetes`, `hypertension`, `smoking` (others are ignored).

## CSV dataset format

```
patient_id,age,gender,blood_pressure,heart_rate,total_cholesterol,glucose,medical_history
PAT001,65,M,155,88,230,112,hypertension|smoking
PAT002,42,F,118,72,185,95,
```

`medical_history` is pipe-separated (`|`). Optional columns: `glucose`, `temperature`, `spo2`, `weight_kg`, `height_cm`, `hba1c`.

