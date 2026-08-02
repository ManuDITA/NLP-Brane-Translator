# data_masking package

Privacy-preserving data masking for Brane workflows. Redact, pseudonymise, and generalise sensitive fields in CSV and JSON datasets before passing data to downstream analytics packages. Designed to align with GDPR pseudonymisation (Art. 4(5)) and anonymisation principles.

---

## Why this package exists

Brane's policy deliberation engine controls *which* data flows to *which* compute locations, but it does not transform the data itself. `data_masking` fills that gap: it sits at the **start of a pipeline** and strips or replaces sensitive fields so that downstream packages (e.g. `healthcare`, `text_analysis`) never receive raw PII.

Typical workflow pattern:

```
patients.csv (Data)
  → mask_csv_file(...)     → masked_patients IR
  → analyze_patients_file  → analysis IR
  → commit_result(...)
```

---

## Brane invocation model

- **Action name**: passed as `sys.argv[1]` via `container.yml` `command.args`.
- **Inputs**: each input argument is set as an **uppercase environment variable** by branelet, JSON-serialised (e.g. input `value` → env `VALUE = "\"alice@example.com\""`).
- **String outputs**: printed to stdout as `output: <json-quoted-string>`.
- **Class outputs**: printed as a 2-element YAML list `["ClassName", {fields}]`.
- **File outputs**: written to `/result/`; Brane wraps `/result/` as an `IntermediateResult`.

---

## Masking strategies

| Strategy | Description | GDPR classification |
|---|---|---|
| `redact` | Replace with `[REDACTED]` | Anonymisation |
| `hash` | SHA-256 hex digest (deterministic) | Pseudonymisation |
| `partial` | Keep last 4 chars, mask rest with `*` | Pseudonymisation |
| `generalise_date` | Extract year only (`"1985-03-15"` → `"1985"`) | Anonymisation |
| `mask_email` | Replace local part (`"alice@example.com"` → `"****@example.com"`) | Anonymisation |

> **`hash` note**: SHA-256 is *deterministic* — the same input always produces the same output. This preserves record linkability within a dataset (useful for joins) but is reversible if the original value is known. Treat hashed values as pseudonymous, not anonymous.

---

## Functions

### `mask_value(value, strategy)` → `string`

Apply a masking strategy to a single string value.

**Inputs**

| Name | Type | Description |
|---|---|---|
| `value` | `string` | The value to mask |
| `strategy` | `string` | One of: `redact`, `hash`, `partial`, `generalise_date`, `mask_email` |

**Output**: the masked string.

**BraneScript example**
```braneScript
import data_masking;

let masked_id := mask_value("PAT-00123", "partial");
println(masked_id);  // "****-00123" (last 4 visible)

let masked_dob := mask_value("1985-03-15", "generalise_date");
println(masked_dob);  // "1985"

let hash_id := mask_value("PAT-00123", "hash");
println(hash_id);  // "3a7b9f..." (64-char hex)
```

---

### `detect_pii(text)` → `string` (JSON)

Scan a text string for likely PII patterns using regex matching. Returns a JSON object listing detected pattern types and up to 3 example matches per type.

**Detected patterns**: `email`, `phone_intl`, `date_iso`, `date_eu`, `ssn_us`, `credit_card`, `postcode_nl`, `iban`.

**Inputs**

| Name | Type | Description |
|---|---|---|
| `text` | `string` | Free-form text to scan |

**Output**: JSON string with structure:
```json
{
  "total_matches": 3,
  "pattern_types_found": 2,
  "findings": {
    "email": ["alice@example.com"],
    "date_iso": ["1985-03-15", "2024-01-01"]
  }
}
```

**BraneScript example**
```braneScript
import data_masking;

let report := detect_pii("Contact alice@example.com, born 1985-03-15.");
println(report);
```

---

### `mask_json_record(record, fields_json)` → `string`

Apply per-field masking strategies to a JSON object string. Non-targeted fields are passed through unchanged.

**Inputs**

| Name | Type | Description |
|---|---|---|
| `record` | `string` | JSON object string (single record) |
| `fields_json` | `string` | JSON array of `{"field": "name", "strategy": "..."}` objects |

**Output**: masked JSON object as a string.

**BraneScript example**
```braneScript
import data_masking;

let record := "{\"name\": \"Alice Smith\", \"dob\": \"1985-03-15\", \"diagnosis\": \"hypertension\"}";
let spec := "[{\"field\": \"name\", \"strategy\": \"redact\"}, {\"field\": \"dob\", \"strategy\": \"generalise_date\"}]";
let masked := mask_json_record(record, spec);
println(masked);
// {"name": "[REDACTED]", "dob": "1985", "diagnosis": "hypertension"}
```

---

### `get_masking_summary(strategy, fields_json)` → `MaskingResult`

Return a `MaskingResult` class instance describing a masking configuration. Useful for inspecting a spec before running it on real data.

**Inputs**

| Name | Type | Description |
|---|---|---|
| `strategy` | `string` | Primary strategy label for the summary |
| `fields_json` | `string` | JSON array of `{"field": "...", "strategy": "..."}` objects |

**Output**: `MaskingResult` instance with `records_processed = 0` and `fields_masked = len(fields)`.

**BraneScript example**
```braneScript
import data_masking;

let spec := "[{\"field\": \"name\", \"strategy\": \"redact\"}, {\"field\": \"dob\", \"strategy\": \"generalise_date\"}]";
let summary := get_masking_summary("mixed", spec);
println(summary.fields_masked);    // 2
println(summary.strategy);         // "mixed"
println(summary.output_format);    // "config_only"
```

---

### `mask_csv_file(csv_file, fields_json)` → `IntermediateResult`

Read a CSV dataset, apply per-column masking, and write output files to `/result/`.

**Inputs**

| Name | Type | Description |
|---|---|---|
| `csv_file` | `Data` | Registered Brane dataset (CSV file or directory containing a CSV) |
| `fields_json` | `string` | JSON array of `{"column": "col_name", "strategy": "..."}` objects |

**Output files in IR**

| File | Description |
|---|---|
| `masked_data.csv` | CSV with targeted columns replaced by masked values |
| `masking_report.json` | Per-column counts, config used, and timestamp |

**BraneScript example**
```braneScript
import data_masking;

let fields := "[{\"column\": \"name\", \"strategy\": \"redact\"}, {\"column\": \"patient_id\", \"strategy\": \"hash\"}, {\"column\": \"dob\", \"strategy\": \"generalise_date\"}]";
let masked := mask_csv_file(heal_pa_2, fields);
```

---

### `mask_json_file(json_file, fields_json)` → `IntermediateResult`

Read a JSON dataset (single object or array of objects), apply per-field masking, and write output files to `/result/`.

**Inputs**

| Name | Type | Description |
|---|---|---|
| `json_file` | `Data` | Registered Brane dataset (JSON file or directory containing a JSON) |
| `fields_json` | `string` | JSON array of `{"field": "field_name", "strategy": "..."}` objects |

**Output files in IR**: same as `mask_csv_file` but with `masked_data.json`.

---

### `generate_masking_report(masking_result)` → `IntermediateResult`

Read the `masking_report.json` from a prior masking IR and produce a formatted HTML audit report.

**Inputs**

| Name | Type | Description |
|---|---|---|
| `masking_result` | `IntermediateResult` | Output of `mask_csv_file` or `mask_json_file` |

**Output files in IR**

| File | Description |
|---|---|
| `masking_audit_report.html` | HTML report with per-field breakdown and GDPR compliance notes |
| `masking_report.json` | Original masking report (passed through) |

**BraneScript example**
```braneScript
import data_masking;

let fields := "[{\"column\": \"name\", \"strategy\": \"redact\"}, {\"column\": \"dob\", \"strategy\": \"generalise_date\"}]";
let masked_ir := mask_csv_file(heal_pa_2, fields);
let audit_ir  := generate_masking_report(masked_ir);
commit_result("masking_audit", audit_ir);
```

---

## `MaskingResult` class

Returned by `get_masking_summary`.

| Field | Type | Description |
|---|---|---|
| `records_processed` | `integer` | Number of records processed (0 for config-only summaries) |
| `fields_masked` | `integer` | Number of fields targeted by the masking spec |
| `strategy` | `string` | Strategy label provided to the function |
| `output_format` | `string` | Output format descriptor (e.g. `config_only`, `csv`, `json`) |

---

## `fields_json` format reference

All multi-field functions accept a `fields_json` string containing a JSON array. Each element specifies one field and its masking strategy:

```json
[
  {"column": "patient_id",  "strategy": "hash"},
  {"column": "name",        "strategy": "redact"},
  {"column": "dob",         "strategy": "generalise_date"},
  {"column": "email",       "strategy": "mask_email"},
  {"column": "ssn",         "strategy": "partial"}
]
```

Use `"column"` for CSV functions and `"field"` for JSON functions (both keys are accepted in either context).

---

## Full pipeline example

Mask the healthcare dataset before analysis:

```braneScript
import data_masking;
import healthcare;

// Step 1: mask sensitive columns
let mask_spec := "[{\"column\": \"patient_id\", \"strategy\": \"hash\"}, {\"column\": \"name\", \"strategy\": \"redact\"}, {\"column\": \"dob\", \"strategy\": \"generalise_date\"}]";
let masked := mask_csv_file(heal_pa_2, mask_spec);

// Step 2: generate audit report
let audit := generate_masking_report(masked);
commit_result("masking_audit", audit);
```

> **Note**: `mask_csv_file` outputs an `IntermediateResult` containing the masked CSV. To feed the masked CSV into a subsequent `Data`-typed function you would need to register it as a new dataset. For now, this pipeline demonstrates the masking and audit steps in isolation.
