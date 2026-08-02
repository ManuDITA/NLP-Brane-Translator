# Datetime Package

Brane package for retrieving and formatting the current date and time in various formats.

All functions are **zero-input** — they return the current moment in different representations. No dataset reference required.

## Functions

### `get_iso()`

Return the current local datetime in ISO 8601 format with UTC offset.

**Returns** string:
```
"2026-06-14T22:58:17+02:00"
```

**BraneScript usage:**
```
import datetime;
let iso_time := get_iso();
println(iso_time);
```

---

### `get_date()`

Return today's date as `YYYY-MM-DD`.

**Returns** string:
```
"2026-06-14"
```

**BraneScript usage:**
```
import datetime;
let today := get_date();
println("Today is: " + today);
```

---

### `get_time()`

Return the current local time as `HH:MM:SS` (24-hour format).

**Returns** string:
```
"22:58:17"
```

**BraneScript usage:**
```
import datetime;
let now := get_time();
println(now);
```

---

### `get_human()`

Return the current datetime in a human-readable format (day name, month name, year, time, AM/PM).

**Returns** string:
```
"Saturday, June 14 2026 10:58 PM"
```

**BraneScript usage:**
```
import datetime;
let readable := get_human();
println("Current time: " + readable);
```

---

### `get_unix()`

Return the current UTC Unix timestamp as a decimal string (seconds since epoch).

**Returns** string:
```
"1749945497"
```

**BraneScript usage:**
```
import datetime;
let timestamp := get_unix();
println("Timestamp: " + timestamp);
```

---

### `get_formatted(format_str)`

Return the current local datetime using a custom `strftime` format string.

**Input:**
- `format_str` (string) — a `strftime` format pattern (e.g. `"%d/%m/%Y %H:%M"`)

**Returns** string:
```
"14/06/2026 22:58"  (if format_str = "%d/%m/%Y %H:%M")
```

**Common strftime codes:**
| Code | Meaning | Example |
|---|---|---|
| `%Y` | 4-digit year | 2026 |
| `%m` | 2-digit month | 06 |
| `%d` | 2-digit day | 14 |
| `%H` | 2-digit hour (24h) | 22 |
| `%M` | 2-digit minute | 58 |
| `%S` | 2-digit second | 17 |
| `%A` | Full day name | Saturday |
| `%B` | Full month name | June |
| `%I` | Hour (12h, 01–12) | 10 |
| `%p` | AM/PM | PM |

**BraneScript usage:**
```
import datetime;
let custom := get_formatted("%d/%m/%Y %I:%M %p");
println(custom);  // Output: 14/06/2026 10:58 PM
```

---

## Examples

### Log current timestamp with context
```
import datetime;

let date := get_date();
let time := get_time();
let msg := "Analysis started at " + date + " " + time;
println(msg);
```

### Generate a filename with timestamp
```
import datetime;

let ts := get_formatted("%Y%m%d_%H%M%S");
let filename := "report_" + ts + ".txt";
println("Saving to: " + filename);
```

### Check if an action runs within business hours
```
import datetime;

let hour_str := get_formatted("%H");
let hour := /* need to parse string → int */;
if (hour >= 9 && hour <= 17) {
    println("Business hours");
} else {
    println("Outside business hours");
}
```

## Implementation notes

- All functions return the **local time** of the Brane instance where they run.
- Times are **not** synchronized across multiple sites; each site reports its own local time.
- `get_human()` uses the system locale for day/month names.
- `get_formatted()` delegates to Python's `strftime()` — invalid format strings will raise an error.

## Use cases

- **Timestamping logs** — append current time to workflow output
- **Workflow scheduling** — check time to decide whether to proceed
- **Report naming** — include date/time in generated file names
- **Audit trails** — record when decisions were made
