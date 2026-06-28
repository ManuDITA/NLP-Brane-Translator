# BraneScript Language Reference

BraneScript is the workflow language for the Brane framework. It orchestrates calls to external packages and datasets across distributed sites.

---

## Variables

Declare with `let`, assign with `:=`. No `=` alone.

```bscript
let x := 42;
let name := "Alice";
let flag := true;
let pi := 3.14;
```

Reassign (no `let` needed for an existing variable):

```bscript
x := x + 1;
name := "Bob";
```

---

## Imports

```bscript
import <package_name>;
```

After importing, call functions directly as `<function>(args)`.

```bscript
import healthcare;

let result := analyze_heart_disease(patient_json);
println(result);
```

---

## Builtin Functions

```bscript
println("Hello, world!");   // print with newline
print("no newline");        // print without newline
```

---

## If / Else

```bscript
if (condition) {
    // true branch
} else {
    // false branch
}
```

Example:

```bscript
let score := 75;
if (score >= 50) {
    println("Pass");
} else {
    println("Fail");
}
```

---

## While Loop

```bscript
let i := 0;
while (i < 5) {
    println(i);
    i := i + 1;
}
```

---

## For Loop

```bscript
for (let i := 0; i < 10; i := i + 1) {
    println(i);
}
```

---

## Functions

```bscript
func <name>(<param>, ...) {
    // body
    return <value>;
}
```

Example:

```bscript
func add(a, b) {
    return a + b;
}

let sum := add(3, 4);
println(sum);
```

Functions with no return value omit the `return` statement:

```bscript
func greet(name) {
    println("Hello, " + name);
}
greet("world");
```

---

## Classes

Define a class with typed fields. Field types can be primitives (`int`, `real`, `bool`, `string`) or other class types.

```bscript
class Jedi {
    name: string;
    is_master: bool;
    lightsaber_colour: string;
}

let obi_wan := new Jedi {
    name              := "Obi-Wan Kenobi",
    is_master         := true,
    lightsaber_colour := "blue",
};
println(obi_wan.name);
```

### Nested / complex class types

For complex structured data, define a separate class for each nested type. **Never use a JSON string to represent structured data** — define a class instead.

```bscript
class VitalSigns {
    blood_pressure: int;
    heart_rate: int;
}

class LabResults {
    total_cholesterol: int;
}

class Patient {
    patient_id: string;
    age: int;
    gender: string;
    vital_signs: VitalSigns;
    lab_results: LabResults;
}

let vitals := new VitalSigns {
    blood_pressure := 150,
    heart_rate     := 80,
};

let labs := new LabResults {
    total_cholesterol := 220,
};

let patient := new Patient {
    patient_id  := "PAT001",
    age         := 55,
    gender      := "M",
    vital_signs := vitals,
    lab_results := labs,
};

println(patient.patient_id);
println(patient.vital_signs.blood_pressure);
```

Classes can also have methods (using `self`):

```bscript
class Counter {
    count: int;

    func increment(self) {
        self.count := self.count + 1;
    }
}
```

---

## Arrays

```bscript
let arr := [1, 2, 3, 4, 5];
println(arr[0]);     // index access
println(arr[2]);
```

Iterate with a for loop:

```bscript
let values := [10, 20, 30];
for (let i := 0; i < 3; i := i + 1) {
    println(values[i]);
}
```

---

## Parallel Execution

Run multiple branches at once and merge results:

```bscript
parallel [all] [{
    return branch_one();
}, {
    return branch_two();
}];
```

With merge strategies `all` (wait for all) or `first` (take first result):

```bscript
let results := parallel [all] [{
    return analyze(data1);
}, {
    return analyze(data2);
}];
```

---

## Data / Datasets and Intermediate Results

Brane separates *variables* (in-memory values) from *data* (files/large datasets on disk).

### `Data` — reference to a registered dataset

Use the builtin `Data` class to reference a dataset by name. It has exactly one field: `name`.

```bscript
let ds := new Data { name := "my-dataset" };
let result := process(ds);
```

`Data` can also be created inline:

```bscript
let result := process(new Data { name := "my-dataset" });
```

### `IntermediateResult` — output of a package function that produces data

When a package function outputs a file or dataset, it returns an `IntermediateResult`.
You **cannot** create an `IntermediateResult` yourself — it is always the return value of a package call.

```bscript
import copy_result;

let ds := new Data { name := "colours" };
let copy := copy_result(ds);   // copy is an IntermediateResult
```

An `IntermediateResult` can be passed as input to another package function, just like `Data`.

### `commit_result` — persisting an IntermediateResult as a dataset

`IntermediateResult` values are scoped to the workflow. To make one persist beyond the workflow, commit it:

```bscript
commit_result("new-dataset-name", result_variable);
```

Full example — load a dataset, process it, save the output:

```bscript
import copy_result;
import cat;

let raw := new Data { name := "colours" };
println(cat(raw, "-"));             // print original

let copy := copy_result(raw);       // returns IntermediateResult
println(cat(copy, "contents"));     // print copy

commit_result("colours_copy", copy); // persist as a new dataset
```

### Key rules

- `Data` → input only (reference to existing dataset)
- `IntermediateResult` → output of a package function that produces data; cannot be created by user
- `commit_result("name", result)` → save an `IntermediateResult` permanently
- You cannot inspect the contents of a `Data` or `IntermediateResult` in BraneScript — they are opaque references; use a package function (e.g. `cat`) to read them
- In `container.yml`, functions declare `type: Data` for dataset inputs and `type: IntermediateResult` for data outputs

---

## Attributes

Tag a call or block to route it to a specific site:

```bscript
#[on("Amy")]
let result := compute(input);
```

Apply to a whole block:

```bscript
#[on("site-a")]
{
    let r1 := step1(data);
    let r2 := step2(r1);
}
```

---

## Return

Return a value from a function or workflow:

```bscript
func double(n) {
    return n * 2;
}
```

Use `return;` (no value) for early exit from a function.

---

## Types

| Type                 | Example / Notes                                      |
|----------------------|------------------------------------------------------|
| `int`                | `42`, `-7`                                           |
| `real`               | `3.14`, `-0.5`                                       |
| `bool`               | `true`, `false`                                      |
| `string`             | `"hello"`                                            |
| `Data`               | `new Data { name := "ds" }` — reference to a dataset |
| `IntermediateResult` | returned by package functions; use `commit_result("name", result)` to persist |
| Array                | `[1, 2, 3]`                                          |

---

## Complete Example — Healthcare Analysis

```bscript
import healthcare;

class VitalSigns {
    blood_pressure: int;
    heart_rate: int;
}

class LabResults {
    total_cholesterol: int;
}

class Patient {
    patient_id: string;
    age: int;
    gender: string;
    vital_signs: VitalSigns;
    lab_results: LabResults;
}

let vitals := new VitalSigns {
    blood_pressure := 150,
    heart_rate     := 80,
};

let labs := new LabResults {
    total_cholesterol := 220,
};

let patient := new Patient {
    patient_id  := "PAT001",
    age         := 55,
    gender      := "M",
    vital_signs := vitals,
    lab_results := labs,
};

let risk := analyze_heart_disease(patient);
println(risk);

let report := generate_report(patient);
println(report);
```

---

## Complete Example — Parallel Workflow with Function

```bscript
import compute;

func process_item(item) {
    let result := run(item);
    return result;
}

let a := "input_a";
let b := "input_b";

let results := parallel [all] [{
    return process_item(a);
}, {
    return process_item(b);
}];

println(results[0]);
println(results[1]);
```
