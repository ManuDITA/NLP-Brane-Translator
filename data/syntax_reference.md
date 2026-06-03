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

After importing, call functions via `<package>::<function>(args)`.

```bscript
import healthcare;

let result := healthcare::analyze_heart_disease(patient_json);
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
func <name>(<param>: <type>, ...) -> <return_type> {
    // body
    return <value>;
}
```

Example:

```bscript
func add(a: int, b: int) -> int {
    return a + b;
}

let sum := add(3, 4);
println(sum);
```

Functions with no return value use `-> unit` or omit the return type:

```bscript
func greet(name: string) {
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

## Data / Datasets

Reference a dataset registered in the Brane instance:

```bscript
let ds := new Data { name := "my-dataset" };
let output := somepackage::process(ds);
```

---

## Attributes

Tag a call or block to route it to a specific site:

```bscript
#[on("Amy")]
let result := somepackage::compute(input);
```

Apply to a whole block:

```bscript
#[on("site-a")]
{
    let r1 := pkg::step1(data);
    let r2 := pkg::step2(r1);
}
```

---

## Return

Return a value from a function or workflow:

```bscript
func double(n: int) -> int {
    return n * 2;
}
```

Use `return;` (no value) for early exit from a `unit` function.

---

## Types

| Type     | Example literal      |
|----------|----------------------|
| `int`    | `42`, `-7`           |
| `real`   | `3.14`, `-0.5`       |
| `bool`   | `true`, `false`      |
| `string` | `"hello"`            |
| `Data`   | `new Data { name := "ds" }` |
| Array    | `[1, 2, 3]`          |

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

let risk := healthcare::analyze_heart_disease(patient);
println(risk);

let report := healthcare::generate_report(patient);
println(report);
```

---

## Complete Example — Parallel Workflow with Function

```bscript
import compute;

func process_item(item: string) -> string {
    let result := compute::run(item);
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
