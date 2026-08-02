# Text Analysis Package

Brane package for NLP text analysis: word/sentence counting, keyword extraction,
sentiment classification, readability scoring, and batch processing of text files.
Pure Python stdlib — no heavy NLP dependencies.

---

## How Brane invokes this package

- **Action dispatch** — the action name is passed as `argv[1]` via `command.args` in `container.yml`. `text_analysis.py` reads `sys.argv[1]` to dispatch to the correct handler.
- **Input arguments** — branelet JSON-serialises each input and sets it as an uppercase env var (e.g. input `text` → env var `TEXT` containing `"\"Hello world\""`).
- **String / integer outputs** — printed to stdout as `output: <value>`. Strings are JSON-quoted; integers are plain numbers.
- **Class outputs** — printed as `output: ["ClassName", {fields}]`, deserialised by serde as `FullValue::Instance`.
- **Data / IntermediateResult outputs** — written to `/result/` on disk; nothing printed to stdout.

---

## Custom classes

### `TextStats`

Returned by `get_text_stats`. Gives direct field access in BraneScript.

| Field | Type | Description |
|---|---|---|
| `word_count` | integer | Number of word tokens |
| `sentence_count` | integer | Number of sentences |
| `avg_word_length` | real | Average characters per word |
| `readability_score` | real | Flesch Reading Ease (0–100) |
| `sentiment` | string | `positive` / `negative` / `neutral` |

**BraneScript usage:**
```
import text_analysis;
let s := get_text_stats("The quick brown fox jumps over the lazy dog.");
println(s.word_count);
if (s.sentiment == "positive") {
    println("The text is positive!");
}
```

### `KeywordResult`

Metadata for a single extracted keyword (returned as elements inside the JSON array from `extract_keywords`).

| Field | Type | Description |
|---|---|---|
| `keyword` | string | The keyword |
| `frequency` | integer | Raw occurrence count |
| `score` | real | TF score (frequency / content words) |

---

## Functions

### String-based (inline text via `text` argument)

#### `count_words(text)`

Count all word tokens (alphabetic sequences).

**Returns** integer.

```
import text_analysis;
let n := count_words("Hello world, this is a test.");
println(n);   // 7
```

---

#### `count_sentences(text)`

Count sentences split on `.`, `!`, `?`.

**Returns** integer.

---

#### `extract_keywords(text, top_n)`

Extract the top `top_n` keywords by term-frequency after stop-word removal.

**Returns** JSON string — an array of objects:
```json
[
  { "keyword": "brane", "frequency": 12, "score": 0.0923 },
  { "keyword": "workflow", "frequency": 8, "score": 0.0615 }
]
```

**BraneScript usage:**
```
import text_analysis;
let kws := extract_keywords("Brane is a workflow engine for distributed computing...", 5);
println(kws);
```

---

#### `compute_sentiment(text)`

Lexicon-based sentiment with negation handling (e.g. "not good" → negative).

**Returns** JSON string:
```json
{ "polarity": "positive", "score": 0.42, "word_count": 120 }
```

- `score` is normalised to [-1.0, 1.0].
- `polarity`: `positive` (score > 0.05), `negative` (score < -0.05), `neutral` otherwise.

---

#### `compute_readability(text)`

Flesch Reading Ease formula: `206.835 − 1.015 × (words/sentences) − 84.6 × (syllables/words)`.

**Returns** JSON string:
```json
{ "flesch_score": 68.4, "grade_level": "8th–9th grade (standard)" }
```

| Score | Grade level |
|---|---|
| 90–100 | 5th grade (very easy) |
| 80–89 | 6th grade (easy) |
| 70–79 | 7th grade (fairly easy) |
| 60–69 | 8th–9th grade (standard) |
| 50–59 | 10th–12th grade (fairly difficult) |
| 30–49 | College level (difficult) |
| 0–29 | Professional (very difficult) |

---

### Class-returning function

#### `get_text_stats(text)`

Returns a `TextStats` instance with all key metrics in one call.

**BraneScript usage:**
```
import text_analysis;
let s := get_text_stats("Climate change is a pressing global issue...");
println(s.readability_score);
println(s.sentiment);
```

---

### Data-based function

#### `analyze_text_file(text_file)`

Accepts a `Data` reference pointing to a plain-text `.txt` file (or a directory of `.txt` files).

**Output files in `/result/`:**
- `text_analysis.json` — per-file breakdown with word count, sentences, readability, sentiment, top keywords, and word frequencies

**BraneScript usage:**
```
import text_analysis;
let corpus := new Data{ name := "news_articles" };
let analysis := analyze_text_file(corpus);
```

---

### IntermediateResult chaining

#### `generate_frequency_report(analysis)`

Reads the `IntermediateResult` from `analyze_text_file` and produces a
word-frequency bar chart (HTML) and a full frequency CSV.

**Output files in `/result/`:**
- `word_frequencies.csv` — all words with their frequencies, sorted descending
- `report.html` — per-file summary table + top-50 word frequency bar chart

**BraneScript usage:**
```
import text_analysis;
let corpus   := new Data{ name := "news_articles" };
let analysis := analyze_text_file(corpus);
let report   := generate_frequency_report(analysis);
commit_result("news_report", report);
```

---

## Notes on analysis

- **Tokenisation** — splits on `[a-zA-Z']+`; punctuation is discarded.
- **Stop-words** — ~70 common English words are excluded from keyword extraction and scoring.
- **Sentiment lexicon** — ~100 positive and ~50 negative English words with integer weights. Negators (`not`, `never`, `n't`, ...) flip the sign of the next sentiment word.
- **Syllable counting** — vowel-group heuristic (approximate; works well for common English words).
- **No external dependencies** — uses Python stdlib only (`re`, `csv`, `json`, `math`, `os`, `yaml`).
