# Text Analysis Dataset

A collection of diverse English-language documents for testing the `text_analysis` Brane package.

## Contents

```
text_analysis/
├── data.yml                        ← Brane dataset descriptor
├── texts/                          ← 6 sample documents
│   ├── simple_story.txt            ← 150-word narrative
│   ├── news_article_ai_policy.txt  ← 400-word news piece
│   ├── product_review_positive.txt ← 100-word positive review
│   ├── product_review_negative.txt ← 120-word negative review
│   ├── nlp_research.txt            ← 350-word technical abstract
│   └── healthcare_overview.txt     ← 300-word health article
└── workflows/
    └── full_pipeline.bs            ← Example workflow: load → analyze → commit
```

## Registered name

This dataset is registered in Brane as **`text_corpus`**.

To reference it in BraneScript:
```
let corpus := new Data { name := "text_corpus" };
```

## Document overview

| File | Word count | Topic | Tone |
|---|---|---|---|
| `simple_story.txt` | 150 | Fiction narrative | Conversational |
| `news_article_ai_policy.txt` | 400 | AI policy | Formal, journalistic |
| `product_review_positive.txt` | 100 | Product feedback | Positive, casual |
| `product_review_negative.txt` | 120 | Product feedback | Negative, critical |
| `nlp_research.txt` | 350 | Machine learning | Technical, academic |
| `healthcare_overview.txt` | 300 | Health/medicine | Informative, formal |

**Total:** ~1400 words across 6 diverse documents.

## Text analysis functions available

The `text_analysis` package exposes functions that accept this dataset:

- **`count_words(corpus)`** — total word count across all documents
- **`analyze_sentiment(corpus)`** — sentiment classification per document
- **`extract_entities(corpus)`** — named entity recognition (persons, organizations, locations)
- **`summarize_corpus(corpus, lines=3)`** — extractive summary of each document

See `packages/text_analysis/README.md` for full function signatures and examples.

## Example workflow

```branescript
import text_analysis;

let corpus := new Data { name := "text_corpus" };

// Analyze the entire corpus
let word_count := count_words(corpus);
println("Total words: " + word_count);

let sentiment := analyze_sentiment(corpus);
println("Sentiment analysis complete");

// Commit the result for downstream processing
let analysis := sentiment;
commit_result("text_analysis_results", analysis);
```

## How to use

1. **Register the dataset:**
   ```bash
   brane data build datasets/text_analysis/data.yml
   ```

2. **Run a workflow:**
   ```bash
   brane workflow run datasets/text_analysis/workflows/full_pipeline.bs
   ```

3. **Access in BraneScript:**
   ```branescript
   let corpus := new Data { name := "text_corpus" };
   let results := count_words(corpus);
   ```

## File format

Each `.txt` file contains plain UTF-8 text. No special formatting or structure — simple newline-separated paragraphs.

Example excerpt from `simple_story.txt`:
```
The old bookshop sat quietly on the corner, its wooden shelves groaning under 
the weight of stories collected over decades. Sarah pushed through the heavy door, 
the bell above jangling a familiar greeting...
```

## Use cases

- **Text classification testing** — sentiment across different genres and tones
- **NLP pipeline validation** — end-to-end text processing workflows
- **Entity extraction evaluation** — diverse entity types (persons, places, organizations)
- **Summarization benchmarks** — documents of varying length and complexity
