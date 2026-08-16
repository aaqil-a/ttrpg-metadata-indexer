# TTRPG Adventure Indexer
Ingest TTRPG adventure metadata from a variety of data sources.

Index them into Markdown files based on metadata fields for easier and knowledge-base-friendly navigation.

## Output 
```bash
# Sample Structure
index/
├── adventures/ # Metadata of ingested adventures
│   ├── a-giant-problem.md
│   └── chambers-of-antiquities.md
│   └── ...
├── environments/ # Index files, pointing to adventure files
│   ├── dungeon.md 
│   ├── jungle.md
│   └── ...
├── start_level/ # Index files, pointing to adventure files
│   ├── 1.md
│   ├── 2.md
│   └── ...
├── ...
└── index.md # Entrypoint, points to index files
```

## Data Sources
See [downloaders](src/indexer/downloader) for implementation.
1. [AdventureLookup](https://www.adventurelookup.com/) using [public API](https://www.adventurelookup.com/api)
2. [5etools](https://5e.tools/) using data from public [GitHub repository](https://github.com/5etools-mirror-3/5etools-src)

## Requirements
- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) package manager
- Optionally [sqlite3](https://sqlite.org/), to inspect ingested data

## Usage

### 0. Setup Project
```bash
uv sync
```

### 1. Ingest Data
Ingests adventures from data sources into a sqlite database. 
```bash
ingest
ingest --help
```

#### 1a. Infer
Optional step to infer missing labels based on existing data using simple [TFIDF logistic regression](src/indexer/tfidf.py). Currently used to infer `environments` for adventures missing the label.
```bash
infer
infer --help

# Only train
tfidf train
tfidf train --eval-split 0.2 # Evaluate model
tfidf train --help

# Only infer
tfidf infer
tfidf infer --no-update --output /path/to/result/json # Dry run
tfidf infer --help
```

### 2. Index
Create index Markdown files based on metadata fields as declared in [types.py](src/indexer/types.py) (to be extended).
```bash
index
index --on environments start_level
index --help
```
