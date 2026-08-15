# TTRPG Adventure Indexer
Ingest TTRPG adventure metadata from a variety of data sources.

Index them into Markdown files based on metadata fields for knowledge-base-friendly navigation.

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
See [downloaders](src/repo/downloader) for implementation.
1. [AdventureLookup](https://www.adventurelookup.com/) using [public API](https://www.adventurelookup.com/api)
2. [5etools](https://5e.tools/) using data from public [GitHub repository](https://github.com/5etools-mirror-3/5etools-src)

## Requirements
- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) package manager
- Optionally [sqlite3](https://sqlite.org/), to inspect ingested data

## Usage
### Ingest Data
Overwrites database file if already exists.
```bash
uv run ingest
uv run ingest --db /path/to/sqlite/db
```

### Infer
Optional step to infer missing labels for adventure's `environments` based on existing data using simple [TFIDF logistic regression](src/repo/tfidf.py). Requires some data to already be ingested.
```bash
uv run infer
uv run infer --db /path/to/sqlite/db --model /path/to/save/inference/model

# Only train
uv run tfidf train
uv run tfidf train --db /path/to/sqlite/db --model /path/to/save/inference/model

# Only infer
uv run tfidf infer
uv run tfidf infer --no-update --output /path/to/save/inference/result/json
```

### Index
```bash
uv run index
uv run index --db /path/to/sqlite/db --dir /path/to/index/files/directory
```
