# Pinecone Ingestion Toolkit — PDF & CSV → Vector Search

A small Python toolkit that turns your documents into a searchable Pinecone index:

1. Extracts text from a **PDF** (page by page) or a **CSV** (row by row).
2. Splits the text into overlapping chunks so context is not lost at the borders.
3. Embeds the chunks and **upserts** them into a Pinecone index — either a
   *document index* (Pinecone embeds for you) or a *classic vector index*
   (the script embeds locally via Pinecone Inference).
4. Lets you run **semantic search** over the uploaded content from the CLI.

No server, no database — just three scripts and a `.env` file.

---

## Table of contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Index a PDF](#index-a-pdf)
  - [Index a CSV](#index-a-csv)
  - [Search the index](#search-the-index)
- [Supported index types](#supported-index-types)
- [What gets stored](#what-gets-stored)
- [Tuning](#tuning)
- [Troubleshooting](#troubleshooting)
- [Security notes](#security-notes)

---

## How it works

```
PDF / CSV file
      │
      ▼
  extract text          pypdf (pages)  ·  csv (rows)
      │
      ▼
  chunk + overlap       greedy packing, CHUNK_SIZE / CHUNK_OVERLAP
      │
      ▼
  embed + upsert        document index  → Pinecone embeds the text
                        vector index    → pc.inference.embed, then upsert
      │
      ▼
  semantic search       query_pdf.py "your question"
```

Each chunk is stored with enough metadata (`source`, `page` / `row_index`,
`chunk_index`) that search results can always be traced back to their origin.

---

## Project structure

| File | Purpose |
| --- | --- |
| `embed_pdf.py` | Extract → chunk → embed → upsert a **PDF** into Pinecone |
| `embed_csv.py` | Extract → chunk → embed → upsert a **CSV** into Pinecone |
| `query_pdf.py` | Semantic search over whatever has been uploaded |
| `example.env` | Template for your environment variables — copy it to `.env` |
| `requirements.txt` | Python dependencies |
| `Automatic accident detection and alert system .pdf` | Sample PDF (default input) |
| `EPL_Match_Data_2000_2026.csv` | Sample CSV (default input) |

---

## Requirements

- **Python 3.10+** (uses `X | None` type hints and `list[str]`)
- A **Pinecone account** and API key — [app.pinecone.io](https://app.pinecone.io)

---

## Setup

### 1. Get the code

```bash
git clone <your-repo-url>
cd p3
```

### 2. Create a virtual environment (recommended)

**Windows — PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

**Windows — CMD**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file

The repository ships `example.env` (safe to commit) but **not** `.env`
(your real secrets — it is listed in `.gitignore`). Create your own copy:

**Windows — PowerShell**

```powershell
Copy-Item example.env .env
```

**Windows — CMD**

```bat
copy example.env .env
```

**macOS / Linux**

```bash
cp example.env .env
```

Then open `.env` and fill in at least your API key:

```dotenv
PINECONE_API_KEY=your-real-key-here
PINECONE_INDEX=my-index
```

> `.env` is never committed. If you ever lose it, just repeat this step and
> re-enter your values — `example.env` always has the full list of variables.

---

## Configuration

All settings are read from `.env` at startup.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `PINECONE_API_KEY` | ✅ | — | Your Pinecone API key |
| `PINECONE_INDEX` | — | `donbosco` | Name of the target index |
| `PINECONE_NAMESPACE` | — | `default` | Namespace to read/write. **Must be non-empty** for document indexes |
| `PINECONE_HOST` | — | *(blank)* | Optional index host URL. Leave blank to resolve it from the index name |
| `PINECONE_EMBED_MODEL` | — | `multilingual-e5-large` | Embedding model, used **only** for classic vector indexes |
| `PINECONE_CLOUD` | — | *(blank)* | e.g. `aws` — needed only to auto-create a missing index |
| `PINECONE_REGION` | — | *(blank)* | e.g. `us-east-1` — needed only to auto-create a missing index |
| `PDF_PATH` | — | sample PDF | Default input when no path is passed to `embed_pdf.py` |
| `CSV_PATH` | — | sample CSV | Default input when no path is passed to `embed_csv.py` |
| `CHUNK_SIZE` | — | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | — | `100` | Characters of overlap carried into the next chunk |

---

## Usage

### Index a PDF

```powershell
python embed_pdf.py "Automatic accident detection and alert system .pdf"
```

Omit the path to use `PDF_PATH` from `.env`:

```powershell
python embed_pdf.py
```

Preview the chunks **without** contacting Pinecone:

```powershell
python embed_pdf.py --dry-run
```

### Index a CSV

```powershell
python embed_csv.py "EPL_Match_Data_2000_2026.csv"
```

Omit the path to use `CSV_PATH` from `.env`:

```powershell
python embed_csv.py
```

Preview only:

```powershell
python embed_csv.py --dry-run
```

> A CSV header row is skipped. Every data row is converted to a single text
> string (values joined by spaces) and indexed together with its `row_index`.

### Search the index

```powershell
python query_pdf.py "what happens if the driver does not respond within 10 seconds?"
```

Limit the number of results:

```powershell
python query_pdf.py "buzzer timeout" -k 3
```

Output looks like:

```
Top 3 matches for: 'buzzer timeout'

[1] score=0.8412  source=Automatic accident detection and alert system .pdf  page=7
    ...the buzzer is activated and the alert is sent after a 10 second timeout...
```

> `query_pdf.py` searches whichever index and namespace are in `.env`, so it
> works for CSV content too. Wait a few seconds after uploading before
> searching — new records take a moment to become searchable.

---

## Supported index types

The scripts call `pc.describe_index(...)` and adapt automatically:

| Index type | How it is detected | What the script does |
| --- | --- | --- |
| **Document index** (integrated embedding) | a schema field declares `embed` | Sends text via `index.documents.batch_upsert(...)` — **Pinecone embeds it** |
| **Classic vector index** | schema only has `_values` | Embeds locally with `pc.inference.embed(...)`, then `index.upsert(vectors=...)` |

Connecting follows the Pinecone quickstart — only the index **name** is needed:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("donbosco")
```

If the index does not exist yet, the script creates it automatically **only
when** `PINECONE_CLOUD` and `PINECONE_REGION` are set; otherwise create it in
the Pinecone console first.

---

## What gets stored

| Index type | Text field | Metadata |
| --- | --- | --- |
| Document | the schema's embedded field, e.g. `chunk_text` | `source`, `page` (PDF) or `row_index` (CSV), `chunk_index` |
| Classic vector | `metadata.text` | `source`, `page` (PDF) or `row_index` (CSV), `chunk_index` |

Vector IDs are derived from the file name plus the chunk number
(`automatic-accident-detection-and-alert-system-12`), so re-uploading the same
file overwrites its previous chunks instead of duplicating them.

---

## Tuning

| Setting | Default | Meaning |
| --- | --- | --- |
| `CHUNK_SIZE` | `800` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `100` | Characters shared between consecutive chunks |
| `PINECONE_EMBED_MODEL` | `multilingual-e5-large` | Embed model for classic vector indexes |

> Chunking is by **characters**, not tokens. If you switch to a model with a
> smaller token limit, keep `CHUNK_SIZE` comfortably below it — roughly
> 4 characters per token for English text.

Typical tweaks:

- Answers feel cut off → increase `CHUNK_SIZE` or `CHUNK_OVERLAP`.
- Retrieval returns too much irrelevant text → decrease `CHUNK_SIZE`.
- Embedding calls fail with token-limit errors → decrease `CHUNK_SIZE`.

---

## Troubleshooting

| Problem | Likely cause / fix |
| --- | --- |
| `PINECONE_API_KEY is missing` | No `.env` file, or the key line is empty — repeat [step 4](#4-create-your-env-file) |
| `Index '...' does not exist` | Create it in the Pinecone console, or set `PINECONE_CLOUD` and `PINECONE_REGION` to allow auto-creation |
| `Document indexes require a non-empty namespace` | Set `PINECONE_NAMESPACE` in `.env` (e.g. `default`) |
| `Index '...' has dimension X but model '...' produces Y` | `PINECONE_EMBED_MODEL` does not match the index dimension — change the model or use a different index |
| `No text could be extracted` | The PDF is scanned/image-only and needs OCR before ingesting |
| `PDF not found` / `CSV not found` | The path is wrong or contains unquoted spaces — wrap it in quotes |
| Search returns nothing right after upload | Wait a few seconds, then retry |
| Import errors for `pinecone` | Activate the virtual environment and re-run `pip install -r requirements.txt` |
| PowerShell blocks venv activation | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

---

## Security notes

- **Never commit `.env`** — it contains your API key. It is already listed in
  `.gitignore`; only `example.env` with placeholder values belongs in the repo.
- Keep `pinecone.txt` out of version control as well (also gitignored).
- If a key was ever committed or shared, rotate it in the Pinecone console.
- `.env` files with real credentials should not be pasted into issues or chats.

---

## License

No license file is included yet. Add one (e.g. MIT) if you plan to share or
reuse this project publicly.
