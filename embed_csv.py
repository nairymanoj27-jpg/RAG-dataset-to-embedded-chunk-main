"""
Chunk a CSV file and embed it into a Pinecone vector database.

Pipeline:
  1. Extract text row-by-row from the CSV.
  2. Split the text into overlapping chunks so context isn't lost at borders.
  3. Embed the chunks and upsert them into your Pinecone index.

The script automatically handles BOTH kinds of Pinecone index:

  * classic vector index  -> we embed with Pinecone Inference
                             (pc.inference.embed) and call index.upsert(vectors=...)
  * document index with integrated embedding (schema field has `embed`)
                          -> we send the text and Pinecone embeds it for us
                             (index.documents.batch_upsert(...))

Connect exactly like the Pinecone quickstart - only the index name is needed:

    pc = Pinecone(api_key="...")
    index = pc.Index("donbosco")

Usage:
    python embed_csv.py                       # uses CSV_PATH from .env / default below
    python embed_csv.py "my_data.csv"       # or pass a path on the command line
    python embed_csv.py --dry-run             # chunk only, print results, no Pinecone calls

Env vars (see .env):
    PINECONE_API_KEY      required
    PINECONE_INDEX        index name                 (default: donbosco)
    PINECONE_HOST         optional host; blank -> resolved from the index name
    PINECONE_NAMESPACE    namespace; must be non-empty for document indexes (default: default)
    PINECONE_EMBED_MODEL  embed model for classic vector indexes (default: multilingual-e5-large)
    PINECONE_CLOUD        used to auto-create a missing index (e.g. aws)
    PINECONE_REGION       used to auto-create a missing index (e.g. us-east-1)
    CHUNK_SIZE            max characters per chunk    (default: 800)
    CHUNK_OVERLAP         overlap between chunks      (default: 100)
    CSV_PATH              path to the CSV to index  (default: EPL_Match_Data_2000_2026.csv)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import csv

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------------------------------------------------------------- Config
DEFAULT_CSV_PATH = "EPL_Match_Data_2000_2026.csv"

PINECONE_API_KEY = (os.getenv("PINECONE_API_KEY") or "").strip()
PINECONE_HOST = (os.getenv("PINECONE_HOST") or "").strip()
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "donbosco").strip()
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default").strip()
PINECONE_EMBED_MODEL = os.getenv("PINECONE_EMBED_MODEL", "multilingual-e5-large").strip()

PINECONE_CLOUD = (os.getenv("PINECONE_CLOUD") or "").strip()
PINECONE_REGION = (os.getenv("PINECONE_REGION") or "").strip()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
EMBED_BATCH_SIZE = 96  # Pinecone hosted models accept at most 96 inputs per request

# Placeholder that ships in .env; if it is still there we ignore it.
_PLACEHOLDER_HOST = "your-pinecone-host"


# ---------------------------------------------------------------- CSV text extraction
def extract_rows(csv_path: Path) -> list[str]:
    """Return the string representation of each row in order."""
    rows: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            # Convert row to text: join non-empty values with space
            values = [str(v).strip() for v in row if v is not None]
            text = " ".join(values)
            rows.append(text)
    return rows


# ---------------------------------------------------------------- Text chunking
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _slug(text: str) -> str:
    """Make an ID-safe slug out of a text snippet."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-") or "doc"


def _tail_overlap(text: str, overlap: int) -> str:
    """Return the last `overlap` characters of `text`, for context carry-over."""
    return text[-overlap:].strip() if overlap > 0 and len(text) > overlap else ""


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks.

    Strategy: split into paragraphs; paragraphs that are still too long are
    split into sentences; pieces are then greedily packed up to ``chunk_size``,
    with the tail of each finished chunk carried into the next one.
    """
    text = _normalize(text)
    if not text:
        return []

    # For CSV rows, we treat each row as a paragraph and pack them
    paragraphs = [text] if text.strip() else []

    # Break each paragraph into atomic pieces small enough to place in a chunk.
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            pieces.append(para)
            continue
        # Split long paragraphs into sentences
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= chunk_size:
                pieces.append(sentence)
            else:
                # A single "sentence" is still too long -> hard cut it.
                for i in range(0, len(sentence), chunk_size):
                    pieces.append(sentence[i : i + chunk_size])

    # Greedily pack pieces into chunks.
    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        if buffer and len(buffer) + len(piece) + 1 > chunk_size:
            chunks.append(buffer.strip())
            buffer = _tail_overlap(buffer, chunk_overlap)
        buffer = f"{buffer} {piece}".strip() if buffer else piece
    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if c]


def chunk_rows(rows: list[str]) -> list[dict]:
    """Chunk CSV rows while remembering row index.

    Each row becomes a chunk. If a row is too long, it gets split.
    """
    chunks: list[dict] = []
    for row_idx, row_text in enumerate(rows, start=1):
        row_text = _normalize(row_text)
        if not row_text:
            continue
        row_chunks = chunk_text(row_text)
        for text_chunk in row_chunks:
            chunks.append({"text": text_chunk, "row_index": row_idx})
    return chunks


# ---------------------------------------------------------------- Pinecone helpers
def get_index(pc: Pinecone, index_name: str):
    """Open a data-plane client, mirroring ``pc.Index("name")`` from the quickstart."""
    if PINECONE_HOST and _PLACEHOLDER_HOST not in PINECONE_HOST:
        return pc.Index(index_name, host=PINECONE_HOST)
    return pc.Index(index_name)


def ensure_index(pc: Pinecone, index_name: str) -> None:
    """Create a classic vector index if it is missing (needs cloud + region)."""
    if pc.has_index(index_name):
        return
    if not (PINECONE_CLOUD and PINECONE_REGION):
        raise ValueError(
            f"Index '{index_name}' does not exist.\n"
            "Create it in the Pinecone console, or set PINECONE_CLOUD and "
            "PINECONE_REGION in .env (e.g. aws / us-east-1) to auto-create it."
        )
    info = pc.inference.model.get(PINECONE_EMBED_MODEL)
    metric = "cosine" if "cosine" in (info.supported_metrics or []) else "euclidean"
    print(f"Creating vector index '{index_name}' ({info.default_dimension} dims, {metric})...")
    pc.create_index(
        name=index_name,
        dimension=info.default_dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )


def describe_target(pc: Pinecone, index_name: str) -> dict:
    """Work out how to write to this index: integrated-embedding or classic vectors."""
    desc = pc.describe_index(index_name)
    fields = getattr(getattr(desc, "schema", None), "fields", None) or {}
    for field_name, field in fields.items():
        embed_cfg = getattr(field, "embed", None)
        if embed_cfg is not None:
            return {
                "mode": "documents",
                "text_field": field_name,
                "model": getattr(embed_cfg, "model", None),
                "dimension": getattr(embed_cfg, "dimension", None) or desc.dimension,
            }
    return {"mode": "vectors", "dimension": desc.dimension}


def embed_texts(pc: Pinecone, texts: list[str], model: str) -> list[list[float]]:
    """Embed texts with Pinecone Inference, batching to respect the request limit."""
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        result = pc.inference.embed(
            model=model,
            inputs=batch,
            parameters={"input_type": "passage", "truncate": "END"},
        )
        vectors.extend(item.values for item in result.data)
        print(f"  embedded {min(start + EMBED_BATCH_SIZE, total)}/{total}")
    return vectors


def upload_documents(index, chunks: list[dict], csv_path: Path, target: dict) -> int:
    """Write to a document index - Pinecone embeds the text field automatically."""
    if not PINECONE_NAMESPACE:
        raise ValueError(
            "Document indexes require a non-empty namespace. Set PINECONE_NAMESPACE in .env."
        )
    text_field = target["text_field"]
    id_prefix = _slug(csv_path.stem)
    documents = [
        {
            "_id": f"{id_prefix}-{i}",
            text_field: chunk["text"],
            "source": csv_path.name,
            "row_index": chunk["row_index"],
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]
    print(
        f"Upserting {len(documents)} documents into '{PINECONE_INDEX}' "
        f"(namespace '{PINECONE_NAMESPACE}', field '{text_field}', "
        f"embedded by '{target.get('model')}')..."
    )
    index.documents.batch_upsert(
        namespace=PINECONE_NAMESPACE,
        documents=documents,
        batch_size=50,
        show_progress=False,
    )
    return len(documents)


def upload_vectors(pc: Pinecone, index, chunks: list[dict], csv_path: Path, target: dict) -> int:
    """Write to a classic vector index - we embed the chunks ourselves."""
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with '{PINECONE_EMBED_MODEL}'...")
    embeddings = embed_texts(pc, texts, PINECONE_EMBED_MODEL)

    if target.get("dimension") and embeddings:
        actual = len(embeddings[0])
        if actual != target["dimension"]:
            raise ValueError(
                f"Index '{PINECONE_INDEX}' has dimension {target['dimension']} but "
                f"model '{PINECONE_EMBED_MODEL}' produces {actual}. "
                "Change PINECONE_INDEX or PINECONE_EMBED_MODEL."
            )

    id_prefix = _slug(csv_path.stem)
    vectors = [
        {
            "id": f"{id_prefix}-{i}",
            "values": embeddings[i],
            "metadata": {
                "text": chunks[i]["text"],
                "source": csv_path.name,
                "row_index": chunks[i]["row_index"],
                "chunk_index": i,
            },
        }
        for i in range(len(chunks))
    ]
    print(f"Upserting {len(vectors)} vectors into '{PINECONE_INDEX}' (namespace '{PINECONE_NAMESPACE}')...")
    index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE, batch_size=100, show_progress=False)
    return len(vectors)


def embed_and_upload(csv_path: Path, chunks: list[dict], dry_run: bool = False) -> None:
    if dry_run:
        print("\n[dry run] Would embed and upload these chunks (no Pinecone writes):")
        for c in chunks:
            print(f"  row {c['row_index']}: {c['text'][:80]}...")
        return

    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc, PINECONE_INDEX)
    index = get_index(pc, PINECONE_INDEX)
    target = describe_target(pc, PINECONE_INDEX)
    print(f"Index '{PINECONE_INDEX}' detected as a '{target['mode']}' index.")

    if target["mode"] == "documents":
        count = upload_documents(index, chunks, csv_path, target)
    else:
        count = upload_vectors(pc, index, chunks, csv_path, target)

    print(f"Done - uploaded {count} chunks from '{csv_path.name}'.")


# ---------------------------------------------------------------- Main
def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv

    if not PINECONE_API_KEY and not dry_run:
        raise SystemExit("PINECONE_API_KEY is missing. Add it to your .env file.")

    csv_path = Path(args[0]) if args else Path(os.getenv("CSV_PATH") or DEFAULT_CSV_PATH)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    print(f"Reading CSV: {csv_path}")
    rows = extract_rows(csv_path)
    print(f"Extracted {len(rows)} row(s).")
    chunks = chunk_rows(rows)
    print(f"Extracted {len(chunks)} chunk(s).")
    if not chunks:
        raise SystemExit("No text could be extracted from the CSV.")

    embed_and_upload(csv_path, chunks, dry_run=dry_run)


if __name__ == "__main__":
    main()