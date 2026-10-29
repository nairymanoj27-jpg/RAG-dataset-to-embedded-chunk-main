"""
Search the chunks that embed_pdf.py uploaded.

Works with both index types:
  * document index with integrated embedding -> index.documents.search(...)
  * classic vector index                     -> pc.inference.embed + index.query(...)

Usage:
    python query_pdf.py "how does the system detect an accident?"
    python query_pdf.py "buzzer timeout" -k 3


Reads the same .env file as embed_pdf.py.
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.models.documents.score_by import EmbedQuery

load_dotenv()

PINECONE_API_KEY = (os.getenv("PINECONE_API_KEY") or "").strip()
PINECONE_HOST = (os.getenv("PINECONE_HOST") or "").strip()
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "donbosco").strip()
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default").strip()
PINECONE_EMBED_MODEL = os.getenv("PINECONE_EMBED_MODEL", "multilingual-e5-large").strip()

_PLACEHOLDER_HOST = "your-pinecone-host"


def find_text_field(desc) -> str | None:
    """Return the name of the integrated-embedding text field, or None if classic."""
    fields = getattr(getattr(desc, "schema", None), "fields", None) or {}
    for field_name, field in fields.items():
        if getattr(field, "embed", None) is not None:
            return field_name
    return None


def search_documents(index, question: str, top_k: int, text_field: str) -> list[dict]:
    response = index.documents.search(
        namespace=PINECONE_NAMESPACE,
        score_by=[EmbedQuery(field=text_field, query=question)],
        top_k=top_k,
        include_fields=[text_field, "source", "page"],
    )
    return [
        {
            "score": match.score,
            "text": match.get(text_field) or "",
            "source": match.get("source"),
            "page": match.get("page"),
        }
        for match in response.matches
    ]


def search_vectors(pc, index, question: str, top_k: int) -> list[dict]:
    query_vector = pc.inference.embed(
        model=PINECONE_EMBED_MODEL,
        inputs=question,
        parameters={"input_type": "query"},
    ).data[0].values

    response = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True,
    )
    return [
        {
            "score": match.score,
            "text": (match.metadata or {}).get("text", ""),
            "source": (match.metadata or {}).get("source"),
            "page": (match.metadata or {}).get("page"),
        }
        for match in response.matches
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a Pinecone index built from a PDF.")
    parser.add_argument("question", help="the question to search for")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="number of results (default: 5)")
    args = parser.parse_args()

    if not PINECONE_API_KEY:
        raise SystemExit("PINECONE_API_KEY is missing. Add it to your .env file.")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Same pattern as the quickstart: pc.Index("<name>"). A host in .env is optional.
    if PINECONE_HOST and _PLACEHOLDER_HOST not in PINECONE_HOST:
        index = pc.Index(PINECONE_INDEX, host=PINECONE_HOST)
    else:
        index = pc.Index(PINECONE_INDEX)

    text_field = find_text_field(pc.describe_index(PINECONE_INDEX))
    if text_field:
        results = search_documents(index, args.question, args.top_k, text_field)
    else:
        results = search_vectors(pc, index, args.question, args.top_k)

    print(f"\nTop {len(results)} matches for: {args.question!r}\n")
    for rank, r in enumerate(results, start=1):
        print(f"[{rank}] score={r['score']:.4f}  source={r['source']}  page={r['page']}")
        print(f"    {r['text'][:300]}\n")


if __name__ == "__main__":
    main()