# Vector-store: Chroma ========================================================

## Dependency -----------------------------------------------------------------
import json
import logging
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

## Logging --------------------------------------------------------------------
logger = logging.getLogger(__name__)


## Opening --------------------------------------------------------------------
### Open (or create) persistent vector-store --------------
# Vectors depend on the embedding model; use separate collection per model (e.g. include model name in collection_name).
# Characters not allowed in Chroma collection name are replaced with "_", and non-alphanumeric ends are trimmed.
def Chroma_get(embedding, collection_name, path):

    collection_name = re.sub(r"[^a-zA-Z0-9._-]", "_", collection_name).strip("._-")

    Path(path).mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=str(path),
        )

    return vectorstore


## Syncing --------------------------------------------------------------------
### Add stored PubMed abstracts not yet in vector-store ---
# "PMID:<PMID>" is used as the id of each entry.
# Returns the number of added entries.
def Chroma_sync_abstracts_PubMed(vectorstore, file_PubMed_abstracts, batch_size=1000):

    # Load stored abstracts from PubMed
    file_path = Path(file_PubMed_abstracts)
    if not file_path.exists():
        raise FileNotFoundError(f"Stored PubMed abstracts not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        PubMed_stored_absts = json.load(f)

    # Pick up ones not in vector-store (de-duplicated by PMID)
    ids_stored = set(vectorstore.get(include=[])["ids"])
    absts_new = {f"PMID:{item['PMID']}": item for item in PubMed_stored_absts if f"PMID:{item['PMID']}" not in ids_stored}

    # Abstracts into documents
    docs_new = [
        Document(
            page_content=f"[PMID: {item['PMID']}]\n[publication_date: {item['publication_date']}]\n{item['abstract']}",
            metadata={"PMID": item['PMID'], "publication_date": item['publication_date']}
            )
        for item in absts_new.values()
    ]
    ids_new = list(absts_new.keys())

    # Batch add
    for i in range(0, len(docs_new), batch_size):
        batch = docs_new[i:i + batch_size]
        vectorstore.add_documents(batch, ids=ids_new[i:i + batch_size])
        logger.info(f"Chroma_sync_abstracts_PubMed: {i + len(batch)} / {len(docs_new)} added")

    return len(docs_new)
