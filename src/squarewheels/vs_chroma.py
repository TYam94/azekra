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
def Chroma_get(embedding, collection_name, path):
    """Open (or create) persistent vector-store.

    Args:
        embedding: Embedding model (e.g. OpenAIEmbeddings).
            Vectors depend on the embedding model; use separate collection per model (e.g. include model name in collection_name).
        collection_name: Name of the collection. Characters not allowed in Chroma collection name are replaced with "_",
            and non-alphanumeric ends are trimmed.
        path: Directory of the vector-store. Created if not exists.

    Returns:
        vectorstore: Chroma vector-store.
    """
    collection_name = re.sub(r"[^a-zA-Z0-9._-]", "_", collection_name).strip("._-")

    Path(path).mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=str(path),
        )

    return vectorstore


## Syncing --------------------------------------------------------------------
### Metadata of a stored PubMed abstract ------------------
# Publication types are joined with "; " (Chroma metadata values are scalars).
# Entries stored before publication types were recorded get "" and False.
# doi and doi_prefix are left out for entries without DOI (Chroma metadata values cannot be None).
def _metadata_PubMed(item):

    metadata = {
        "PMID": item["PMID"],
        "publication_date": item["publication_date"],
        "publication_types": "; ".join(item.get("publication_types", [])),
        "retracted": item.get("retracted", False),
    }
    if item.get("doi"):
        metadata["doi"] = item["doi"]
        metadata["doi_prefix"] = item["doi_prefix"]

    return metadata


### Add stored PubMed abstracts not yet in vector-store ---
def Chroma_sync_abstracts_PubMed(vectorstore, file_PubMed_abstracts, batch_size=1000):
    """Add stored PubMed abstracts not yet in vector-store.
    "PMID:<PMID>" is used as the id of each entry.
    Entries already in vector-store get their metadata updated if it changed (no re-embedding).

    Args:
        vectorstore: Chroma vector-store (see Chroma_get).
        file_PubMed_abstracts: JSON file of stored abstracts (see PubMed_fetch_abstracts).
        batch_size: Number of entries added/updated at once.

    Returns:
        Number of added entries.

    Raises:
        FileNotFoundError: If file_PubMed_abstracts does not exist.
    """
    # Load stored abstracts from PubMed
    file_path = Path(file_PubMed_abstracts)
    if not file_path.exists():
        raise FileNotFoundError(f"Stored PubMed abstracts not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        PubMed_stored_absts = json.load(f)

    # Split into ones not in vector-store and ones already there (de-duplicated by PMID)
    stored = vectorstore.get(include=["metadatas"])
    metadatas_stored = dict(zip(stored["ids"], stored["metadatas"]))
    absts_all = {f"PMID:{item['PMID']}": item for item in PubMed_stored_absts}
    absts_new = {id_: item for id_, item in absts_all.items() if id_ not in metadatas_stored}

    # Update metadata of existing ones
    metadatas_update = {
        id_: _metadata_PubMed(item)
        for id_, item in absts_all.items()
        if id_ in metadatas_stored and metadatas_stored[id_] != _metadata_PubMed(item)
    }
    ids_update = list(metadatas_update.keys())
    for i in range(0, len(ids_update), batch_size):
        batch = ids_update[i:i + batch_size]
        vectorstore._collection.update(ids=batch, metadatas=[metadatas_update[id_] for id_ in batch])
        logger.info(f"Chroma_sync_abstracts_PubMed: {i + len(batch)} / {len(ids_update)} metadata updated")

    # Abstracts into documents
    docs_new = [
        Document(
            page_content=f"[PMID: {item['PMID']}]\n[publication_date: {item['publication_date']}]\n{item['abstract']}",
            metadata=_metadata_PubMed(item)
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


## Getting --------------------------------------------------------------------
### Get stored PubMed abstracts by PMID -------------------
def Chroma_get_abstracts_PubMed(vectorstore, pmids, output="docs", level_offset=0):
    """Get stored PubMed abstracts by PMID (not by similarity).
    Missing and retracted ones are skipped with a warning.

    Args:
        vectorstore: Chroma vector-store (see Chroma_get).
        pmids: PMIDs to get.
        output: "docs" for a list of Document (same as search results), or "md" for Markdown (see abstracts_docs_to_md_PubMed).
        level_offset: Passed to abstracts_docs_to_md_PubMed. Used only when output is "md".

    Returns:
        Abstracts in the order of pmids, as a list of Document or as a Markdown string ([] or "" for none).

    Raises:
        ValueError: If output is neither "docs" nor "md".
    """
    if output not in ("docs", "md"):
        raise ValueError(f"output must be 'docs' or 'md', got {output!r}")

    docs = []

    # Chroma does not accept an empty list for $in
    if pmids:
        stored = vectorstore.get(where={"PMID": {"$in": list(pmids)}}, include=["metadatas", "documents"])
        pmids_found = {metadata["PMID"]: Document(page_content=document, metadata=metadata) for metadata, document in zip(stored["metadatas"], stored["documents"])}

        for pmid in pmids:
            if pmid not in pmids_found:
                logger.warning(f"Chroma_get_abstracts_PubMed: PMID {pmid} is not in the vector-store; skipped")
                continue
            if pmids_found[pmid].metadata.get("retracted"):
                logger.warning(f"Chroma_get_abstracts_PubMed: PMID {pmid} is retracted; skipped")
                continue
            docs.append(pmids_found[pmid])

    if output == "md":
        return abstracts_docs_to_md_PubMed(docs, level_offset=level_offset)
    return docs


## Markdown -------------------------------------------------------------------
### PubMed abstracts (documents) as Markdown --------------
def abstracts_docs_to_md_PubMed(docs, level_offset=0):
    """PubMed abstracts (documents) as Markdown.
    Used both for passing literature to LLM and for writing it out.
    Each entry is "# PMID: ..." with its publication date and types as sub-headings, followed by the abstract.

    Args:
        docs: List of Document, with 'PMID' and 'publication_date' in metadata ('publication_types' is optional).
        level_offset: Number of "#" added to each heading (e.g. 2 to put them under "## ...").

    Returns:
        Markdown string. Each entry starts with a blank line; "" for no documents.
    """
    h ="#" * level_offset

    return "".join(
        f"\n"
        f"#{h} PMID: {doc.metadata["PMID"]} \n"
        f"##{h} Publication_date: {doc.metadata["publication_date"]} \n"
        f"##{h} Publication_types: {doc.metadata.get("publication_types") or "Unknown"} \n"
        f"{doc.page_content}\n"
        for doc in docs
    )
