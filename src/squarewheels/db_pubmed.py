
# Data-Base integration: Pubmed ===============================================

## Dependency -----------------------------------------------------------------
import json
import logging
import re
import time
from pathlib import Path

from Bio import Entrez

## Logging --------------------------------------------------------------------
logger = logging.getLogger(__name__)


## Setting --------------------------------------------------------------------
### Set Entrez identity -----------------------------------
def PubMed_set_Entrez(email, api_key=None):
    """Set Entrez identity for NCBI E-utilities.

    Args:
        email: Email address. NCBI asks for it (works without it, but recommended); a warning is logged if not given.
        api_key: NCBI API key (optional). Raises the rate limit.

    Returns:
        None
    """
    if email:
        Entrez.email = email
    else:
        logger.warning("Email for NCBI E-utilities is not given. Setting it in config.toml [PubMed] is recommended.")

    if api_key:
        Entrez.api_key = api_key


## Searching ------------------------------------------------------------------
### Run ESearch, retrying transient backend errors --------
# Biopython spaces requests and retries HTTP errors by itself, but NCBI may report a temporary
# backend failure inside a successful (HTTP 200) response, which Entrez.read raises as RuntimeError.
# Only such transient errors are retried here; others are raised immediately.
_ESEARCH_TRANSIENT = ("temporarily unavailable", "Search Backend failed")

def _esearch(esearch_args, max_tries=3, sleep_between_tries=5):

    for attempt in range(1, max_tries + 1):
        try:
            handle = Entrez.esearch(**esearch_args)
            result = Entrez.read(handle)
            handle.close()
            return result
        except RuntimeError as e:
            if attempt == max_tries or not any(s in str(e) for s in _ESEARCH_TRANSIENT):
                raise
            logger.warning(f"PubMed ESearch failed temporarily (attempt {attempt}/{max_tries}); retrying in {sleep_between_tries} s: {e}")
            time.sleep(sleep_between_tries)


### Basic search ------------------------------------------
def PubMed_search_PMID(query, retmax=10000, lang="En", exc_Review=True, exc_Components=True, sort=None, return_count=False):
    """Search PubMed and get PMIDs.

    Args:
        query: PubMed query (terms are combined with AND by PubMed).
        retmax: Maximum number of PMIDs returned.
        lang: "En" for English only, "EnJp" for English or Japanese. Any other value applies no language filter.
        exc_Review: If True, excludes reviews.
        exc_Components: If True, excludes "Publication Components" (comments, editorials, errata, ...) except letters.
            Letter is itself a Publication Component, so it is put back explicitly;
            letters that are comments on other articles stay excluded.
        sort: Passed to ESearch (e.g. "relevance" for Best Match). None uses PubMed's default.
        return_count: If True, the total number of hits is returned too.

    Returns:
        PMIDs (list of str, in the order of sort), or (PMIDs, total number of hits) if return_count is True.
        The total can exceed retmax.
    """
    # Parenthesized so that OR in the query is not mixed with the filters below
    term = f"({query})"

    if lang == "En":
        term = f"{term} AND (english[LA])"
    elif lang == "EnJp":
        term = f"{term} AND (english[LA] OR japanese[LA])"
        
    if exc_Review == True:
        term = f"{term} NOT Review[PT]"

    if exc_Components == True:
        term = f'{term} NOT ("Publication Components"[PT] NOT (Letter[PT] NOT Comment[PT]))'

    esearch_args = {"db": "pubmed", "term": term, "retmax": retmax}
    if sort is not None:
        esearch_args["sort"] = sort

    result = _esearch(esearch_args)

    if return_count:
        return result["IdList"], int(result["Count"])
    return result["IdList"]


### Search in array ---------------------------------------
def PubMed_search_PMID_array(array_query, flatten=True, retmax=10000, lang="En", exc_Review=True, exc_Components=True, sort=None):
    """Search PubMed with each query in turn (see PubMed_search_PMID).

    Args:
        array_query: Queries to search.
        flatten: If True, returns PMIDs of all queries together (de-duplicated, in no particular order).
        retmax, lang, exc_Review, exc_Components, sort: Passed to PubMed_search_PMID (retmax applies per query).

    Returns:
        If flatten is True, list of PMIDs.
        If False, list of {'query', 'result' (PMIDs in the order of sort), 'count' (total hits)}, one per query.
    """
    PubMed_search_results=[]
    PubMed_search_results_PMIDs=[]

    for query in array_query:
        search_result, search_count = PubMed_search_PMID(query=query, retmax=retmax, lang=lang, exc_Review=exc_Review, exc_Components=exc_Components, sort=sort, return_count=True)
        PubMed_search_results.append({'query': query, 'result': search_result, 'count': search_count})
        PubMed_search_results_PMIDs.extend(search_result)

    if flatten == True:
        return list(set(PubMed_search_results_PMIDs))
    else:
        return PubMed_search_results



## Fetching abstracts ---------------------------------------------------------
### Make abstract text clean ------------------------------
def _clean_abstract_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    text = re.sub(r"\n+", "\n", text)

    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    text = re.sub(r"[ \u3000]+", " ", text)

    return text.strip()


### Extract abtstract from fetched XML --------------------
def _extract_abstract(article):
    
    abstract = article["MedlineCitation"]["Article"].get("Abstract")
    
    if abstract is None:
        return None
    
    parts = []

    for section in abstract["AbstractText"]:
        label = section.attributes.get("Label")
        text = str(section)
        parts.append(f"{label}: {text}" if label else text)

    return _clean_abstract_text("\n".join(parts))


### Fields from publication types and DOI ----------------
# Retracted articles have "Retracted Publication" among their publication types.
# doi_prefix is the registrant part of DOI, which identifies the publisher (e.g. "10.1234" in "10.1234/abc.567").
# Articles without DOI get None for both.
def _summary_fields(publication_types, doi):

    doi = str(doi).strip() if doi else None

    return {
        "publication_types": publication_types,
        "retracted": "Retracted Publication" in publication_types,
        "doi": doi,
        "doi_prefix": doi.split("/", 1)[0].lower() if doi else None,
    }


### Extract publication types and DOI from fetched XML ----
def _extract_summary_fields(article):

    publication_types = [str(pt) for pt in article["MedlineCitation"]["Article"].get("PublicationTypeList", [])]
    doi = next((x for x in article.get("PubmedData", {}).get("ArticleIdList", []) if x.attributes.get("IdType") == "doi"), None)

    return _summary_fields(publication_types, doi)


### Refresh publication types and DOI of stored entries --
# Uses ESummary (lighter than EFetch) to tag entries stored without these fields
# and to catch articles retracted after they were stored. Updates the entries in place.
# Entries not returned by ESummary (e.g. deleted from PubMed) are left as they are.
def _refresh_summary_fields(PubMed_stored_absts, batch_size=500):

    pmids = [item["PMID"] for item in PubMed_stored_absts]
    fields_all = {}

    for i in range(0, len(pmids), batch_size):
        handle = Entrez.esummary(db="pubmed", id=",".join(pmids[i:i + batch_size]))
        summaries = Entrez.read(handle)
        handle.close()

        for summary in summaries:
            if "PubTypeList" in summary:
                fields_all[str(summary["Id"])] = _summary_fields(
                    [str(pt) for pt in summary["PubTypeList"]],
                    summary.get("ArticleIds", {}).get("doi"),
                )

    n_updated = 0
    pmids_retracted_new = []
    for item in PubMed_stored_absts:
        fields = fields_all.get(item["PMID"])
        if fields is None or all(item.get(key) == value for key, value in fields.items()):
            continue
        if fields["retracted"] and not item.get("retracted", False):
            pmids_retracted_new.append(item["PMID"])
        item.update(fields)
        n_updated += 1

    logger.info(f"PubMed publication types/DOI refreshed: {n_updated} / {len(pmids)} stored entries updated")
    if pmids_retracted_new:
        logger.warning(f"PubMed: newly marked as retracted: {', '.join(pmids_retracted_new)}")


### Fetch abstract based on PMID --------------------------
def PubMed_fetch_abstracts(pmids, file_PubMed_abstracts, batch_size=1000, refresh_stored=True):
    """Get abstracts by PMID, from the local store or from PubMed (EFetch).
    Ones not stored yet are fetched and added to the store, which is saved automatically.

    Each entry is {'PMID', 'publication_date' ("YYYY-MM-DD"), 'publication_types', 'retracted',
    'doi', 'doi_prefix' (registrant part of DOI, e.g. "10.1234"; None for both without DOI), 'abstract' (None without abstract)}.

    Args:
        pmids: PMIDs to get.
        file_PubMed_abstracts: JSON file of stored abstracts. Created if not exists.
        batch_size: Number of PMIDs fetched at once.
        refresh_stored: If True, publication types (and so retraction status) and DOI of all stored entries
            are refreshed on load via ESummary, which also tags entries stored before these fields were recorded.

    Returns:
        Entries of given PMIDs: stored ones first, then fetched ones.
        PMIDs not returned by PubMed (e.g. not existing, or book chapters) are left out.
    """
    # Containers
    results = []
    pmids_fetch = []
    pmids_fetch_absts = []
    
    # Load stored abstracts from PubMed (if exist)
    file_path = Path(file_PubMed_abstracts)
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            PubMed_stored_absts = json.load(f)
    else:
        PubMed_stored_absts = []

    if refresh_stored and len(PubMed_stored_absts) > 0:
        _refresh_summary_fields(PubMed_stored_absts)

    # Pick up stored ones (index by PMID for O(1) lookup)
    PubMed_stored_absts_index = {item.get("PMID"): item for item in PubMed_stored_absts}
    for pmid in pmids:
        if pmid in PubMed_stored_absts_index:
            results.append(PubMed_stored_absts_index[pmid])
        else:
            pmids_fetch.append(pmid)

    # Run Entrez.efetch for non-found literatures if such cases exist
    if len(pmids_fetch) > 0:
        # Batch run
        for i in range(0, len(pmids_fetch), batch_size):

            batch = pmids_fetch[i:i + batch_size]

            handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml")
            records = Entrez.read(handle)
            handle.close()

            for article in records["PubmedArticle"]:

                # PMID
                pmid = str(article["MedlineCitation"]["PMID"])

                # Publication date
                date_str = None
                # 1. ArticleDate
                art_dates = article["MedlineCitation"]["Article"].get("ArticleDate", [])
                if art_dates:
                    d = art_dates[0]
                    date_str = f"{d['Year']}-{d['Month'].zfill(2)}-{d['Day'].zfill(2)}"
                # 2. History (PubMed/Entrez Date)
                if not date_str:
                    history = article.get("PubmedData", {}).get("History", [])
                    for h in history:
                        if h.attributes.get("PubStatus") in ["pubmed", "entrez"]:
                            date_str = (f"{h['Year']}-{h['Month'].zfill(2)}-{h['Day'].zfill(2)}")
                            break
                # 3. JournalIssue PubDate (fallback)
                if not date_str:
                    pub_date = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]
                    year = pub_date.get("Year", "Unknown")
                    month = pub_date.get("Month", "00").zfill(2)
                    day = pub_date.get("Day", "00").zfill(2)
                    date_str = f"{year}-{month}-{day}"

                pmids_fetch_absts.append({
                    "PMID": pmid,
                    "publication_date": date_str,
                    **_extract_summary_fields(article),
                    "abstract": _extract_abstract(article),
                    })

        # Add them to the results and store
        results.extend(pmids_fetch_absts)
        for item in pmids_fetch_absts:
            PubMed_stored_absts_index[item["PMID"]] = item
        PubMed_stored_absts = list(PubMed_stored_absts_index.values())

    # Save the store automatically
    if len(PubMed_stored_absts) > 0:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(PubMed_stored_absts, f, indent=4, ensure_ascii=False)

    # Return the result
    return results


### Save abstracts ----------------------------------------
def PubMed_save_abstracts(litres_absts, file_PubMed_abstracts):
    """Save abstracts to a JSON file (overwritten if exists).

    Args:
        litres_absts: Entries of abstracts (see PubMed_fetch_abstracts).
        file_PubMed_abstracts: JSON file to save. Parent directories are created if not exist.

    Returns:
        file_path: Path of the saved file.
    """
    file_path = Path(file_PubMed_abstracts)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(litres_absts, f, indent=4, ensure_ascii=False)

    return file_path