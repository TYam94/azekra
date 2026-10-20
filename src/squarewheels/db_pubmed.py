
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
# NCBI asks for an email address for E-utilities (works without it, but recommended). api_key is optional (raises the rate limit).
def PubMed_set_Entrez(email, api_key=None):

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
# exc_Components: Excludes "Publication Components" (comments, editorials, errata, ...) except letters.
#   Letter is itself a Publication Component, so it is put back explicitly;
#   letters that are comments on other articles stay excluded.
# sort: Passed to ESearch (e.g. "relevance" for Best Match). None uses PubMed's default.
# return_count: If True, returns (PMIDs, total number of hits). The total can exceed retmax.
def PubMed_search_PMID(query, retmax=10000, lang="En", exc_Review=True, exc_Components=True, sort=None, return_count=False):

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
# Just call searching function in turn.
# When flatten is False, each item is {'query', 'result' (PMIDs in the order of sort), 'count' (total hits)}.
def PubMed_search_PMID_array(array_query, flatten=True, retmax=10000, lang="En", exc_Review=True, exc_Components=True, sort=None):

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


### Fetch abstract based on PMID --------------------------
def PubMed_fetch_abstracts(pmids, file_PubMed_abstracts, batch_size=1000):

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

                pmids_fetch_absts.append({"PMID": pmid, "publication_date": date_str, "abstract": _extract_abstract(article)})

        # Add them to the results and store
        results.extend(pmids_fetch_absts)
        PubMed_stored_absts.extend(pmids_fetch_absts)

    # Save the store automatically
    if len(PubMed_stored_absts) > 0:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(PubMed_stored_absts, f, indent=4, ensure_ascii=False)

    # Return the result
    return results


def PubMed_save_abstracts(litres_absts, file_PubMed_abstracts):

    file_path = Path(file_PubMed_abstracts)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(litres_absts, f, indent=4, ensure_ascii=False)

    return file_path