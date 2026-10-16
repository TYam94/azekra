
# Data-Base integration: Pubmed ===============================================

## Dependency -----------------------------------------------------------------
import logging
import re

from Bio import Entrez

## Logging --------------------------------------------------------------------
logger = logging.getLogger(__name__)


## Searching ------------------------------------------------------------------
### Basic search ------------------------------------------
def PubMed_search_PMID(query, retmax=10000, lang="En", exc_Review=True):

    term = query

    if lang == "En":
        term = f"{term} AND (english[LA])"
    elif lang == "EnJp":
        term = f"{term} AND (english[LA]) OR japanese[LA]"
        
    if exc_Review == True:
        term = f"{term} NOT Review[PT]"

    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax)
    result = Entrez.read(handle)
    handle.close()

    return result["IdList"]


### Search in array ---------------------------------------
# Just call searching function in turn.
def PubMed_search_PMID_array(array_query, flatten=True, retmax=10000, lang="En", exc_Review=True):

    PubMed_search_results=[]
    PubMed_search_results_PMIDs=[]

    for query in array_query:
        search_result = PubMed_search_PMID(query=query, retmax=retmax, lang=lang, exc_Review=exc_Review)
        PubMed_search_results.append({'query': query, 'result': search_result})
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
def PubMed_fetch_abstract(pmids, batch_size=1000):

    results = []

    for i in range(0, len(pmids), batch_size):

        batch = pmids[i:i + batch_size]

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

            results.append({"PMID": pmid, "publication_date": date_str, "abstract": _extract_abstract(article)})

    return results
