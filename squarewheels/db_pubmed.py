
import re

from Bio import Entrez


## Basic utilities ------------------------------------------------------------
def PubMed_search_PMID(term, retmax=200):
    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax)
    result = Entrez.read(handle)
    handle.close()
    return result["IdList"]

def _extract_abstract(article):
    abstract = article["MedlineCitation"]["Article"].get("Abstract")
    if abstract is None:
        return None
    parts = []
    for section in abstract["AbstractText"]:
        label = section.attributes.get("Label")
        text = str(section)
        parts.append(f"{label}: {text}" if label else text)    
    return clean_abstract_text("\n".join(parts))


def PubMed_fetch_abstract(pmids, batch_size=100):
    results = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml")
        records = Entrez.read(handle)
        handle.close()
        for article in records["PubmedArticle"]:
            pmid = str(article["MedlineCitation"]["PMID"])
            results.append({"PMID": pmid, "abstract": _extract_abstract(article)})
    return results


## Quick wrappers -------------------------------------------------------------
### Search in PubMed with an array of query -----------------------------------
# Just call searching function in turn.
def PubMed_search_PMID_array(array_query, retmax=200):
    PubMed_search_results=[]

    for query in array_query:
        search_result = PubMed_search_PMID(term=query, retmax=retmax)
        PubMed_search_results.append({'query': query, 'result': search_result})

    return PubMed_search_results





def clean_abstract_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    text = re.sub(r"\n+", "\n", text)

    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    text = re.sub(r"[ \u3000]+", " ", text)

    return text.strip()

