
from Bio import Entrez


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
    return "\n".join(parts)

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
