
## Environment ----------------------------------------------------------------
import logging
import subprocess

import pymupdf4llm
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

import squarewheels

## Instances ------------------------------------------------------------------
### Logger --------------------------------------------------------------------
logger = logging.getLogger(__name__)

# logging.getLogger("squarewheels").setLevel(logging.INFO)
# logging.basicConfig(level=logging.INFO)

### Zotero --------------------------------------------------------------------
myZotero = squarewheels.Zotero_get()

### LLM/Embedding -------------------------------------------------------------
# lmstudio
subprocess.run("lms daemon up", check=True, shell=True, stdin=subprocess.DEVNULL)
subprocess.run("lms server start", check=True, shell=True, stdin=subprocess.DEVNULL)

# LLM client
client_LLM = OpenAI(
    base_url="http://localhost:1234/v1", 
    api_key="lm-studio"
    )

# Embedding
client_embedding = OpenAIEmbeddings(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model=squarewheels.load_embeddingmodel("text-embedding-embeddinggemma-300m-qat"),
    check_embedding_ctx_length=False # Set this to `False` to pass things from PyMuPDF4LLMLoader to OpenAIEmbeddings
    )

### Other utilities -----------------------------------------------------------
# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)



## Main -----------------------------------------------------------------------
### From local zotero run -----------------------------------------------------
# Zotero to Shelf
shelf = squarewheels.Zotero_makelist(myZotero, limit = 2)

# Shelf to PDF
paper_PDF = squarewheels.Zotero_pullPDF(myZotero, shelf[1])

# PDF to MD
paper_md = pymupdf4llm.to_markdown(paper_PDF)

# MD to abstract
squarewheels.PDF_to_MMICKURR(client=client_LLM, model=None, paper_md=paper_md)


### From PubMed ---------------------------------------------------------------
cands = squarewheels.PubMed_search_PMID(term="rheumatoid arthritis GWAS", retmax=20)
cands_abst = squarewheels.PubMed_fetch_abstract(pmids=cands)
