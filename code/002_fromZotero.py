
## Environment ----------------------------------------------------------------
import logging

import pymupdf4llm
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

import squarewheels

## Instances ------------------------------------------------------------------
### Logger --------------------------------------------------------------------
logger = logging.getLogger(__name__)

### Config ---------------------------------------------------------------------
config = squarewheels.config_load()

### Zotero --------------------------------------------------------------------
myZotero = squarewheels.Zotero_get()

### LLM/Embedding -------------------------------------------------------------
# lmstudio
squarewheels.lms_start()

# LLM client
client_LLM = OpenAI(
    base_url=config["lmstudio"]["base_url"], 
    api_key=config["lmstudio"]["api_key"]
    )

# Embedding
client_embedding = OpenAIEmbeddings(
    base_url=config["lmstudio"]["base_url"],
    api_key=config["lmstudio"]["api_key"],
    model=squarewheels.lms_load_embeddingmodel(config["lmstudio"]["embedding"]),
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
paper_PDF = squarewheels.Zotero_pullPDF(myZotero, shelf[1], path=config["paths"]["Zotero_temp"])

# PDF to MD
paper_md = pymupdf4llm.to_markdown(paper_PDF)

# MD to abstract
squarewheels.PDF_to_MMICKURR(client=client_LLM, model=None, paper_md=paper_md)

