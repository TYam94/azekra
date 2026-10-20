
## Input (optional) -----------------------------------------------------------
# Leave None to use the input file. The input file is chosen in this order:
#   .toml given as command-line argument > input_file below > [paths] litres_input in config.toml
# research_context / request written here override the ones in the input file.
input_file = None         # e.g. "data/user/another_topic.toml" (relative to the current directory)
research_context = None   # e.g. "I aim to elucidate ..."
request = None            # e.g. "Please write introduction section ..."


## Environment ----------------------------------------------------------------
### Dependency ----------------------------------------------------------------
import logging
import sys
import tomllib
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

import squarewheels

### Logger --------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

### Config --------------------------------------------------------------------
config = squarewheels.config_load()
squarewheels.PubMed_set_Entrez(**config["PubMed"])

### Input ---------------------------------------------------------------------
# The input file is read only when research_context or request is not written at the top.
# Arguments not ending with .toml are ignored (e.g. ones added by IPython kernels).
file_input = None

if research_context is None or request is None:
    file_input = next(
        (arg for arg in sys.argv[1:] if arg.endswith(".toml")),
        input_file or config["paths"]["litres_input"],
    )

    with open(file_input, "rb") as f:
        litres_input = tomllib.load(f)

    if research_context is None: research_context = litres_input["research_context"]
    if request is None: request = litres_input["request"]

research_context = research_context.strip()
request = request.strip()

### LLM/Embedding -------------------------------------------------------------
# lmstudio
squarewheels.lms_start()

# LLM client
LLM_client = OpenAI(
    base_url=config["lmstudio"]["base_url"], 
    api_key=config["lmstudio"]["api_key"],
)

LLM_model = squarewheels.lms_load_LLM(config["lmstudio"]["llm"])

# Embedding
embedding_client = OpenAIEmbeddings(
    base_url=config["lmstudio"]["base_url"],
    api_key=config["lmstudio"]["api_key"],
    model=squarewheels.lms_load_embeddingmodel(config["lmstudio"]["embedding"]),
    check_embedding_ctx_length=False, # Set this to `False` to pass things from PyMuPDF4LLMLoader to OpenAIEmbeddings
)


## Main -----------------------------------------------------------------------
def main():
    
    # Search PubMed using LLM
    global search_thread
    search_thread = squarewheels.run_litres_search_PubMed(
        client=LLM_client,
        model=LLM_model,
        research_context=research_context,
    )

    squarewheels.thread_save(
        search_thread,
        path=config["paths"]["threads"] / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}_litres_search_PubMed.json",
        meta={"workflow": "run_litres_search_PubMed", "input": None if file_input is None else str(file_input), "model": LLM_model, "research_context": research_context},
    )

    # Searched PMIDs are in the last post
    PubMed_PMIDs = [pmid for pmid in search_thread[-1]['message'].split(", ") if pmid]

    # Gather abstracts from locally-stored ones and PubMed
    squarewheels.PubMed_fetch_abstracts(
        pmids=PubMed_PMIDs, 
        file_PubMed_abstracts=config["paths"]["PubMed_abstracts"],
    )

    # Open vector-store and add stored abstracts not yet in it
    PubMed_vectorstore = squarewheels.Chroma_get(
        embedding=embedding_client,
        collection_name=f"PubMed_Chroma_{embedding_client.model}",
        path=config["paths"]["PubMed_chroma"],
    )

    squarewheels.Chroma_sync_abstracts_PubMed(
        PubMed_vectorstore,
        file_PubMed_abstracts=config["paths"]["PubMed_abstracts"],
    )

    # Literature-based answer from LLM
    global litres_thread
    litres_thread = squarewheels.run_litres_vectorstore(
        client=LLM_client, 
        model=LLM_model, 
        research_context=research_context, 
        request=request,
        vectorstore=PubMed_vectorstore
    )

    squarewheels.thread_save(
        litres_thread,
        path=config["paths"]["threads"] / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}_litres_vectorstore.json",
        meta={"workflow": "run_litres_vectorstore", "input": None if file_input is None else str(file_input), "model": LLM_model, "embedding": embedding_client.model, "research_context": research_context, "request": request},
    )


## Entry point ----------------------------------------------------------------
if __name__ == "__main__":
    main()

