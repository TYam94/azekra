
## Input (optional) -----------------------------------------------------------
# Leave None to use the input file. The input file is chosen in this order:
#   .toml given as command-line argument > input_file below > [paths] litres_input in config.toml
# research_context / request / exclude_doi_prefixes / must_cite written here override the ones in the input file.
input_file = None         # e.g. "data/user/another_topic.toml" (relative to the current directory)
research_context = None   # e.g. "I aim to elucidate ..."
request = None            # e.g. "Please write introduction section ..."
exclude_doi_prefixes = None   # e.g. ["10.xxxx", "10.yyyy"]; publishers excluded from literature search, by DOI prefix
must_cite = None          # e.g. ["12345678", "23456789"]; PMIDs the answer must cite (fetched from PubMed if not stored)


## Environment ----------------------------------------------------------------
### Dependency ----------------------------------------------------------------
import logging
import sys
import tomllib
from datetime import datetime
from pathlib import Path

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
    if exclude_doi_prefixes is None: exclude_doi_prefixes = litres_input.get("exclude_doi_prefixes")
    if must_cite is None: must_cite = litres_input.get("must_cite")

research_context = research_context.strip()
request = request.strip()
must_cite = [str(pmid).strip() for pmid in (must_cite or [])] # PMIDs may be written as numbers

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
    
    # # Search PubMed using LLM
    # global search_thread
    # search_thread = squarewheels.run_litres_search_PubMed(
    #     client=LLM_client,
    #     model=LLM_model,
    #     research_context=research_context,
    # )

    # squarewheels.thread_save(
    #     search_thread,
    #     path=config["paths"]["threads"] / f"{datetime.now().astimezone():%Y%m%d_%H%M%S}_litres_search_PubMed.json",
    #     meta={"workflow": "run_litres_search_PubMed", "input": None if file_input is None else str(file_input), "model": LLM_model, "research_context": research_context},
    # )

    # # Searched PMIDs are in the last post
    # PubMed_PMIDs = [pmid for pmid in search_thread[-1]['message'].split(", ") if pmid]

    # # Gather abstracts from locally-stored ones and PubMed (must-cite ones too)
    # squarewheels.PubMed_fetch_abstracts(
    #     pmids=list(dict.fromkeys(PubMed_PMIDs + must_cite)),
    #     file_PubMed_abstracts=config["paths"]["PubMed_abstracts"],
    # )

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
    global litres_thread, pmids_cited
    litres_thread, pmids_cited = squarewheels.run_litres_vectorstore(
        client=LLM_client,
        model=LLM_model,
        research_context=research_context,
        request=request,
        vectorstore=PubMed_vectorstore,
        exclude_doi_prefixes=exclude_doi_prefixes,
        must_cite=must_cite,
    )

    squarewheels.thread_save(
        litres_thread,
        path=config["paths"]["threads"] / f"{datetime.now().astimezone():%Y%m%d_%H%M%S}_litres_vectorstore.json",
        meta={"workflow": "run_litres_vectorstore", "input": None if file_input is None else str(file_input), "model": LLM_model, "embedding": embedding_client.model, "research_context": research_context, "request": request, "exclude_doi_prefixes": exclude_doi_prefixes, "litres_pmids": litres_pmids},
    )

    # Save the final answer as Markdown next to the input file (<input>_litres_<datetime>.md)
    # Without an input file, it goes to the folder of the default input (litres_<datetime>.md).
    now = datetime.now().astimezone()
    if file_input is None:
        path_result = Path(config["paths"]["litres_input"]).parent / f"litres_{now:%Y%m%d_%H%M%S}.md"
    else:
        path_result = Path(file_input).with_name(f"{Path(file_input).stem}_litres_{now:%Y%m%d_%H%M%S}.md")

    answer = next((post['message'] for post in reversed(litres_thread) if post['speaker'] == "assistant"), "(There was no answer)")
    answer = squarewheels.md_hard_breaks(answer) # e.g. reference list of "PMID: ..." lines

    # Literature checked by the script (not written by LLM); all arrays are in the saved thread too
    def _pmids_md(pmids):
        return "\n".join(f"- PMID: {pmid}" for pmid in pmids) if pmids else "(None)"

    literature_md = squarewheels.Chroma_get_abstracts_PubMed(PubMed_vectorstore, pmids_cited, output="md", level_offset=2)

    path_result.write_text(
        f"# Literature research result\n\n"
        f"- Date: {now:%Y-%m-%d %H:%M:%S (%:z)}\n"
        f"- Input: {'(written in the script)' if file_input is None else file_input}\n"
        f"- Model: {LLM_model} / Embedding: {embedding_client.model}\n\n"
        f"## Research context\n\n{research_context}\n\n"
        f"## Request\n\n{request}\n\n"
        f"## Answer\n\n{answer}\n\n"
        f"## Ciations\n\n{_pmids_md(pmids_cited)}\n\n"
        f"## Literature\n\n{literature_md}",
        encoding="utf-8",
    )
    print(f"Result saved: {path_result}")


## Entry point ----------------------------------------------------------------
if __name__ == "__main__":
    main()

