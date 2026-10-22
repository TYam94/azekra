# Workflow: literature research ===============================================

## Dependency -----------------------------------------------------------------
import json
import logging
import re

from .ms_chat import thread_create_response, thread_master_append
from .db_pubmed import PubMed_search_PMID_array

## Logging --------------------------------------------------------------------
logger = logging.getLogger(__name__)


## Running pre-defined thread =================================================
### Searching PubMed with LLM -------------------------------------------------
def run_litres_search_PubMed(
    client,
    model,
    research_context,
    litres_n_minimum = 100, 
    litres_n_maximum = 3000, 
    litres_i_minimum = 3, 
    litres_i_maximum = 20,
    litres_gain_minimum = 0.05
    ):
    """Searching PubMed with LLM

    Returns:
        thread_master: Thread of the search. The last post holds the searched PMIDs, joined by ", ".
    """
    # Literature research managers
    litres_searched = []    # Accumulated PMIDs (accepted rounds only)
    litres_round = set()    # PMIDs found in the current round, filled by the tool below

    ### Compose tools for LLM -------------------------------------------------
    # PMIDs are stored into litres_round directly; the LLM gets only a summary of counts.
    # Each query takes at most litres_n_maximum PMIDs, in the order of relevance (Best Match);
    # a query with more hits (too broad) keeps only the top of them.
    # "new" of each query is counted against PMIDs known before this call, so overlaps
    # between queries of the same call are counted in each of them. The top-level "new"
    # is the de-duplicated count for the whole call.
    def _tool_PubMed_search_PMID_array(array_query: str):
        litres_search = PubMed_search_PMID_array(array_query = array_query, retmax=litres_n_maximum, flatten=False, sort="relevance")

        litres_known = set(litres_searched) | litres_round
        litres_call = set()
        per_query = []

        for item in litres_search:
            pmids = set(item['result'])
            summary = {"query": item['query'], "hits": item['count'], "new": len(pmids - litres_known)}
            if item['count'] > len(item['result']):
                summary["kept"] = len(item['result'])
                summary["note"] = f"Too broad: only the top {len(item['result'])} by relevance were kept."
            per_query.append(summary)
            litres_call |= pmids

        litres_round.update(litres_call)

        return json.dumps({
            "per_query": per_query,
            "total_kept": len(litres_call),
            "new": len(litres_call - litres_known),
            "already_found": len(litres_call & litres_known),
        })

    LLM_tools = [
        {
            "type": "function",
            "function": {
                "name": "_tool_PubMed_search_PMID_array",
                "description": (
                    "Function to search Pubmed. "
                    "Takes an array of query (set of terms) and run these queries one by one. "
                    "Returns the number of hits for each query and the number of literatures kept for all queries together, "
                    "with how many of them are new (not found before) and already found. "
                    "For a too broad query, only the most relevant ones are kept. "
                    "Database is in English, so input query in English. "
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "array_query": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of keywords to search. Not a phrase, but a set of temrs. In English."
                        },
                    },
                    "required": ["array_query"],
                },
            },
        }
    ]

    LLM_tool_functions = {"_tool_PubMed_search_PMID_array": _tool_PubMed_search_PMID_array}


    ### Prepare the thread ----------------------------------------------------
    # Tips for querying. Repeated in every automatic user message too,
    # since small models tend to forget the system prompt as rounds go on.
    litres_tips = (
        "Please keep in mind following tips: \n"
        " * Be sure not to input a phrase but a set of terms as query. \n"
        " * PubMed searches all terms of a query with AND: more terms give fewer hits. Use 2-3 terms per query. \n"
        " * To make scope broader, remove terms from the query, rather than adding other terms. \n"
        " * Consider break down elements and search subset of them separately. \n"
        " * Abbreviations can be good or bad: Consider replacing abbreviations to full-spell, or vice versa. \n"
        " * It can help to replacing too strict or specific terms to more broad-sensed expression. \n"
        " * Do not add words like 'review', 'management' or 'study' to the query; reviews are excluded anyway. \n"
    )

    thread_master = thread_master_append(
        thread_master=[],
        speaker="system",
        speaker_message=(
            "You are working as a literature research assistant. \n"
            "Your current target is to gather tons of literature that can be related to given context. \n"
            "Continue search as long as user requests. \n"
            "Start from specific search query, and then make scope broader when user requests more information. \n"
            "\n"
            f"[Context]\n{research_context}\n"
            "\n"
            f"{litres_tips}"
        )
    )

    thread_master = thread_master_append(
        thread_master,
        speaker="user",
        speaker_message="Please search for the literatures related to the topic."
    )

    # Automatic user response -------------------------------------------------
    # Tips are appended unless the search is to be finished.
    def _user_automatic():
        user_message = _user_automatic_message()
        if "Let's finish." not in user_message:
            user_message = f"{user_message} \n\n{litres_tips}"
        return user_message

    def _user_automatic_message():

        # Accumulated results reached the maximum: enough (results of this round are added)
        if len(litres_searched) >= litres_n_maximum:
            return (
                f"This is round #{(i + 1)}. You have found {len(litres_searched)} literatures. "
                f"Newly added: {litres_gain}. This reaches the limit of {litres_n_maximum}. \n"
                "That is enough. Let's finish."
            )

        user_message = f"This is round #{(i + 1)}. You have found {len(litres_searched)} literatures. "

        try:
            user_message = (
                f"{user_message}" 
                f"Newly added: {litres_gain} ({(100 * (litres_gain / len(litres_searched))):.1f}%). "
            )
        except ZeroDivisionError:
            pass

        if len(litres_searched) < litres_n_minimum:
            user_message = (
                f"{user_message} \n"
                f"It's not enough. I want at least {litres_n_minimum}. Please continue to search for literatures."
            )
        elif len(litres_searched) < litres_n_maximum and (i + 1) < litres_i_minimum:
            user_message = (
                f"{user_message} \n"
                "It looks fine, but I want peripheral informations too. Please make another search with wider scope."
            )
        elif len(litres_searched) < litres_n_maximum and (litres_gain / len(litres_searched)) > litres_gain_minimum:
            user_message = (
                f"{user_message} \n"
                "There might be more literature to get. Please continue to search for more a bit."
            )            
        else:
            user_message = (
                f"{user_message} \n"
                "It looks like enough search was done. Let's finish."
            )

        return user_message

    ### Run the thread --------------------------------------------------------
    for i in range(litres_i_maximum):
        # Count current records
        litres_gain = len(litres_searched)
        litres_round.clear()

        # Create next response (search results are stored into litres_round by the tool)
        thread_master = thread_create_response(
            client=client,
            model=model,
            thread_master=thread_master,
            speaker_current="assistant",
            fmt_dialog=False,
            tools=LLM_tools,
            tool_choice="auto",
            tool_functions=LLM_tool_functions,
        )

        # Add search results of this round (de-duplicated).
        # When the total reaches litres_n_maximum, they are still added and the search finishes.
        litres_searched = list(set(litres_searched) | litres_round)

        # Count updated records
        litres_gain = len(litres_searched) - litres_gain

        # User's response (automatic)
        thread_master = thread_master_append(
            thread_master, 
            speaker="user",
            speaker_message=_user_automatic()
        )

        # If exit flag was given,
        if thread_master[-1]['speaker'] == "user" and "Let's finish." in thread_master[-1]['message']:
            break

    # Record the result as the last post (also when litres_i_maximum is reached)
    thread_master = thread_master_append(
        thread_master,
        speaker="user",
        speaker_message= ", ".join(litres_searched)
    )

    return thread_master


### Literature entry as Markdown (passed to LLM) ------------------------------
def _litres_md(metadata, page_content):

    return (
        f"# PMID: {metadata["PMID"]} \n"
        f"## Publication_date: {metadata["publication_date"]} \n"
        f"## Publication_types: {metadata.get("publication_types") or "Unknown"} \n"
        f"{page_content}\n\n"
    )


### Answering with literature vector-store (RAG) ------------------------------
def run_litres_vectorstore(
    client,
    model,
    research_context,
    vectorstore,
    pmids = None,
    exclude_doi_prefixes = None,
    must_cite = None,
    request = "Please give brief introduction of that context. I assume ~10 citations will be required.",
    thread_maxlen = 100,
    ):
    """Answering with literature vector-store (RAG)

    Args:
        client: LLM client.
        model: Model name to use.
        research_context: Research context passed to system prompt.
        vectorstore: Vector-store of literatures (e.g. Chroma).
            Each document should have 'PMID' and 'publication_date' in its metadata. ('publication_types' and 'retracted' are optional; entries with retracted=True are never returned).
        pmids: If given, search is limited to these PMIDs. If None, whole vector-store is searched.
        exclude_doi_prefixes: If given, entries whose 'doi_prefix' in metadata is one of these are excluded
            (e.g. ["10.xxxx", "10.yyyy"]). Entries without DOI are kept.
        must_cite: PMIDs the answer must cite. They should be in the vector-store already (missing ones are skipped with a warning).
            Their abstracts are passed to LLM with the request, regardless of the search filters above.
        request: User's request to the assistant.
        thread_maxlen: Maximum number of rounds.

    Returns:
        thread_master: Thread, with LLM response appended.
        litres_pmids: PMIDs of literature, each in an array:
            'must_cite' (given and found in the vector-store), 'seen' (must_cite + all returned by search, in the order seen),
            'cited' (seen ones cited in the answer), 'cited_unseen' (cited as "PMID: ..." but never seen; possibly made up),
            'must_cite_uncited' (must_cite ones not cited in the answer).
    """
    #### Literature passed to LLM -------------------------
    # PMIDs seen by LLM, in the order seen (must_cite first, then the ones returned by search)
    pmids_seen = []

    def _seen(pmid):
        if pmid not in pmids_seen:
            pmids_seen.append(pmid)

    # Must-cite ones are taken directly from the vector-store (not by similarity)
    must_cite = [str(pmid) for pmid in (must_cite or [])]
    must_cite_md = ""
    if must_cite:
        stored = vectorstore.get(where={"PMID": {"$in": must_cite}}, include=["metadatas", "documents"])
        must_cite_found = {metadata["PMID"]: (metadata, document) for metadata, document in zip(stored["metadatas"], stored["documents"])}

        for pmid in must_cite:
            if pmid not in must_cite_found:
                logger.warning(f"run_litres_vectorstore: must-cite PMID {pmid} is not in the vector-store; skipped")
                continue
            metadata, document = must_cite_found[pmid]
            if metadata.get("retracted"):
                logger.warning(f"run_litres_vectorstore: must-cite PMID {pmid} is retracted")
            must_cite_md = f"{must_cite_md}{_litres_md(metadata, document)}"
            _seen(pmid)

        must_cite = [pmid for pmid in must_cite if pmid in must_cite_found]

    #### Compose tools for LLM ----------------------------
    def _tool_litres_vectorstore_search(query: str, score_threshold: float = 0.5) -> str:

        # Generate retriever dynamically with given threshold
        search_kwargs = {
            "score_threshold": score_threshold,
            "k": 50,
        }

        # Exclude retracted ones ($ne keeps entries without 'retracted' in metadata)
        search_filters = [{"retracted": {"$ne": True}}]

        # Exclude given publishers by DOI prefix ($nin keeps entries without 'doi_prefix' in metadata)
        if exclude_doi_prefixes:
            search_filters.append({"doi_prefix": {"$nin": [prefix.lower() for prefix in exclude_doi_prefixes]}})

        # Limit to given PMIDs
        if pmids is not None:
            search_filters.append({"PMID": {"$in": list(pmids)}})

        search_kwargs["filter"] = search_filters[0] if len(search_filters) == 1 else {"$and": search_filters}

        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs,
        )

        docs = retriever.invoke(query)

        if not docs:
            return f"No entry had similarity score above {score_threshold}"
        else:
            docs_md = ""
            for doc in docs:
                docs_md = f"{docs_md}{_litres_md(doc.metadata, doc.page_content)}"
                _seen(doc.metadata["PMID"])
            return docs_md

    LLM_tools = [
        {
            "type": "function",
            "function": {
                "name": "_tool_litres_vectorstore_search",
                "description": (
                    "Function to search literature database. "
                    "Threshold of similarity can be adjusted via `score_threshold`. "
                    "Database is in English, so input query in English. "
                    ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Statement/Keywords to search."
                            },
                        "score_threshold": {
                            "type": "number",
                            "description": (
                                "Threshold of similarity. Default is 0.5. "
                                "Make it higher (0.7~) to obtain more narrow-ranged results. "
                                "Make it lower (~0.3) to obtain more broad results. "
                                )
                            },
                    },
                    "required": ["query"],
                    # "score_threshold" is optional
                },
            },
        }
    ]

    LLM_tool_functions = {"_tool_litres_vectorstore_search": _tool_litres_vectorstore_search}

    #### Prepare the thread -------------------------------
    thread_master = thread_master_append(
        thread_master=[],
        speaker="system",
        speaker_message=(
            "You are working as a literature research assistant. \n"
            "Respond to the user's request BASED ON database search using given tools. \n"
            "Do not proceed to answering before you get enough amount of citations. \n"
            "If single search is not enough, **MAKE MULTIPLE TOOL CALLS to refine your answer**. \n"
            "Write out your citations explicitly in-line with identifier in database, like PMID. \n"
            "Unless the user's request specifies otherwise, cite in-line as \"(PMID: 12345678)\" (multiple: \"(PMID: 12345678; PMID: 23456789)\"), "
            "and list the cited ones at the end, one per line as \"PMID: 12345678\". \n"
            "When selecting literature to cite, consider its publication date/year to make your response up-to-date. \n"
            "Also consider its publication types (study design, e.g. Randomized Controlled Trial, Meta-Analysis, Case Reports) "
            "as a rough indicator of the strength of evidence. Size and ethnicity of study cohort matters as well. \n"
            "If the user gives literature that must be cited, cite ALL of them, in addition to the ones you find by search. \n"
            "\n"
            f"[Context]\n{research_context}\n"
            "\n"
            "Please keep in mind following tips: \n"
            " * Adjust `score_threshold` (Threshold of similarity) properly. If there are little citations, do make lower the threshold. \n"
            " * Consider break down elements and search subset of them separately. \n"
            " * Abbreviations can be good or bad: Consider replacing abbreviations to full-spell, or vice versa. \n"
        )
    )

    if must_cite_md:
        request = (
            f"{request}\n"
            "\n"
            "[Literature that must be cited]\n"
            "Cite ALL of the following, in addition to the ones you find by search. \n"
            "\n"
            f"{must_cite_md}"
        )

    thread_master = thread_master_append(
        thread_master,
        speaker="user",
        speaker_message=request
    )

    #### Run the thread -----------------------------------
    for i in range(thread_maxlen):

        thread_master = thread_create_response(
            client=client,
            model=model,
            thread_master=thread_master,
            speaker_current="assistant",
            fmt_dialog=False,
            tools=LLM_tools,
            tool_choice="auto",
            tool_functions=LLM_tool_functions,
            temperature=0
        )

        # Exit when the assistant replied (content or no response) instead of calling tools
        if thread_master[-1]['speaker'] == "assistant": break

    #### Literature cited in the answer -------------------
    # Seen ones are matched as bare numbers too, so that they are caught in any citation style (e.g. "[12345678]").
    answer = thread_master[-1]['message'] if thread_master[-1]['speaker'] == "assistant" else ""
    pmids_cited = [pmid for pmid in pmids_seen if re.search(rf"(?<!\d){pmid}(?!\d)", answer)]
    pmids_cited_unseen = list(dict.fromkeys(pmid for pmid in re.findall(r"PMID:?\s*(\d+)", answer) if pmid not in pmids_seen))

    litres_pmids = {
        "must_cite": must_cite,
        "seen": pmids_seen,
        "cited": pmids_cited,
        "cited_unseen": pmids_cited_unseen,
        "must_cite_uncited": [pmid for pmid in must_cite if pmid not in pmids_cited],
    }

    if pmids_cited_unseen:
        logger.warning(f"run_litres_vectorstore: cited but never seen (possibly made up): {', '.join(pmids_cited_unseen)}")
    if litres_pmids["must_cite_uncited"]:
        logger.warning(f"run_litres_vectorstore: must-cite ones not cited: {', '.join(litres_pmids['must_cite_uncited'])}")

    return thread_master, litres_pmids
