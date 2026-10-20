# Workflow: literature research ===============================================

## Dependency -----------------------------------------------------------------
import json
import logging

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
    # "new" of each query is counted against PMIDs known before this call, so overlaps
    # between queries of the same call are counted in each of them. The top-level "new"
    # is the de-duplicated count for the whole call.
    def _tool_PubMed_search_PMID_array(array_query: str):
        litres_search = PubMed_search_PMID_array(array_query = array_query, retmax=10000, flatten=False)

        litres_known = set(litres_searched) | litres_round
        litres_call = set()
        per_query = []

        for item in litres_search:
            pmids = set(item['result'])
            summary = {"query": item['query'], "hits": len(pmids), "new": len(pmids - litres_known)}
            if len(pmids) >= 10000:
                summary["note"] = "10000 or more hits (truncated). Too broad."
            per_query.append(summary)
            litres_call |= pmids

        litres_round.update(litres_call)

        return json.dumps({
            "per_query": per_query,
            "total_hits": len(litres_call),
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
                    "Returns the number of hits for each query and for all queries together, "
                    "with how many of them are new (not found before) and already found. "
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
            "Please keep in mind following tips: \n"
            " * Be sure not to input a phrase but a set of terms as query. \n"
            " * Consider break down elements and search subset of them separately. \n"
            " * Abbreviations can be good or bad: Consider replacing abbreviations to full-spell, or vice versa. \n"
            " * It can help to replacing too strict or specific terms to more broad-sensed expression. \n"
        )
    )

    thread_master = thread_master_append(
        thread_master,
        speaker="user",
        speaker_message="Please search for the literatures related to the topic."
    )

    # Automatic user response -------------------------------------------------
    def _user_automatic():

        # A single search reached the maximum: the query is too broad (results discarded)
        if litres_status == "too_broad":
            return (
                f"This is round #{(i + 1)}. Your last search alone returned {len(litres_round)} literatures, "
                f"which reaches the limit of {litres_n_maximum}. These results were discarded; "
                f"you still have {len(litres_searched)} literatures. \n"
                "The query is too broad. Please make it narrower: use more specific terms, "
                "combine terms related to the core of the topic, and avoid overly general words."
            )

        # Accumulated results reached the maximum: enough (results of this round not added)
        if litres_status == "full":
            return (
                f"This is round #{(i + 1)}. Adding your last search would bring the total to {len(litres_candidate)} literatures, "
                f"which reaches the limit of {litres_n_maximum}. These results were not added; "
                f"you have {len(litres_searched)} literatures. \n"
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

        # Add search results of this round (de-duplicated), unless the maximum is reached
        #   too_broad: this round alone reaches the maximum -> discard, ask for narrower query
        #   full:      the total reaches the maximum        -> discard, finish with current results
        litres_candidate = set(litres_searched) | litres_round
        if len(litres_round) >= litres_n_maximum:
            litres_status = "too_broad"
        elif len(litres_candidate) >= litres_n_maximum:
            litres_status = "full"
        else:
            litres_status = None
            litres_searched = list(litres_candidate)

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


### Answering with literature vector-store (RAG) ------------------------------
def run_litres_vectorstore(
    client,
    model,
    research_context,
    vectorstore,
    pmids = None,
    request = "Please give brief introduction of that context. I assume ~10 citations will be required.",
    thread_maxlen = 100,
    ):
    """Answering with literature vector-store (RAG)

    Args:
        client: LLM client.
        model: Model name to use.
        research_context: Research context passed to system prompt.
        vectorstore: Vector-store of literatures (e.g. Chroma). Each document should have 'PMID' and 'publication_date' in its metadata.
        pmids: If given, search is limited to these PMIDs. If None, whole vector-store is searched.
        request: User's request to the assistant.
        thread_maxlen: Maximum number of rounds.

    Returns:
        thread_master: Thread, with LLM response appended.
    """
    #### Compose tools for LLM ----------------------------
    def _tool_litres_vectorstore_search(query: str, score_threshold: float = 0.5) -> str:

        # Generate retriever dynamically with given threshold
        search_kwargs = {
            "score_threshold": score_threshold,
            "k": 50,
        }

        # Limit to given PMIDs
        if pmids is not None:
            search_kwargs["filter"] = {"PMID": {"$in": list(pmids)}}

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
                docs_md = (
                    f"{docs_md}"
                    f"# PMID: {doc.metadata["PMID"]} \n"
                    f"## Publication_date: {doc.metadata["publication_date"]} \n"
                    f"{doc.page_content}\n\n"
                )
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
            "When selecting literature to cite, consider its publication date/year to make your response up-to-date. \n"
            "\n"
            f"[Context]\n{research_context}\n"
            "\n"
            "Please keep in mind following tips: \n"
            " * Adjust `score_threshold` (Threshold of similarity) properly. If there are little citations, do make lower the threshold. \n"
            " * Consider break down elements and search subset of them separately. \n"
            " * Abbreviations can be good or bad: Consider replacing abbreviations to full-spell, or vice versa. \n"
        )
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

    return thread_master
