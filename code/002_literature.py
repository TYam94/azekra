
## Environment ----------------------------------------------------------------
import logging
import subprocess

import json

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

import squarewheels

from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=+9), 'JST')

## Instances ------------------------------------------------------------------
### Logger --------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

### LLM/Embedding -------------------------------------------------------------
# lmstudio
subprocess.run("lms daemon up", check=True, shell=True, stdin=subprocess.DEVNULL)
subprocess.run("lms server start", check=True, shell=True, stdin=subprocess.DEVNULL)

# LLM client
LLM_client = OpenAI(
    base_url="http://localhost:1234/v1", 
    api_key="lm-studio"
    )

LLM_model = squarewheels.lms_load_LLM()

# Embedding
client_embedding = OpenAIEmbeddings(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model=squarewheels.lms_load_embeddingmodel("text-embedding-embeddinggemma-300m-qat"),
    check_embedding_ctx_length=False # Set this to `False` to pass things from PyMuPDF4LLMLoader to OpenAIEmbeddings
    )

### Other utilities -----------------------------------------------------------
# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Vector store
litres_vectorstore = None

rq_text = "関節リウマチの難治性についての遺伝的素因を明らかにし、精密医療を実現したい" # ただの一例



## Main -----------------------------------------------------------------------
def main():
    
    # Run 
    litres_PMIDs = run_litres_search_PubMed(litres_n_minimum=10, litres_n_maximum=100, litres_i_minimum=1, litres_i_maximum=5, litres_gain_minimum=0.05)

    litres_absts = squarewheels.PubMed_fetch_abstract(pmids=litres_PMIDs)

    # Abstracts into documents
    litres_docs=[]
    for item in litres_absts:

        doc = Document(
            page_content=f"[PMID: {item['PMID']}]\n[publication_date: {item['publication_date']}]\n{item['abstract']}", 
            metadata={"PMID": item['PMID'], "publication_date": item['publication_date']}
            )

        litres_docs.append(doc)

    # Make vector-store,
    global litres_vectorstore
    litres_vectorstore = Chroma.from_documents(documents=litres_docs, embedding=client_embedding)

    run_litres_vectorstore()



### From PubMed ---------------------------------------------------------------


## Iterative search of PubMed with LLM ----------------------------------------
def run_litres_search_PubMed(
    litres_n_minimum = 100, 
    litres_n_maximum = 10000, 
    litres_i_minimum = 3, 
    litres_i_maximum = 20, 
    litres_gain_minimum = 0.05
    ):

    ### Literature research managers ------------------------------------------
    litres_searched = []
    litres_exit = False

    ### Compose tools for LLM -------------------------------------------------
    def _tool_PubMed_search_PMID_array(array_query: str):
        litres_search = squarewheels.PubMed_search_PMID_array(array_query = array_query, retmax=litres_n_maximum, flatten=True)
        return litres_search 

    LLM_tools = [
        {
            "type": "function",
            "function": {
                "name": "_tool_PubMed_search_PMID_array",
                "description": (
                    "Function to search Pubmed. "
                    "Takes an array of query (set of terms) and run these queries one by one. "
                    "Returns unique set of PMID from conducted searches. "
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
    thread_master = squarewheels.thread_master_append(
        thread_master=[],
        speaker="system",
        speaker_message=(
            "You are working as a literature research assistant. \n"
            "Your current target is to gather tons of literature that can be related to given context. \n"
            "Continue search as long as user requests. \n"
            "User may request broader range of informations; make your scope wider in that case. \n"
            "\n"
            f"[Context]\n{rq_text}\n"
            "\n"
            "Please keep in mind following tips: \n"
            " * Be sure not to input a phrase but a set of terms as query. \n"
            " * Consider break down elements and search subset of them separately. \n"
            " * Abbreviations can be good or bad: Consider replacing abbreviations to full-spell, or vice versa. \n"
            " * It can help to replacing too strict or specific terms to more broad-sensed expression. \n"
        )
    )

    thread_master = squarewheels.thread_master_append(
        thread_master,
        speaker="user",
        speaker_message="Please search for the literatures related to the topic."
    )


    ### Run the thread --------------------------------------------------------
    for i in range(litres_i_maximum):

        # Construct a prompt from master-thread
        thread_prompt = squarewheels.thread_master_to_prompt(
            thread_master=thread_master,
            speaker_current="assistant"
        )

        # Generate response
        speaker_response = LLM_client.chat.completions.create(
            model=LLM_model,
            messages=thread_prompt,
            tools=LLM_tools,
            tool_choice="required",
            )

        speaker_response_message = speaker_response.choices[0].message
        logger.debug(speaker_response_message)

        # If tool is called, 
        if speaker_response_message.tool_calls:
            squarewheels.thread_master_append(
                thread_master, 
                speaker="ChatCompletionMessage",
                speaker_message=speaker_response_message.model_dump_json(),
            )

            litres_gain = len(litres_searched)
            
            for tool_call in speaker_response_message.tool_calls:
                # Parse and run the call
                tool_response = LLM_tool_functions[tool_call.function.name](**json.loads(tool_call.function.arguments))

                # Record the result
                squarewheels.thread_master_append(
                    thread_master, 
                    speaker="tool_response", 
                    speaker_message=json.dumps({
                        "tool_call_id": tool_call.id, 
                        "role": "tool", 
                        "name": tool_call.function.name, 
                        "content": ", ".join(tool_response)
                    }),
                )

                # Add them to the list
                litres_searched.extend(tool_response)
                litres_searched = list(set(litres_searched))

            
            litres_gain = len(litres_searched) - litres_gain

        # If tools is not called and the reply is generated, 
        elif speaker_response_message.content:
            squarewheels.thread_master_append(
                thread_master, 
                speaker="assistant",
                speaker_message=squarewheels.post_clean(speaker_response_message.content, "assistant"),
            )

        # If LLM did not call a tool nor speak,
        else:
            squarewheels.thread_master_append(
                thread_master, 
                speaker="assistant",
                speaker_message="(There was no response from assistant)",
            )
            litres_exit = True


        # User's response (automatic) ---------------------
        user_message = f"You have found {len(litres_searched)} literatures. "
        try:
            user_message = (
                f"{user_message}" 
                f"Newly added: {litres_gain} ({(100 * (litres_gain / len(litres_searched))):.1f}%)"
            )
        except ZeroDivisionError:
            pass

        if len(litres_searched) < litres_n_minimum:
            user_message = (
                f"{user_message} \n"
                f"It's not enough. I want at least {litres_n_minimum}. Please continue to search for literatures."
            )
        elif i < litres_i_minimum:
            user_message = (
                f"{user_message} \n"
                "I want more broader informations. Please continue to search for literatures."
            )
        elif len(litres_searched) > litres_n_maximum or (litres_gain / len(litres_searched)) < litres_gain_minimum:
            user_message = (
                f"{user_message} \n"
                "It looks like enough search was done. Let's finish."
            )
            litres_exit = True
        else:
            user_message = (
                f"{user_message} \n"
                "I want more broader informations. Please continue to search for literatures."
            )

        speaker_post = {
            'datetime': datetime.now(JST).strftime('%Y/%m/%d %H:%M:%S'),
            'speaker': "user",
            'message': user_message,
        }
        # Print and append
        thread_master.append(speaker_post)
        squarewheels.post_print(speaker_post)
        logger.info(speaker_post)

        if litres_exit: break

    return litres_searched




### 
def run_litres_vectorstore(
    thread_maxlen=100
    ):

    #### Compose tools for LLM ----------------------------
    def _tool_litres_vectorstore_search(query: str, score_threshold: float = 0.5) -> str:

        # Generate retriever dynamically with given threshold
        retriever = litres_vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": score_threshold,
                "k": 50,
            },
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
    thread_master = []

    post_system = {
        'datetime': datetime.now(JST).strftime('%Y/%m/%d %H:%M:%S'),
        'speaker': "system",
        'message': (
            "You are working as a literature research assistant. \n"
            "Respond to the user's request BASED ON database search using given tools. \n"
            "Do not proceed to answering before you get enough amount of citations. \n"
            "If single search is not enough, **MAKE MULTIPLE TOOL CALLS to refine your answer**. \n"
            "Write out what entry of database you cite explicitly with identifier in database, like PMID. \n"
            "When selecting literature to cite, consider its publication date/year to make your response up-to-date. \n"
            "\n"
            f"[Context]\n{rq_text}\n"
            "\n"
            "Please keep in mind following tips: \n"
            " * Adjust `score_threshold` (Threshold of similarity) properly. If there are little citations, do make lower the threshold. \n"
            " * Consider break down elements and search subset of them separately. \n"
            " * Abbreviations can be good or bad: Consider replacing abbreviations to full-spell, or vice versa. \n"
        )
    }
    thread_master.append(post_system)
    squarewheels.post_print(post_system)

    post_user = {
        'datetime': datetime.now(JST).strftime('%Y/%m/%d %H:%M:%S'),
        'speaker': "user",
        'message': (
            "そのContextについて、データベースを検索し、既に知られていることを教えてください。"
            "10件程度の関連文献を含めつつintroductionをしてくれると助かります。"
            )
        }
    thread_master.append(post_user)
    squarewheels.post_print(post_user)

    #### Run the thread -----------------------------------
    for i in range(thread_maxlen):

        # Construct a prompt from master-thread
        thread_prompt = squarewheels.thread_master_to_prompt(
            prompt_system=None,
            thread_master=thread_master,
            speaker_current="assistant",
            fmt_dialog=False
            )

        # Generate response
        speaker_response = LLM_client.chat.completions.create(
            model=LLM_model,
            messages=thread_prompt,
            temperature=0,
            tools=LLM_tools,
            tool_choice="auto",
            )

        speaker_response_message = speaker_response.choices[0].message
        logger.debug(speaker_response_message)

        # If tool is called, 
        if speaker_response_message.tool_calls:

            # Record message itself into the thread
            squarewheels.thread_master_append(
                thread_master, 
                speaker="ChatCompletionMessage",
                speaker_message=speaker_response_message.model_dump_json(),
            )
            
            for tool_call in speaker_response_message.tool_calls:
                # Parse and run the call
                tool_response = LLM_tool_functions[tool_call.function.name](**json.loads(tool_call.function.arguments))

                # Record the result
                squarewheels.thread_master_append(
                    thread_master, 
                    speaker="tool_response", 
                    speaker_message=json.dumps({
                        "tool_call_id": tool_call.id, 
                        "role": "tool", 
                        "name": tool_call.function.name, 
                        "content": tool_response
                    }),
                )

        # If tools is not called and the reply is generated, 
        elif speaker_response_message.content:
            # Form a response into a post
            squarewheels.thread_master_append(
                thread_master, 
                speaker="assistant", 
                speaker_message=squarewheels.post_clean(speaker_response_message.content, "assistant"),
            )

        else:
            squarewheels.thread_master_append(
                thread_master, 
                speaker="assistant", 
                speaker_message="(There was no response from assistant)",
            )
            break

    return thread_master



## Entry point ----------------------------------------------------------------
if __name__ == "__main__":
    main()

