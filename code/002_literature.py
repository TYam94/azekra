
## Environment ----------------------------------------------------------------
import logging
import subprocess

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

import squarewheels

## Instances ------------------------------------------------------------------
### Logger --------------------------------------------------------------------
logger = logging.getLogger(__name__)

### Zotero --------------------------------------------------------------------
#myZotero = squarewheels.Zotero_get()

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
    model=squarewheels.lms_load_embeddingmodel("text-embedding-embeddinggemma-300m-qat"),
    check_embedding_ctx_length=False # Set this to `False` to pass things from PyMuPDF4LLMLoader to OpenAIEmbeddings
    )

### Other utilities -----------------------------------------------------------
# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)



## Main -----------------------------------------------------------------------
### From PubMed ---------------------------------------------------------------
rq_text="関節リウマチの難治性についての遺伝的素因を明らかにし、精密医療を実現したい" # ただの一例

litres_query = squarewheels.lms_rq_to_query(client=client_LLM, model=squarewheels.lms_load_LLM(), rq=rq_text)

litres_search = squarewheels.PubMed_search_PMID_array(array_query = litres_query, retmax=10000)

litres_PMIDs = list({PMID for paper in litres_search for PMID in paper['result']})

litres_absts = squarewheels.PubMed_fetch_abstract(pmids=litres_PMIDs)

litres_docs=[]
for item in litres_absts:
    doc = Document(page_content=f"[PMID: {item['PMID']}]\n[Publicated: {item['yearmonth']}]\n{item['abstract']}", metadata={"PMID": item['PMID'], "yearmonth": item['yearmonth']})
    litres_docs.append(doc)

litres_vectorstore = Chroma.from_documents(documents=litres_docs, embedding=client_embedding)

retriever = litres_vectorstore.as_retriever(search_kwargs={"k": 100})


litres_context_docs = retriever.invoke(f"What is already known about following research question, based on most recent findings?: {rq_text}")
litres_context_text = "\n\n".join(doc.page_content for doc in litres_context_docs)

response = client_LLM.chat.completions.create(
    model=squarewheels.lms_load_LLM(),
    messages=[
        {
            "role": "system",
            "content": (
                "あなたはFAQアシスタントです。提供された【参考情報】のみに基づいて質問に答えてください。\n"
                "出典を適宜示しながら回答すること。\n"
                "根拠の選定にあたっては、その発表時期も勘案し、確実かつ最新の知見に基づいて回答すること。\n\n"
                f"【参考情報】:\n{litres_context_text}"
            ),
        },
        {"role": "user", "content": f"次に述べる研究課題について、既に知られていることを教えてください。 研究課題: {rq_text}"},
    ],
    temperature=0,
)

print(response.choices[0].message.content)

