# Workflow: paper summary =====================================================

## Dependency -----------------------------------------------------------------
import json
import logging
from datetime import datetime
from pathlib import Path

from .ch_chat import JST, post_clean, thread_master_to_prompt

MODULE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = MODULE_DIR / "config" / "qset_TAMMICKURR_ja.json"

## Logging --------------------------------------------------------------------
logger = logging.getLogger(__name__)


## Running pre-defined thread =================================================
### Ask LLM a series of questions on one theme --------------------------------
# client: OpenAI-compatible LLM client, which should have `client.chat.completions.create()` method.
# model:  Name of the model to be used, which should be valid and loaded.
# theme:  String, the theme of the series of questions.
# qset:   Question set. Set containing one or more dictionaries. Each dict should have 'field' and 'question'.
#         "assistant" is a preserved word and not allowed as a value of 'field'.
def run_theme_and_questions(client, model, theme, qset):

    thread_master = []

    while len(qset) > 0:
        # Ask questions in pre-defined list
        question = qset.pop(0)

        if question['field'] == "assistant":
            logger.error("Invalid question field.")

        post_user = {
            'speaker':  question['field'],
            'message':  post_clean(question['question'], speaker=question['field']),
            'datetime': datetime.now(JST).strftime('%Y/%m/%d %H:%M:%S')
            }

        thread_master.append(post_user)

        # Construct a prompt
        thread_prompt = thread_master_to_prompt(
            prompt_system=theme,
            thread_master=thread_master,
            speaker_current="assistant",
            fmt_dialog=False
            )

        # Generate responses
        speaker_response = client.chat.completions.create(
            model=model,
            messages=thread_prompt
            )

        # Form a response into a post
        speaker_post = {
            'speaker': "assistant",
            'message': post_clean(speaker_response.choices[0].message.content, "assistant"),
            'datetime': datetime.now(JST).strftime('%Y/%m/%d %H:%M:%S')
            }

        # Print and append
        thread_master.append(speaker_post)

    return(thread_master)


### 
def PDF_to_MMICKURR(client, model, paper_md, qset_path=CONFIG_PATH):

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        qset = json.load(f)

    prompt_system = f"""
    下記の論文に記載されている内容に基づいて、質問に回答してください。
    Markdown記法は極力使用せず、150文字程度の長さで、述べてください。
    【論文】
    {paper_md}
    """
    tammickurr_thread = run_theme_and_questions(client=client, model=model, theme=prompt_system, qset=qset)

    return(tammickurr_thread)
