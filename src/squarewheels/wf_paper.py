# Workflow: paper summary =====================================================

## Dependency -----------------------------------------------------------------
import json
import logging
from pathlib import Path

from .ms_chat import post_clean, thread_master_append, thread_master_to_prompt

MODULE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = MODULE_DIR / "config" / "qset_TAMMICKURR_ja.json"

## Logging --------------------------------------------------------------------
logger = logging.getLogger(__name__)


## Running pre-defined thread =================================================
### Ask LLM a series of questions on one theme --------------------------------
def run_theme_and_questions(client, model, theme, qset):
    """Ask LLM a series of questions on one theme, one by one in the same thread.

    Args:
        client: OpenAI-compatible LLM client, which should have `client.chat.completions.create()` method.
        model: Name of the model to be used, which should be valid and loaded.
        theme: The theme of the series of questions, used as system prompt.
        qset: Question set. List of one or more dicts, each with 'field' (used as the speaker) and 'question'.
            "assistant" is a preserved word and not allowed as a value of 'field' (logged as an error).

    Returns:
        thread_master: Thread of the questions and answers.
    """
    thread_master = []

    # Ask questions in pre-defined list
    for question in qset:

        if question['field'] == "assistant":
            logger.error("Invalid question field.")

        thread_master = thread_master_append(
            thread_master,
            speaker=question['field'],
            speaker_message=post_clean(question['question'], speaker=question['field']),
            print_post=False,
            )

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

        # Append the response
        thread_master = thread_master_append(
            thread_master,
            speaker="assistant",
            speaker_message=post_clean(speaker_response.choices[0].message.content, "assistant"),
            print_post=False,
            )

    return(thread_master)


### Summarize paper with the TAMMICKURR question set ---------------------------
def PDF_to_MMICKURR(client, model, paper_md, qset_path=CONFIG_PATH):
    """Summarize a paper by asking LLM the TAMMICKURR question set about it (answers in Japanese, ~150 characters each).

    Args:
        client: LLM client.
        model: Model name to use.
        paper_md: Paper text (e.g. Markdown converted from PDF), passed in the system prompt.
        qset_path: Question set file (see run_theme_and_questions). Currently ignored: CONFIG_PATH is always used.

    Returns:
        tammickurr_thread: Thread of the questions and answers.
    """
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
