
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
import textwrap

JST = timezone(timedelta(hours=+9), 'JST')

MODULE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = MODULE_DIR / "config" / "qset_TAMMICKURR_ja.json"

logger = logging.getLogger(__name__)


# LLM client and loaded models is NOT checked for its validity here.
# It is the responsibility of caller.


## Utilities for threads/posts ================================================
### Clean the text into tidy post ---------------------------------------------
def post_clean(text: str, speaker: str) -> str:
    text = text.strip()
    text = re.sub(rf"\A\s*(?:{speaker}\s*:\s*)+", "", text)
    post = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return post

def post_print(post):

    if post['speaker'] == "ChatCompletionMessage":
        
        post_parsed = json.loads(post['message'])

        print(
            f"{post['speaker']} ({post['datetime']})\n"
            f"  reasoning_content:\n{textwrap.indent(post_parsed['reasoning_content'], "    ")}"
            )
        
        if post_parsed.get("tool_calls"):
            print(
                f"{post['speaker']} ({post['datetime']})\n"
                f"  tool_calls:\n{textwrap.indent(json.dumps(post_parsed.get("tool_calls")), "    ")}"
                )

    elif post['speaker'] == "tool_response":
        post_parsed = json.loads(post['message'])
        print(
            f"{post['speaker']} ({post['datetime']})\n"
            f"{textwrap.indent(
                (   f"name: {post_parsed.get("name")}\n"
                    f"tool_call_id: {post_parsed.get("tool_call_id")}\n"
                    f"content: {textwrap.shorten(post_parsed.get("content"), width=60, placeholder=" ...")}"),
                "  ")}"
            )
    else:
        print(f"{post['speaker']} ({post['datetime']})\n{textwrap.indent(post['message'], "  ")}")


### Make prompt-thread from the master-thread ---------------------------------

# fmt_dialog: Whether the output should be formatted in dialog style.
#             When true, speakers are denoted like "<speaker>: <post>" for multi-agent run.
#             When False, speakers are not denoted. Useful in one-to-one conversation.
def thread_master_to_prompt(thread_master, speaker_current, prompt_system=None, fmt_dialog=False):
    thread_prompt = []

    # Make system instruction
    if prompt_system:
        thread_prompt.append({"role": "system", "content": prompt_system})

    ### Make up the thread into prompt
    for post in thread_master:
        
        if post['speaker'] == "system":
            thread_prompt.append({"role": "system", "content": post['message']})
            continue
        if post['speaker'] == "ChatCompletionMessage" or post['speaker'] == "tool_response":
            thread_prompt.append(json.loads(post['message']))
            continue
        elif post['speaker'] == speaker_current:
            post_prompt_role = "assistant"
        else:
            post_prompt_role = "user"
        
        if fmt_dialog == True:
            post_prompt_content = f"{post['speaker']}: {post['message']}"
        else:
            post_prompt_content = post['message']

        thread_prompt.append({"role": post_prompt_role, "content": post_prompt_content})

    return thread_prompt

def thread_master_append(thread_master, speaker, speaker_message):

    speaker_post = {
        'datetime': datetime.now(JST).strftime('%Y/%m/%d %H:%M:%S'),
        'speaker': speaker,
        'message': speaker_message,
        }

    thread_master.append(speaker_post)
    post_print(speaker_post)

    return thread_master

## Runnning threads ===========================================================


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




## Wrappers
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

### Boolean judge -------------------------------------------------------------
# To pass proper client (LLM client instance) and model (valid name of the model) is the responsibility of the CALLER.
# prompt_system: Background for the question
# prompt_user:   Question which should be answered with True/False
def LLM_boolean(client, model, prompt_system, prompt_user):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user",   "content": prompt_user}
            ],
            max_tokens=10,
            temperature=0.0,
            logprobs=True,
            top_logprobs=10,
            extra_body={"reasoning_effort": "none"},
        )

    first_token_logprobs = response.choices[0].logprobs.content[0].top_logprobs

    logprob_dict = {item.token.strip(): item.logprob for item in first_token_logprobs}

    t_logprob = logprob_dict.get("T", float("-inf"))
    f_logprob = logprob_dict.get("F", float("-inf"))

    max_logprob = max(t_logprob, f_logprob)
    if max_logprob == float("-inf"):
        raise ValueError("T/F のどちらのトークンも上位スコアに含まれませんでした。")

    exp_t = np.exp(t_logprob - max_logprob)
    exp_f = np.exp(f_logprob - max_logprob)

    prob_t = exp_t / (exp_t + exp_f)
    prob_f = exp_f / (exp_t + exp_f)

    return {"T_prob": prob_t, "F_prob": prob_f}



## Quick wrappers -------------------------------------------------------------
### Translation of RQ, Ja to En ---------------------------
def rq_to_En(client, model, rq):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": """
                Please make English breif translation of following research question written in non-English language.
                Only answer with tranlation itself, in single line. Do not include line numbers, labels, or extra text.
                If given text is in English, pass original text through.
                """
                },
            {"role": "user", "content": rq}
            ],
        temperature=0,
        )
    
    rq_En = " ".join(response.choices[0].message.content.splitlines())

    return rq_En


### Research question to keywords ---------------------------------------------
def rq_to_query(client, model, rq, n_set=3, n_keywords=3):

    response = client.chat.completions.create(
        model=model, 
        messages=[
            {
                "role": "system", 
                "content": (
                    f"Generate {n_set} different search query composed of {n_keywords} English keywords related to proposed research context."
                    "Output keywords separated by single white space, one line per each set. Do not include line numbers, labels, or extra text."
                    "Consider using full-spell rather than abbreviations when it looks suitable as searching query."
                    # "Be sure to include at least one term that denote specific object/topic of given theme."
                    )
            },
            {
                "role": "user",
                "content": f"【research context】{rq}"
            }
        ],
        temperature=0.2
    )

    array_query = {line.strip() for line in response.choices[0].message.content.splitlines() if line.strip()}

    return(array_query)



