# LLM chat: threads/posts and quick wrappers ===================================

## Dependency -----------------------------------------------------------------
import json
import logging
import re
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np

## Logging --------------------------------------------------------------------
logger = logging.getLogger(__name__)


# LLM client and loaded models is NOT checked for its validity here.
# It is the responsibility of caller.


## Utilities for threads/posts ================================================
### Clean the text into tidy post ---------------------------------------------
def post_clean(text: str, speaker: str) -> str:
    """Clean the text into tidy post.
    Removes heading speaker name (added by LLM) automatically, and tidies up blank lines.
    Mainly for multi-agent thread.

    Paragraph breaks (single blank lines) and indentation are kept, so that Markdown in the
    text renders as intended (e.g. a paragraph directly followed by "---" would become a heading).

    Args:
        text: Text to be processed.
        speaker: The name of speaker to remove.

    Returns:
        Heading speaker name removed, trailing spaces of each line removed,
        and consecutive blank lines collapsed into one.
    """
    text = text.strip()
    text = re.sub(rf"\A\s*(?:{speaker}\s*:\s*)+", "", text)
    post = "\n".join(line.rstrip() for line in text.splitlines())
    post = re.sub(r"\n{3,}", "\n\n", post).strip()
    return post


### Keep single line breaks in Markdown ---------------------------------------
def md_hard_breaks(text: str) -> str:
    """Keep single line breaks of the text when rendered as Markdown.
    Markdown joins lines separated by a single line break into one line
    (e.g. a list of "PMID: ..." lines without bullets), so two trailing spaces
    (hard line break) are appended to each line directly followed by another non-blank line.

    Args:
        text: Markdown text.

    Returns:
        Text with hard line breaks. Lines inside code blocks (``` or ~~~) are left as they are.
    """
    lines = text.splitlines()
    in_code = False

    for i, line in enumerate(lines):
        if re.match(r"\s*(```|~~~)", line):
            in_code = not in_code
            continue
        if in_code or not line.strip() or line.endswith(("  ", "\\")):
            continue
        if i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"\s*(```|~~~)", lines[i + 1]):
            lines[i] = f"{line.rstrip()}  "

    return "\n".join(lines)


### Print the post to output --------------------------------------------------
def post_print(post: str, print_reasoning=False):
    """Print the post to output.
    Tool calls ("ChatCompletionMessage") are shown as their calls, tool responses with their content shortened,
    and other posts with messages over 5000 characters truncated.

    Args:
        post: The post to print. It must be a dict with keys 'speaker', 'datetime', and 'message'.
        print_reasoning: If True, reasoning_content of tool calls is printed too (when the model returns it).

    Returns:
        None
    """
    if post['speaker'] == "ChatCompletionMessage":
        
        post_parsed = json.loads(post['message'])

        text = f"[{post['speaker']} | {post['datetime']}]"

        # reasoning_content is not returned by some models
        if print_reasoning == True and post_parsed.get("reasoning_content"):
            text = f"{text}\n  reasoning_content:\n{textwrap.indent(post_parsed.get("reasoning_content"), "    ")}"

        if post_parsed.get("tool_calls"):
            text = f"{text}\n  tool_calls:\n{textwrap.indent(json.dumps(post_parsed.get("tool_calls")), "    ")}"

        print(text)

    elif post['speaker'] == "tool_response":
        post_parsed = json.loads(post['message'])
        print(
            f"[{post['speaker']} | {post['datetime']}]\n"
            f"{textwrap.indent(
                (   f"name: {post_parsed.get("name")}\n"
                    f"tool_call_id: {post_parsed.get("tool_call_id")}\n"
                    f"content: {textwrap.shorten(post_parsed.get("content"), width=60, placeholder=" ...")}"),
                "  ")}"
            )
    else:
        # Truncate long message, keeping line breaks (textwrap.shorten would collapse them)
        message = post['message']
        if len(message) > 5000:
            message = f"{message[:5000]} ..."

        print(f"[{post['speaker']} | {post['datetime']}]\n{textwrap.indent(message, "  ")}")


### Make prompt-thread from the master-thread ---------------------------------
def thread_master_to_prompt(thread_master, speaker_current, prompt_system=None, message_tail=None, fmt_dialog=False):
    """Make prompt from thread.

    Args:
        thread_master: Master thread.
        speaker_current: The name of current speaker to whom you want to pass converted prompt.
        prompt_system: If given, this text will be used as system prompt.
        message_tail: If given, added as the last user message. Not recorded in the thread
            (e.g. information that changes every turn, like literature found so far).
        fmt_dialog: Whether the output should be formatted in dialog style.
            When False, speakers are not denoted. Suitable for one-to-one (one user and one assistant) thread.
            When true, speakers are denoted like "<speaker>: <post>". This is for multi-agent thread.

    Returns:
        thread_prompt: Generated prompt (messages) suitable for LLM.

    """
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

    if message_tail:
        thread_prompt.append({"role": "user", "content": message_tail})

    return thread_prompt


### Add the speaker's message to the thread -----------------------------------
def thread_master_append(thread_master, speaker, speaker_message, print_post=True):
    """Add the speaker's message to the thread.
    It automatically add the time (in system's local timezone) and given speaker's name.

    Args:
        thread_master: Thread to append.
        speaker: The name of the speaker.
        speaker_message: Message to add.
        print_post: Automatically print the message into output when True.

    Returns:
        thread_master: Extended thread.

    """
    speaker_post = {
        'datetime': datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S (%:z)'),
        'speaker': speaker,
        'message': speaker_message,
        }

    thread_master.append(speaker_post)

    if print_post: post_print(speaker_post)

    return thread_master


### Create response to the thread with LLM ------------------------------------
def thread_create_response(
    client, 
    model, 
    speaker_current, 
    thread_master, 
    prompt_system=None,
    message_tail=None, 
    fmt_dialog=False,
    string_noresponse = "(There was no response)",
    tools = None, 
    tool_functions = None, 
    tool_choice = None, 
    **kwargs
    ):
    """Create response to the thread with LLM.
    If LLM calls tools, the call and each tool response are appended (as "ChatCompletionMessage" and "tool_response");
    call this again to let LLM continue with the responses. Otherwise LLM's reply is appended as speaker_current.

    Args:
        client: LLM client.
        model: Model name to use.
        speaker_current: Name of the speaker.
        thread_master: Master thread.
        prompt_system: Passed to thread_master_to_prompt: if given, used as system prompt (added before the thread).
        message_tail: Passed to thread_master_to_prompt: added as the last user message, not recorded in the thread.
        fmt_dialog: Passed to thread_master_to_prompt.
        string_noresponse: Message appended when LLM neither called tools nor replied.
        tools: Tools for LLM. Passed to `tools` in `client.chat.completions.create()`.
        tool_functions: Tool functions. Names and contents should be compatible with `tools`. Ideally, should return a string (MD is okay).
        tool_choice: Tool choice. Passed to `tool_choice` in `client.chat.completions.create()`.
        **kwargs: Any other parameters you want to pass to `client.chat.completions.create()`.

    Returns:
        Thread, with LLM response appended.
    """
    # Construct a prompt from master-thread
    thread_prompt = thread_master_to_prompt(
        prompt_system=prompt_system,
        thread_master=thread_master,
        speaker_current=speaker_current,
        message_tail=message_tail,
        fmt_dialog=fmt_dialog
    )

    # Generate response
    speaker_response = client.chat.completions.create(
        model=model,
        messages=thread_prompt,
        tools=tools,
        tool_choice=tool_choice,
        **kwargs
    )

    speaker_response_message = speaker_response.choices[0].message
    logger.debug(speaker_response_message)

    # If tool is called, 
    if speaker_response_message.tool_calls:

        # Record the message into the thread
        thread_master_append(
            thread_master, 
            speaker="ChatCompletionMessage",
            speaker_message=speaker_response_message.model_dump_json(),
        )
        
        # Execute the calls
        for tool_call in speaker_response_message.tool_calls:

            # Parse and run the call
            tool_response = tool_functions[tool_call.function.name](**json.loads(tool_call.function.arguments))

            # Record the result
            thread_master_append(
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
        thread_master_append(
            thread_master, 
            speaker=speaker_current,
            speaker_message=post_clean(speaker_response_message.content, speaker_current),
        )

    # If LLM did not call a tool nor speak,
    else:
        thread_master_append(
            thread_master, 
            speaker=speaker_current,
            speaker_message=string_noresponse,
        )

    return thread_master


## Saving/loading threads =====================================================
### Save the thread to a file -------------------------------------------------
def thread_save(thread_master, path, meta=None):
    """Save the thread to a JSON file.

    Args:
        thread_master: Thread to save.
        path: Path of the file. Parent directories are created if not exist.
        meta: Any JSON-serializable information about the thread (e.g. workflow, model, context).

    Returns:
        path: Path of the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta or {}, "thread": thread_master}, f, indent=4, ensure_ascii=False)

    return path


### Load the thread from a file -----------------------------------------------
def thread_load(path):
    """Load the thread saved by `thread_save()`.

    Args:
        path: Path of the file.

    Returns:
        thread_master: Loaded thread.
        meta: Information saved with the thread.
    """
    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    return saved["thread"], saved["meta"]


## Quick wrappers -------------------------------------------------------------
### Boolean judge -------------------------------------------------------------
def LLM_boolean(client, model, prompt_system, prompt_user):
    """Ask LLM a True/False question, and get the probabilities from logprobs of the first token ("T" or "F").
    To pass proper client (LLM client instance) and model (valid name of the model) is the responsibility of the CALLER.
    The prompts should ask LLM to answer with "T" or "F".

    Args:
        client: LLM client.
        model: Model name to use. It must return logprobs.
        prompt_system: Background for the question.
        prompt_user: Question which should be answered with True/False.

    Returns:
        {"T_prob", "F_prob"}: Probabilities of True and False, normalized between the two.

    Raises:
        ValueError: If neither "T" nor "F" is among the top 10 candidates of the first token.
    """
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


### Translation of RQ, Ja to En ---------------------------
def rq_to_En(client, model, rq):
    """Translate research question into brief English, in a single line.

    Args:
        client: LLM client.
        model: Model name to use.
        rq: Research question. If it is already in English, it is passed through.

    Returns:
        Translated research question (line breaks in the reply are joined with spaces).
    """
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
    """Make search queries (sets of English keywords) from research question.

    Args:
        client: LLM client.
        model: Model name to use.
        rq: Research question (research context).
        n_set: Number of queries to ask for.
        n_keywords: Number of keywords per query to ask for.

    Returns:
        Set of queries, each as keywords separated by a space. LLM may not follow n_set/n_keywords exactly.
    """
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
