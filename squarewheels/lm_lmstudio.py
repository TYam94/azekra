
import json
import subprocess
import numpy as np

from pydantic import BaseModel, Field


## Functions/Modules ==========================================================
### Load LLM ------------------------------------------------------------------
# Load requested LLM. If not found, it will load other available option.
# Returns the name of loaded model.
def lms_load_LLM(model_request=None, model_default=None):
    
    # List available LLM in local
    lms_ls = json.loads(subprocess.run(["lms", "ls", "--llm", "--json"], capture_output=True, text=True, check=True).stdout)
    list_LLM_local = [model['modelKey'] for model in lms_ls]

    # List loaded LLM
    lms_ps = json.loads(subprocess.run(["lms", "ps", "--json"], capture_output=True, text=True, check=True).stdout)
    list_LLM_loaded = [model['modelKey'] for model in lms_ps]

    # Select one to use
    if model_request == None or not model_request in list_LLM_local:
        if model_default in list_LLM_local:
            model_to_use = model_default
        else:
            model_to_use = list_LLM_local[0]
    else:
        model_to_use = model_request

    # Load if not loaded
    if not model_to_use in list_LLM_loaded:
        subprocess.run(f'lms load {model_to_use}', check=True, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Answer
    return model_to_use


### Load embedding model ------------------------------------------------------
# Load requested embedding model. If not found, it will load other available option.
# Returns the name of loaded model.
def lms_load_embeddingmodel(model_request=None, model_default=None):
    
    # List available embedding models in local
    lms_ls = json.loads(subprocess.run(["lms", "ls", "--embedding", "--json"], capture_output=True, text=True, check=True).stdout)
    list_embedding_local = [model['modelKey'] for model in lms_ls]

    # List loaded embedding models
    lms_ps = json.loads(subprocess.run(["lms", "ps", "--json"], capture_output=True, text=True, check=True).stdout)
    list_embedding_loaded = [model['modelKey'] for model in lms_ps]

    # Select one to use
    if model_request == None or not model_request in list_embedding_local:
        if model_default in list_embedding_local:
            model_to_use = model_default
        else:
            model_to_use = list_embedding_local[0]
    else:
        model_to_use = model_request

    # Load if not loaded
    if not model_to_use in list_embedding_loaded:
        subprocess.run(f'lms load {model_to_use}', check=True, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Answer
    return model_to_use


### Boolean judge -------------------------------------------------------------
# To pass proper client (LLM client instance) and model (valid name of the model) is the responsibility of the CALLER.
# prompt_system: Background for the question
# prompt_user:   Question which should be answered with True/False
def lms_boolean(client, model, prompt_system, prompt_user):

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
### Research question to keywords ---------------------------------------------
def lms_rq_to_query(client, model, rq, n_set=3, n_keywords=3):

    response = client.chat.completions.create(
        model=model, 
        messages=[
            {
                "role": "system", 
                "content": f"""
                    Generate {n_set} different sets of {n_keywords} English keywords related to the research question.
                    Output keywords separated by single white space, one line per each set. Do not include line numbers, labels, or extra text.
                """
            },
            {
                "role": "user",
                "content": f"【research question】{rq}"
            }
        ],
        temperature=0.3
    )

    array_query = {line.strip() for line in response.choices[0].message.content.splitlines() if line.strip()}

    return(array_query)