
import json
import subprocess


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
