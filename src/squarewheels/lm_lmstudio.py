
import json
import os
import subprocess
import numpy as np

from pydantic import BaseModel, Field


## Utilities ==================================================================
### Run lms command ---------------------------------------------------------
# Electron-based IDEs (Positron, VS Code) set ELECTRON_RUN_AS_NODE etc. for child processes.
# LM Studio is also Electron, so it starts as plain Node and the daemon times out.
# These variables are dropped before calling lms.
# Returns stdout. Raises RuntimeError with lms's message when it fails.
def _lms_run(args):

    env = {k: v for k, v in os.environ.items() if not k.startswith("ELECTRON_")}

    result = subprocess.run(
        ["lms", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL, env=env,
        )

    if result.returncode != 0:
        message = (result.stderr.strip() or result.stdout.strip())
        raise RuntimeError(f"`lms {' '.join(args)}` failed (exit {result.returncode}):\n{message}")

    return result.stdout


## Functions/Modules ==========================================================
### Start LM Studio -----------------------------------------------------------
def lms_start():
    """Start LM Studio: wake up the daemon and start the API server (via `lms` CLI).

    Returns:
        None

    Raises:
        RuntimeError: If `lms` fails.
    """
    _lms_run(["daemon", "up"])
    _lms_run(["server", "start"])


### Load LLM ------------------------------------------------------------------
def lms_load_LLM(model_request=None, model_default=None):
    """Load LLM in LM Studio (if not loaded yet).

    Args:
        model_request: Model key to load. If None or not available locally, model_default is used.
        model_default: Model key used instead of model_request. If this is not available either,
            the first one of local LLMs is used.

    Returns:
        Model key of the loaded model.

    Raises:
        RuntimeError: If `lms` fails.
    """
    # List available LLM in local
    lms_ls = json.loads(_lms_run(["ls", "--llm", "--json"]))
    list_LLM_local = [model['modelKey'] for model in lms_ls]

    # List loaded LLM
    lms_ps = json.loads(_lms_run(["ps", "--json"]))
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
        _lms_run(["load", model_to_use])

    # Answer
    return model_to_use


### Load embedding model ------------------------------------------------------
def lms_load_embeddingmodel(model_request=None, model_default=None):
    """Load embedding model in LM Studio (if not loaded yet).

    Args:
        model_request: Model key to load. If None or not available locally, model_default is used.
        model_default: Model key used instead of model_request. If this is not available either,
            the first one of local embedding models is used.

    Returns:
        Model key of the loaded model.

    Raises:
        RuntimeError: If `lms` fails.
    """
    # List available embedding models in local
    lms_ls = json.loads(_lms_run(["ls", "--embedding", "--json"]))
    list_embedding_local = [model['modelKey'] for model in lms_ls]

    # List loaded embedding models
    lms_ps = json.loads(_lms_run(["ps", "--json"]))
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
        _lms_run(["load", model_to_use])

    # Answer
    return model_to_use
