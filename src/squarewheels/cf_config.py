# Configuration ===============================================================

## Dependency -----------------------------------------------------------------
import logging
import tomllib
from datetime import datetime
from pathlib import Path

## Logging --------------------------------------------------------------------
logger = logging.getLogger(__name__)


## Loading --------------------------------------------------------------------
### Load config.toml --------------------------------------
# If path is None, config.toml is searched from the current directory up to its parents.
# Values in [paths] are resolved relative to the config file, and returned as absolute Path.
# Returns the config as dict.
def config_load(path=None):

    # Find config file
    if path is None:
        for dir in [Path.cwd(), *Path.cwd().parents]:
            if (dir / "config.toml").exists():
                path = dir / "config.toml"
                break
        else:
            raise FileNotFoundError(
                "config.toml not found in the current directory or its parents. "
                "Copy config.example.toml to config.toml and edit it."
                )

    path = Path(path).resolve()

    # Load
    with open(path, "rb") as f:
        config = tomllib.load(f)

    # Resolve paths relative to the config file
    for key, value in config.get("paths", {}).items():
        config["paths"][key] = (path.parent / value).resolve()

    logger.info(f"config_load: {path}")

    return config


## Time -----------------------------------------------------------------------
### Current time ------------------------------------------
# Timezone-aware current time in the system's local timezone.
def time_now():
    return datetime.now().astimezone()
