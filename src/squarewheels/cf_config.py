# Configuration ===============================================================

## Dependency -----------------------------------------------------------------
import logging
import tomllib
from pathlib import Path

## Logging --------------------------------------------------------------------
logger = logging.getLogger(__name__)


## Loading --------------------------------------------------------------------
### Load config.toml --------------------------------------
def config_load(path=None):
    """Load config.toml.

    Args:
        path: Path of the config file. If None, config.toml is searched from the current directory up to its parents.

    Returns:
        config: Config as dict. Values in [paths] are resolved relative to the config file, as absolute Path.

    Raises:
        FileNotFoundError: If path is None and config.toml is not found.
    """
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
