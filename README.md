# Azekra
Cosily-crafted, homely-made, and insanely-lovely [square wheels](https://en.wikipedia.org/wiki/Reinventing_the_wheel#Related_phrases).  
\*Now this project is under vibe coding — even this README was written by Claude.  

Azekra is a small toolkit for literature research with a **local** LLM.
Given a research context, it lets the LLM search PubMed on its own, collects the abstracts,
stores them in a local vector database, and writes text (e.g. an introduction) citing PMIDs.
Everything except PubMed access runs on your own computer through [LM Studio](https://lmstudio.ai/).

> Status: a personal research tool. Everything here is experimental and may change without notice. Mainly used on Windows.  
> Provided **as is, without warranty of any kind**. Use at your own risk.

---

## License

[MIT](LICENSE)

---

## What you need

| Item | Required? | Notes |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Yes | Installs Python 3.14 and all packages for you |
| [LM Studio](https://lmstudio.ai/) (with the `lms` command) | Yes | Runs the LLM and the embedding model locally |
| An LLM and an embedding model in LM Studio | Yes | Defaults: `google/gemma-4-e2b`, `text-embedding-embeddinggemma-300m-qat` |
| Internet connection | Yes | For PubMed (NCBI E-utilities) and the first download of models/packages |
| [Zotero](https://www.zotero.org/) 9 or later | Only for `002_fromZotero.py` | Must be running with local API enabled (see below) |

A local LLM needs a reasonably powerful machine. The default `gemma-4-e2b` is small and
runs on most recent PCs; larger models need more memory (and preferably a GPU).

---

## Setup

### 1. Install uv

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal afterwards and check that `uv --version` works.

### 2. Install LM Studio and the models

1. Download and install LM Studio from <https://lmstudio.ai/>, and start it once.
2. Make sure the `lms` command works in a new terminal:
   ```bash
   lms version
   ```
   If it is not found, see the LM Studio CLI docs: <https://lmstudio.ai/docs/cli>.
3. Download an LLM and an embedding model, either from the LM Studio app (Discover tab) or with `lms get`.
4. Check the exact model names. **Use these names as they are in `config.toml`** (e.g. `google/gemma-4-e2b`, not `gemma-4-e2b`):
   ```bash
   lms ls
   ```

You do not need to start the LM Studio server yourself; the scripts do it (`lms daemon up` / `lms server start`).

### 3. Get the code and install packages

```bash
git clone https://github.com/TYam94/azekra.git
cd azekra
uv sync
```

`uv sync` creates `.venv/` in the project folder and installs Python 3.14 and every package listed in `pyproject.toml`.

### 4. Create your config file

Copy the example and edit it:

```bash
cp config.example.toml config.toml
```

(On Windows without Git Bash: `copy config.example.toml config.toml`)

`config.toml` is **not** tracked by git, so personal values are safe here.

| Section / key | What to set |
|---|---|
| `[lmstudio] llm`, `embedding` | Model names exactly as shown by `lms ls` |
| `[lmstudio] base_url`, `api_key` | Leave as is unless you changed LM Studio's server port |
| `[PubMed] email` | Your email address. NCBI asks for it; without it you get a warning (it still works) |
| `[PubMed] api_key` | Optional. An [NCBI API key](https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.Coming_in_December_2018_API_Key) raises the request rate limit |
| `[paths]` | Where data is stored, and the default input file of `001_literature.py` (`litres_input`). Relative paths are resolved from the folder of `config.toml` |

### 5. (Only for Zotero features) Enable Zotero's local API

In Zotero: **Settings → Advanced → "Allow other applications on this computer to communicate with Zotero"**.
Zotero must be running when you use the Zotero features (the script tries to start it if it is not).

> The Zotero features rely on Zotero's local API (through [pyzotero](https://github.com/urschrei/pyzotero)).
> They may break with updates or specification changes of Zotero or its API.

---

## Usage

### Running from an IDE (Positron, VS Code, ...)

Select the interpreter in `.venv/` of this project (`.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on macOS/Linux).
Run with the project folder (or any folder inside it) as the working directory, so that `config.toml` is found.

### Literature research: `code/001_literature.py`

1. Write your input file. The sample `data/user/litres_sample.toml` is used by default; copy it (e.g. to `data/user/litres.toml`) and edit:
   ```toml
   # What you are working on. Used for both PubMed search and writing.
   research_context = '''
   I aim to elucidate ...
   '''

   # What the LLM should write, based on the collected literature.
   request = '''
   Please give brief introduction of that context.
   '''
   ```
   Text between `'''` and `'''` is used as is, including line breaks; no escaping is needed.
   Any language works (the LLM is told to search the database in English), but English input tends to give better results, since searching the database relies on the embedding model.
   Then point `[paths] litres_input` in `config.toml` to your file, or use one of the other ways below.
2. Run it from the project folder:
   ```bash
   uv run python code/001_literature.py
   ```

Other ways to give the input (handy when running from an IDE):

- **Another input file, from the command line**: pass any `.toml` file as an argument.
  ```bash
  uv run python code/001_literature.py data/user/another_topic.toml
  ```
- **Another input file, by editing the script**: set `input_file` in the `Input (optional)` section at the top of `code/001_literature.py`.
- **No input file at all**: write `research_context` and/or `request` directly in that section. What you write there overrides the value in the input file; if you write both, no input file is read.

The input file is chosen in this order: command-line argument > `input_file` in the script > `[paths] litres_input` in `config.toml` (default: `data/user/litres_sample.toml`).

What happens:

1. LM Studio is started and the models are loaded.
2. The LLM searches PubMed repeatedly, broadening its queries until enough papers are found.
   A search that alone hits the upper limit (3,000 by default) is discarded and the LLM is asked for narrower queries;
   when the accumulated results reach the limit, the search ends there.
3. Abstracts are fetched from PubMed (already-fetched ones are reused from the local cache).
4. New abstracts are added to the local vector database (Chroma).
5. The LLM searches that database (the whole database, including abstracts collected in earlier runs) and writes the requested text with PMID citations.
6. Both conversations (search and writing) are saved as JSON under `data/threads/`.
7. The final answer is saved as Markdown next to the input file, as `<input name>_litres_<YYYYMMDD_HHMMSS>.md`
   (e.g. `data/user/ra_genetics_litres_20261020_141615.md` for `data/user/ra_genetics.toml`).
   It contains the date, input, models, research context, request and the answer.
   When no input file is used (both values written in the script), it is saved as `litres_<YYYYMMDD_HHMMSS>.md` in the folder of `[paths] litres_input`.

The progress of each conversation is printed to the console.
The first run takes longer, because every abstract has to be embedded once; later runs only embed new ones.

### Paper summary from Zotero: `code/002_fromZotero.py`

Pulls a PDF from your local Zotero library, converts it to text, and asks the LLM a fixed set of questions about it.

```bash
uv run python code/002_fromZotero.py
```

---

## Where data goes

All data is written under `data/` (ignored by git, except `data/user/` with the sample input). With the default `config.toml`:

| Path | Content |
|---|---|
| `data/db_PubMed/PMID_abstracts.json` | Cache of fetched PubMed abstracts (the source of truth) |
| `data/db_PubMed/Chroma/` | Vector database built from the cache. Safe to delete; it is rebuilt from the JSON |
| `data/threads/` | Saved conversations (`<YYYYMMDD_HHMMSS>_<workflow>.json`) |
| `data/_temp/` | PDFs pulled from Zotero |
| `data/user/` | Your own input files (e.g. `litres.toml`) and the results written next to them (`*_litres_<YYYYMMDD_HHMMSS>.md`). Ignored by git, except the sample `litres_sample.toml` |

If you change the embedding model, a new collection is created automatically (the collection name includes the model name).

---

## Troubleshooting

| Symptom | What to check |
|---|---|
| `config.toml not found ...` | You did not create `config.toml` (step 4), or you are running from outside the project folder |
| `` `lms ...` failed (exit 1) `` with a message from LM Studio | Read the message. Try `lms daemon up` in a terminal to see whether LM Studio starts at all |
| `lms` not found | LM Studio is not installed, or the terminal/IDE was opened before installing it. Restart the terminal/IDE |
| A different model than expected is used | The name in `config.toml` does not exactly match `lms ls`; the first available model is used instead |
| Warning about NCBI email | Set `[PubMed] email` in `config.toml` |
| `Stored PubMed abstracts not found` | The PubMed search found nothing, so no cache file was created yet. Revise `research_context` in your input file |

---

## Project structure

```
azekra/
├─ code/                    Scripts you run
│  ├─ 001_literature.py
│  └─ 002_fromZotero.py
├─ src/squarewheels/        The library (imported as `squarewheels`)
├─ config.example.toml      Template of config.toml
└─ pyproject.toml / uv.lock Dependencies
```

Modules in `src/squarewheels/` are named `<prefix>_<name>.py`; the prefix tells the layer:

| Prefix | Layer | Modules |
|---|---|---|
| `cf_` | Configuration | `cf_config` (load `config.toml`) |
| `cm_` | Citation manager | `cm_zotero` |
| `db_` | Database | `db_pubmed` (search, fetch and cache abstracts) |
| `lm_` | LLM server / API | `lm_lmstudio` (start LM Studio, load models) |
| `ms_` | Messages to the LLM | `ms_chat` (conversation threads, tool calling, small LLM helpers) |
| `vs_` | Vector store | `vs_chroma` |
| `wf_` | Workflows | `wf_litres` (literature research), `wf_paper` (paper summary) |

Everything public is re-exported from the package, e.g. `squarewheels.run_litres_search_PubMed(...)`.

