
import logging
import platform
import subprocess
import time

from pyzotero import zotero

logger = logging.getLogger(__name__)


### Get Zotero instance -------------------------------------------------------
def Zotero_get(library_id="0", library_type="user", local=True):
    zot = zotero.Zotero(library_id, library_type, local=local)
    try:
        zot.top(limit=1)
        return zot
    except Exception: # noqa: BLE001
        logger.warning("Running local Zotero is not detected. Trying to launch...")

    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["open", "-a", "Zotero"])
    elif system == "Windows":
        subprocess.Popen(["start", "" , "zotero"], shell=True)
    else:
        subprocess.Popen(["zotero"])

    for _ in range(30):
        time.sleep(1)
        try:
            zot.top(limit=1)
            logger.warning("Local Zotero is successfully launced.")
            return zot
        except Exception:  # noqa: BLE001
            logger.warning("Running local Zotero is not detected. Trying to launch...")
            continue

    raise TimeoutError("Local Zotero could not be launched.")


###
def Zotero_makelist(zot, limit):
    libs=[]

    zotero_items = zot.top(limit=limit)

    for i, item in enumerate(zotero_items):
        logger.info(f'{i+1} / {len(zotero_items)}')

        if item['data']['itemType'] == "journalArticle":
            # Make list of authors
            authors = []
            item_creators = item['data'].get('creators')
            if item_creators:
                for creator in item_creators:
                    authors.append(f'{creator.get('firstName')} {creator.get('lastName')}')

            # Get first attached PDF key if exists
            key_PDF = None
            item_children = None
            item_children = zot.children(item['data']['key'])
            if item_children:
                for child in item_children:
                    if child['data'].get('contentType') == "application/pdf":
                        key_PDF = child['data']['key']
                        break

            # Add to the list
            libs.append({
                'journal':  item['data'].get('publicationTitle'),
                'issue':    item['data'].get('issue'),
                'volume':   item['data'].get('volume'),
                'pages':    item['data'].get('pages'),
                'PMID':     item['data'].get('PMID'),
                'title':    item['data'].get('title'),
                'authors':  authors,
                'abstract': item['data'].get('abstractNote'),
                'Zotero_key_item': item['data']['key'],
                'Zotero_key_PDF':  key_PDF,
            })
            
    return(libs)


### Pull PDF from Zotero 
# Returns the path of PDF
def Zotero_pullPDF(zot, paper, path="data/_temp"):
    if paper['Zotero_key_PDF']:
        filename_PDF = f'{path}/{paper['Zotero_key_PDF']}.pdf'
        zot.dump(paper['Zotero_key_PDF'], filename=filename_PDF)
        return(filename_PDF)
    else:
        return(None)
