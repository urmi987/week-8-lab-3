"""Download the pinned CC0 CivilComments-WILDS mirror and verify its checksum."""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = 'https://huggingface.co/datasets/shlomihod/civil-comments-wilds/resolve/3fbfeca/all_data_with_identities.csv'
SHA256 = '403e638c83a225d738a937ff98b61fd0631e30f710d57928c7766d413526b77f'
target = ROOT / 'data/all_data_with_identities.csv'
target.parent.mkdir(exist_ok=True)
if not target.exists():
    partial = target.with_suffix('.part')
    print('Downloading approximately 273 MB from the pinned public mirror...', flush=True)
    urllib.request.urlretrieve(URL, partial)
    assert hashlib.file_digest(partial.open('rb'), 'sha256').hexdigest() == SHA256, 'Checksum mismatch'
    partial.replace(target)
assert hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() == SHA256, 'Checksum mismatch'
print('CivilComments-WILDS checksum verified:', SHA256)
