"""Download and unpack the evaluation datasets from Zenodo into <root>/raw/.

    python evaluation/get_data.py --root data

Layout produced (this is what evaluation/datasets.py expects):

    <root>/raw/vocalset_audio/FULL/<singer>/...            VocalSet audio     (2.1 GB zip)
    <root>/raw/vocalset_x/Annotated VocalSet/raw 1/csv/... Annotated-VocalSet (0.4 GB zip, annotations only)
    <root>/raw/vocadito/zenodo_data/{Audio,Annotations}    vocadito           (58 MB zip)
    <root>/raw/musdb7/*/*.stem.mp4                         MUSDB18 7 s excerpts (147 MB zip)
"""
import argparse
import os
import shutil
import urllib.request
import zipfile

SOURCES = {
    "vocalset_audio": ("https://zenodo.org/api/records/1193957/files/VocalSet.zip/content", "VocalSet.zip"),
    "vocalset_x": ("https://zenodo.org/api/records/7061507/files/Annotated%20VocalSet.zip/content", "AnnotatedVocalSet.zip"),
    "vocadito": ("https://zenodo.org/api/records/5578259/files/vocadito.zip/content", "vocadito.zip"),
    "musdb7": ("https://zenodo.org/api/records/3270814/files/MUSDB18-7-STEMS.zip/content", "MUSDB18-7-STEMS.zip"),
}


def safe_extract(zip_path: str, dest: str) -> None:
    """Extract, refusing any member that would land outside `dest`."""
    root = os.path.realpath(dest)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            target = os.path.realpath(os.path.join(dest, name))
            if target != root and not target.startswith(root + os.sep):
                raise ValueError(f"unsafe path in archive: {name}")
        zf.extractall(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--only", nargs="*", choices=list(SOURCES), help="subset of datasets")
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()

    dl = os.path.join(args.root, "_downloads")
    os.makedirs(dl, exist_ok=True)
    for name in args.only or SOURCES:
        url, fname = SOURCES[name]
        dest = os.path.join(args.root, "raw", name)
        if os.path.isdir(dest) and os.listdir(dest):
            print(f"[skip] {name}: {dest} already populated")
            continue
        zpath = os.path.join(dl, fname)
        if not os.path.exists(zpath):
            print(f"[download] {name} <- {url}")
            urllib.request.urlretrieve(url, zpath)
        os.makedirs(dest, exist_ok=True)
        print(f"[extract] {zpath} -> {dest}")
        safe_extract(zpath, dest)
        if not args.keep_zips:
            os.remove(zpath)
    if not args.keep_zips:
        shutil.rmtree(dl, ignore_errors=True)
    print("done")


if __name__ == "__main__":
    main()
