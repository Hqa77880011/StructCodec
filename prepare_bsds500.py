"""Download public BSDS500 images and build the codec's image manifest."""

import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from urllib.request import urlopen

from PIL import Image

SOURCE = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html"
URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
COUNTS = {"train": 200, "val": 100, "test": 200}


def download(archive):
    """Download without publishing a partial archive as a complete input."""
    archive = Path(archive)
    if archive.exists():
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    with urlopen(URL, timeout=120) as response, partial.open("wb") as destination:
        shutil.copyfileobj(response, destination)
        expected = response.headers.get("Content-Length")
    if expected and partial.stat().st_size != int(expected):
        raise OSError(f"Incomplete download: {partial}")
    partial.replace(archive)


def image_members(bundle):
    """Whitelist image members; archive paths are never extracted directly."""
    selected = {split: [] for split in COUNTS}
    seen = set()
    for member in bundle.getmembers():
        parts = PurePosixPath(member.name).parts
        if len(parts) != 6 or parts[:4] != ("BSR", "BSDS500", "data", "images"):
            continue
        split, filename = parts[4:]
        if ":" in filename or "\\" in filename:
            raise ValueError(f"Unsafe image filename: {filename}")
        if split not in selected or Path(filename).suffix.lower() != ".jpg":
            continue
        if not member.isfile() or filename in seen:
            raise ValueError("BSDS500 image members must be regular files with disjoint identities")
        seen.add(filename)
        selected[split].append(member)
    for split, expected in COUNTS.items():
        if len(selected[split]) != expected:
            raise ValueError(f"Expected {expected} official {split} images, found {len(selected[split])}")
        selected[split].sort(key=lambda member: member.name)
    return selected


def prepare(archive, output):
    """Keep official train/val/test roles and emit relative image paths."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    rows = []
    with tarfile.open(archive, "r:gz") as bundle:
        selected = image_members(bundle)
        output.mkdir(parents=True, exist_ok=True)
        for split, members in selected.items():
            for member in members:
                filename = PurePosixPath(member.name).name
                with bundle.extractfile(member) as stream:
                    payload = stream.read()
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                relative = Path("images") / split / filename
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                rows.append(dict(image=relative.as_posix(), split=split,
                                 image_id=Path(filename).stem, dataset="BSDS500"))
    with (output / "images.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(source=SOURCE, archive_url=URL, counts=COUNTS,
                    roles="train fits the codec; val reserved for parameter selection; test evaluated after freezing",
                    preprocessing="original RGB files; codec loader performs configured 256px square resizing",
                    citation="Arbelaez et al., Contour Detection and Hierarchical Image Segmentation, TPAMI 2011")
    (output / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output / "images.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/BSR_bsds500.tgz"))
    parser.add_argument("--output", type=Path, default=Path("data/bsds500"))
    parser.add_argument("--offline", action="store_true", help="Use a previously obtained official archive")
    parser.add_argument("--plan", action="store_true", help="Print source and roles without downloading or writing")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(dict(source=SOURCE, download=URL, archive=str(args.archive),
                              output=str(args.output), official_counts=COUNTS), indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    if args.offline and not args.archive.is_file():
        parser.error("--offline requires an existing --archive")
    if not args.offline:
        download(args.archive)
    prepare(args.archive, args.output)


if __name__ == "__main__":
    main()
