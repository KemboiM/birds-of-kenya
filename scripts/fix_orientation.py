#!/usr/bin/env python3
"""Turn photos upright and remove the EXIF orientation tag.

Some photos store their pixels sideways with a tag telling viewers to turn them. The annotation
script draws boxes on the turned photo, but training reads the stored pixels, so a tagged photo
would be cropped in the wrong place. Run this after downloading and before annotating. Photos
without the tag are not touched, and running it again changes nothing.

    python3 scripts/fix_orientation.py                       # data/images/sorted_data
    python3 scripts/fix_orientation.py --images other/folder
"""

import argparse
import os

from PIL import Image, ImageOps

EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
ORIENTATION = 0x0112


def main():
    p = argparse.ArgumentParser(description="Turn photos upright and remove the EXIF orientation tag.")
    p.add_argument("--images", default=os.path.join("data", "images", "sorted_data"))
    args = p.parse_args()

    checked = fixed = 0
    for folder, _, files in sorted(os.walk(args.images)):
        for name in sorted(files):
            if not name.lower().endswith(EXTENSIONS):
                continue
            path = os.path.join(folder, name)
            checked += 1
            try:
                with Image.open(path) as im:
                    if (im.getexif().get(ORIENTATION) or 1) == 1:
                        continue
                    upright = ImageOps.exif_transpose(im)
                    upright.load()
                    fmt = im.format
                upright.save(path, format=fmt, **({"quality": 95} if fmt == "JPEG" else {}))
                fixed += 1
            except Exception as exc:
                print(f"  skipped {path}: {exc}")
    print(f"{fixed} of {checked} photos turned upright")


if __name__ == "__main__":
    main()
