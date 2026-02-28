#!/usr/bin/env python3
"""
concat_tiles.py

Concatenate a folder/glob/list of A5 (or A5-ish) PDFs into ONE PDF of A4 LANDSCAPE
sheets with TWO tiles per sheet (side-by-side).

Usage:
  python concat_tiles.py albumtile_out/*.pdf -o print-a4.pdf
  python concat_tiles.py albumtile_out/ -o print-a4.pdf
  python concat_tiles.py albumtile_out/*.a5.pdf other_dir/ -o print-a4.pdf

Install dependency:
  pip install pypdf
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List

from pypdf import PdfReader, PdfWriter, PageObject, PaperSize
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    DecodedStreamObject,
)


def gather_inputs(inputs: List[str]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        p = Path(item)

        if p.is_dir():
            files.extend(sorted(p.glob("*.pdf")))
            continue

        matches = glob.glob(item)
        if matches:
            files.extend(sorted(Path(m) for m in matches))
        elif p.exists():
            files.append(p)

    # de-dupe (preserve order), then stable sort by filename
    seen = set()
    out: List[Path] = []
    for f in files:
        rf = f.resolve()
        if rf not in seen:
            seen.add(rf)
            out.append(rf)

    out.sort(key=lambda x: (x.name.lower(), str(x).lower()))
    return out


def page_dims(page):
    """Return (width, height) in points from the page's mediabox."""
    box = page.mediabox
    return float(box.width), float(box.height)


def page_to_form_xobject(writer, page):
    """
    Clone a source page into the writer and convert it to a Form XObject.

    Uses writer.add_page() to properly resolve all indirect references (fonts,
    images, etc.), then extracts the cloned page's content and resources into
    a Form XObject, and removes the temporary page from the writer.
    """
    sw, sh = page_dims(page)

    # Add the page to the writer — this clones all referenced objects
    writer.add_page(page)
    cloned = writer.pages[-1]

    # Extract content stream
    contents = cloned["/Contents"]
    if hasattr(contents, "get_object"):
        contents = contents.get_object()
    if hasattr(contents, "get_data"):
        stream_data = contents.get_data()
    else:
        stream_data = b"\n".join(s.get_object().get_data() for s in contents)

    # Build Form XObject
    form_xobj = DecodedStreamObject()
    form_xobj.set_data(stream_data)
    form_xobj.update({
        NameObject("/Type"): NameObject("/XObject"),
        NameObject("/Subtype"): NameObject("/Form"),
        NameObject("/BBox"): ArrayObject([
            FloatObject(0), FloatObject(0), FloatObject(sw), FloatObject(sh),
        ]),
    })

    # Copy resources from the cloned page (already resolved in writer)
    if "/Resources" in cloned:
        form_xobj[NameObject("/Resources")] = cloned["/Resources"]

    # Remove the temporary page from the writer's page list
    from pypdf.generic import NumberObject
    del writer._root_object["/Pages"]["/Kids"][-1]
    writer._root_object["/Pages"][NameObject("/Count")] = NumberObject(
        len(writer._root_object["/Pages"]["/Kids"])
    )

    # Register the Form XObject in the writer
    return writer._add_object(form_xobj), sw, sh


def make_imposed_pdf(pages, output_path, *, allow_upscale=True):
    """Create 2-up A4 landscape PDF from a list of source pages."""
    writer = PdfWriter()

    a4_w = float(PaperSize.A4.height)  # 842 (landscape width)
    a4_h = float(PaperSize.A4.width)   # 595 (landscape height)
    slot_w = a4_w / 2.0
    slot_h = a4_h

    # Convert all source pages to Form XObjects first
    form_refs = []
    for src_page in pages:
        ref, sw, sh = page_to_form_xobject(writer, src_page)
        form_refs.append((ref, sw, sh))

    # Build imposed sheets
    for pair_start in range(0, len(form_refs), 2):
        pair = form_refs[pair_start:pair_start + 2]

        content_parts = []
        xobjects = {}

        for slot_idx, (ref, sw, sh) in enumerate(pair):
            slot_x = slot_idx * slot_w

            scale = min(slot_w / sw, slot_h / sh)
            if not allow_upscale:
                scale = min(scale, 1.0)

            new_w = sw * scale
            new_h = sh * scale
            tx = slot_x + (slot_w - new_w) / 2.0
            ty = (slot_h - new_h) / 2.0

            xobj_name = f"/Tile{slot_idx}"
            xobjects[NameObject(xobj_name)] = ref

            content_parts.append(
                f"q {scale:.6f} 0 0 {scale:.6f} {tx:.4f} {ty:.4f} cm {xobj_name} Do Q"
            )

        content_stream = "\n".join(content_parts)
        stream_obj = DecodedStreamObject()
        stream_obj.set_data(content_stream.encode("latin-1"))

        page_obj = PageObject.create_blank_page(writer, width=a4_w, height=a4_h)
        resources = DictionaryObject()
        resources[NameObject("/XObject")] = DictionaryObject(xobjects)
        page_obj[NameObject("/Resources")] = resources
        page_obj[NameObject("/Contents")] = writer._add_object(stream_obj)

        writer.add_page(page_obj)

    with open(output_path, "wb") as f:
        writer.write(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Impose A5 tiles 2-up onto A4 landscape.")
    ap.add_argument("inputs", nargs="+", help="PDFs, globs, and/or directories")
    ap.add_argument("-o", "--output", default="print-a4.pdf", help="Output PDF filename")
    ap.add_argument("--no-upscale", action="store_true", help="Do not enlarge smaller pages")
    args = ap.parse_args()

    pdf_files = gather_inputs(args.inputs)
    if not pdf_files:
        print("No PDF files found.", file=os.sys.stderr)
        return 2

    pages = []
    for f in pdf_files:
        r = PdfReader(str(f))
        pages.extend(r.pages)

    if not pages:
        print("No pages found in input PDFs.", file=os.sys.stderr)
        return 2

    out_path = Path(args.output).resolve()
    make_imposed_pdf(pages, out_path, allow_upscale=not args.no_upscale)
    print(f"Wrote {len(pages)} tiles onto {(len(pages) + 1) // 2} sheets: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
