#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import zipfile
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import requests
from PIL import Image, ImageDraw, ImageFilter

import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer

from reportlab.lib.pagesizes import A5
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ============================================================
# Inter fonts: allow "drop Inter.zip next to the script"
# ============================================================
HERE = Path(__file__).resolve().parent
FONT_DIR = HERE / "fonts"
INTER_ZIP = HERE / "Inter.zip"

INTER_STATIC_MAP = {
    "Inter-Light.ttf": "Inter_18pt-Light.ttf",
    "Inter-Regular.ttf": "Inter_18pt-Regular.ttf",
    "Inter-SemiBold.ttf": "Inter_18pt-SemiBold.ttf",
}
INTER_TABULAR_FONT = "Inter-Regular-Tabular.ttf"
INTER_LIGHT_TABULAR_FONT = "Inter-Light-Tabular.ttf"


def make_tnum_font(src: Path, dst: Path, ps_name: str) -> None:
    """Derive Inter Regular with tabular digit glyphs (metrics + outlines) baked in."""
    from fontTools.ttLib import TTFont as FTFont
    font = FTFont(str(src))
    gsub = font.get("GSUB")
    if not gsub:
        font.save(str(dst))
        return

    # Build substitution map: proportional glyph name -> tabular glyph name
    tnum_indices = {
        idx
        for fr in gsub.table.FeatureList.FeatureRecord
        if fr.FeatureTag == "tnum"
        for idx in fr.Feature.LookupListIndex
    }
    subst: dict = {}
    for idx in tnum_indices:
        for sub in gsub.table.LookupList.Lookup[idx].SubTable:
            if hasattr(sub, "mapping"):
                subst.update(sub.mapping)

    if not subst:
        font.save(str(dst))
        return

    hmtx = font["hmtx"].metrics

    # Copy advance width + lsb from each tabular glyph onto its proportional counterpart.
    # We intentionally skip glyph outlines: most .tf glyphs are composites that reference
    # the proportional glyph, so copying the outline would create a self-referencing loop.
    for prop, tf in subst.items():
        if tf in hmtx:
            hmtx[prop] = hmtx[tf]

    # Give the font a unique internal PostScript name so ReportLab's dynamic font
    # registry doesn't collapse it onto the base Inter face.
    for record in font["name"].names:
        if record.nameID == 6:  # PostScript name
            record.string = ps_name.encode(record.getEncoding() or "latin-1")

    font.save(str(dst))

def clean_text(s: str) -> str:
    """Collapse whitespace (incl newlines) to single spaces."""
    return " ".join((s or "").split())

def ensure_inter_fonts() -> None:
    FONT_DIR.mkdir(exist_ok=True)

    missing = [name for name in INTER_STATIC_MAP.keys() if not (FONT_DIR / name).exists()]
    if missing:
        if not INTER_ZIP.exists():
            raise FileNotFoundError(
                "Inter fonts not found.\n\n"
                "Do one of:\n"
                "  1) Put Inter.zip next to albumtile.py (recommended)\n"
                "  2) Or create fonts/ and copy in:\n"
                "     - Inter-Light.ttf\n"
                "     - Inter-Regular.ttf\n"
                "     - Inter-SemiBold.ttf\n"
            )

        with zipfile.ZipFile(INTER_ZIP, "r") as z:
            members = set(z.namelist())

            for out_name, zip_name in INTER_STATIC_MAP.items():
                candidate = f"static/{zip_name}"
                if candidate not in members:
                    hits = [m for m in members if m.endswith(f"/static/{zip_name}")]
                    if not hits:
                        continue
                    candidate = hits[0]

                with z.open(candidate) as src, open(FONT_DIR / out_name, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        missing_after = [name for name in INTER_STATIC_MAP.keys() if not (FONT_DIR / name).exists()]
        if missing_after:
            raise RuntimeError(
                "Failed to extract required fonts from Inter.zip.\n"
                "I expected to find these under Inter.zip's static/ folder:\n"
                + "\n".join(f"  - {INTER_STATIC_MAP[n]}" for n in missing_after)
            )

    tabular_path = FONT_DIR / INTER_TABULAR_FONT
    if not tabular_path.exists():
        make_tnum_font(FONT_DIR / "Inter-Regular.ttf", tabular_path, "Inter18pt-Regular-Tabular")

    light_tabular_path = FONT_DIR / INTER_LIGHT_TABULAR_FONT
    if not light_tabular_path.exists():
        make_tnum_font(FONT_DIR / "Inter-Light.ttf", light_tabular_path, "Inter18pt-Light-Tabular")

def register_fonts() -> None:
    ensure_inter_fonts()
    pdfmetrics.registerFont(TTFont("Inter-Light", str(FONT_DIR / "Inter-Light.ttf")))
    pdfmetrics.registerFont(TTFont("Inter-Regular", str(FONT_DIR / "Inter-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("Inter-SemiBold", str(FONT_DIR / "Inter-SemiBold.ttf")))
    pdfmetrics.registerFont(TTFont("Inter-Tabular", str(FONT_DIR / INTER_TABULAR_FONT)))
    pdfmetrics.registerFont(TTFont("Inter-Light-Tabular", str(FONT_DIR / INTER_LIGHT_TABULAR_FONT)))


# ============================================================
# Input parsing: album.link / Apple Music / raw numeric id
# ============================================================
ALBUM_LINK_RE = re.compile(r"album\.link/i/(\d+)")
APPLE_MUSIC_ID_PATH_RE = re.compile(r"/id(\d+)")
APPLE_MUSIC_I_PARAM_RE = re.compile(r"[?&]i=(\d+)")
APPLE_MUSIC_TRAILING_ID_RE = re.compile(r"/(\d+)(?:\?.*)?$")
DIGITS_RE = re.compile(r"^\d+$")

APPLE_MUSIC_STOREFRONT_RE = re.compile(r"music\.apple\.com/([a-z]{2})/", re.IGNORECASE)

def extract_storefront_country(s: str) -> str | None:
    m = APPLE_MUSIC_STOREFRONT_RE.search(s)
    return m.group(1).lower() if m else None

def extract_itunes_collection_id(s: str) -> str:
    """
    Accepts:
      - https://album.link/i/<digits>
      - https://music.apple.com/.../id<digits>
      - https://music.apple.com/...?...&i=<digits>...
      - https://music.apple.com/.../<digits>  (trailing id path segment)
      - <digits>
    Returns the numeric iTunes collection id (album/collection id).
    """
    s = s.strip()

    m = ALBUM_LINK_RE.search(s)
    if m:
        return m.group(1)

    m = APPLE_MUSIC_ID_PATH_RE.search(s)
    if m:
        return m.group(1)

    m = APPLE_MUSIC_I_PARAM_RE.search(s)
    if m:
        return m.group(1)

    if "music.apple.com" in s:
        m = APPLE_MUSIC_TRAILING_ID_RE.search(s)
        if m:
            return m.group(1)

    if DIGITS_RE.match(s):
        return s

    raise ValueError(
        "Could not extract album id. Expected:\n"
        "  - https://album.link/i/<digits>\n"
        "  - https://music.apple.com/.../id<digits>\n"
        "  - https://music.apple.com/...?...&i=<digits>\n"
        "  - https://music.apple.com/.../<digits>\n"
        "  - or just <digits>\n"
    )


# ============================================================
# iTunes lookup (metadata + tracks + artwork)
# ============================================================
@dataclass
class Album:
    title: str
    artist: str
    year: str
    genre: str
    qr_url: str
    artwork: Image.Image
    tracks: List[Tuple[str, str]]  # (title, m:ss)

def fmt_mmss(ms: int) -> str:
    secs = int(round(ms / 1000.0))
    m, s = divmod(secs, 60)
    return f"{m}:{s:02d}"

def fetch_itunes_album(collection_id: str, qr_url: str, country: str | None = None) -> Album:
    lookup = f"https://itunes.apple.com/lookup?id={collection_id}&entity=song"
    if country:
        lookup += f"&country={country}"
    r = requests.get(lookup, timeout=25)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    if not results:
        raise ValueError("No results from iTunes lookup API.")

    album_info = results[0]
    title = clean_text(album_info.get("collectionName", "Unknown Album"))
    artist = clean_text(album_info.get("artistName", "Unknown Artist"))
    genre = clean_text(album_info.get("primaryGenreName", "Unknown"))
    release_date = album_info.get("releaseDate", "")
    year = release_date[:4] if release_date else "—"

    art_url = album_info.get("artworkUrl100") or album_info.get("artworkUrl60")
    if not art_url:
        raise ValueError("No artwork URL found.")
    hi_res = re.sub(r"/\d+x\d+bb\.", "/1000x1000bb.", art_url)

    art_resp = requests.get(hi_res, timeout=25)
    art_resp.raise_for_status()
    artwork = Image.open(BytesIO(art_resp.content)).convert("RGB")

    tracks: List[Tuple[str, str]] = []
    for item in results[1:]:
        if item.get("wrapperType") != "track":
            continue
        name = item.get("trackName")
        dur = item.get("trackTimeMillis")
        if not name or not dur:
            continue
        tracks.append((clean_text(name), fmt_mmss(int(dur))))

    return Album(
        title=title,
        artist=artist,
        year=year,
        genre=genre,
        qr_url=qr_url,
        artwork=artwork,
        tracks=tracks,
    )


# ============================================================
# Image helpers
# ============================================================
def make_art_shadow(
    art: Image.Image,
    art_size_pt: float,
    offset_x_pt: float = 2.0,
    offset_y_pt: float = 4.0,
    blur_pt: float = 20.0,
    spread_pt: float = 5.0,
    opacity: float = 0.05,
) -> tuple:
    """
    Emulates a CSS/Figma drop shadow.  Returns (shadow_image, pad_left_pt, pad_bottom_pt)
    where the pad values tell the caller how far the shadow canvas extends beyond
    the art's left/bottom edges, so the shadow can be placed correctly in the PDF.
    """
    scale     = art.width / art_size_pt  # px per pt
    offset_x_px = round(offset_x_pt * scale)
    offset_y_px = round(offset_y_pt * scale)   # positive → down in image space
    # CSS/Figma: blur is the radius; Pillow GaussianBlur takes sigma = radius/2
    sigma_px  = max(1, round(blur_pt   * scale / 2))
    spread_px = round(spread_pt * scale)
    pad_blur  = round(sigma_px * 3)  # 3σ ≈ invisible tail

    # Canvas extends only right and below the art — no left/top bleed
    pad_right  = offset_x_px + spread_px + pad_blur
    pad_bottom = offset_y_px + spread_px + pad_blur

    canvas = Image.new("RGBA", (art.width + pad_right,
                                art.height + pad_bottom), (0, 0, 0, 0))
    solid = Image.new("RGBA", (art.width + 2 * spread_px, art.height + 2 * spread_px),
                      (0, 0, 0, int(255 * opacity)))
    # Negative paste coords are fine — PIL clips the source, keeping shadow in-bounds
    canvas.paste(solid, (offset_x_px - spread_px, offset_y_px - spread_px))
    blurred = canvas.filter(ImageFilter.GaussianBlur(sigma_px))
    return blurred, 0.0, pad_bottom / scale


def square_center_crop(img: Image.Image) -> Image.Image:
    if img.width == img.height:
        return img
    side = min(img.width, img.height)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    return img.crop((left, top, left + side, top + side))


# ============================================================
# QR: rounded modules + rounded finder eyes (grid aligned)
# ============================================================
def _draw_rounded_finder(draw: ImageDraw.ImageDraw, x: int, y: int, m: int) -> None:
    r_outer = max(1, int(m * 1.3))
    r_mid = max(1, int(m * 1.0))
    r_inner = max(1, int(m * 0.9))

    draw.rounded_rectangle([x, y, x + 7*m, y + 7*m], radius=r_outer, fill=(0, 0, 0, 255))
    draw.rounded_rectangle([x + 1*m, y + 1*m, x + 6*m, y + 6*m], radius=r_mid, fill=(255, 255, 255, 255))
    draw.rounded_rectangle([x + 2*m, y + 2*m, x + 5*m, y + 5*m], radius=r_inner, fill=(0, 0, 0, 255))

def make_rounded_qr_png(url: str, opacity: float = 0.8) -> Image.Image:
    border = 1

    qr_tmp = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_Q, box_size=10, border=border)
    qr_tmp.add_data(url)
    qr_tmp.make(fit=True)

    n = qr_tmp.modules_count
    total = n + 2 * border

    module_px = 16
    img_px = total * module_px

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_Q, box_size=module_px, border=border)
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=RoundedModuleDrawer(),
        fill_color="black",
        back_color="white",
    ).convert("RGBA")

    if img.size != (img_px, img_px):
        img = img.resize((img_px, img_px), Image.NEAREST)

    draw = ImageDraw.Draw(img)
    finder_mods = 7

    def mod_to_px(mx: int, my: int) -> Tuple[int, int]:
        return mx * module_px, my * module_px

    def wipe_finder(mx: int, my: int) -> None:
        x, y = mod_to_px(mx, my)
        draw.rectangle([x, y, x + finder_mods*module_px, y + finder_mods*module_px], fill=(255, 255, 255, 255))

    tl = (border, border)
    tr = (border + n - 7, border)
    bl = (border, border + n - 7)

    wipe_finder(*tl); wipe_finder(*tr); wipe_finder(*bl)
    _draw_rounded_finder(draw, *mod_to_px(*tl), module_px)
    _draw_rounded_finder(draw, *mod_to_px(*tr), module_px)
    _draw_rounded_finder(draw, *mod_to_px(*bl), module_px)

    alpha = img.getchannel("A")
    alpha = alpha.point(lambda a: int(a * opacity))
    img.putalpha(alpha)
    return img


# ============================================================
# Text helpers
# ============================================================
def ellipsize(text: str, font: str, size: float, max_width: float) -> str:
    text = clean_text(text)
    if pdfmetrics.stringWidth(text, font, size) <= max_width:
        return text
    ell = "…"
    ell_w = pdfmetrics.stringWidth(ell, font, size)
    if ell_w >= max_width:
        return ell

    lo, hi = 0, len(text)
    best = ell
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + ell
        if pdfmetrics.stringWidth(candidate, font, size) <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best

def wrap_to_width_words(words: List[str], font: str, size: float, max_width: float) -> List[str]:
    if not words:
        return [""]
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        if pdfmetrics.stringWidth(trial, font, size) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def wrap_text(text: str, font: str, size: float, max_width: float) -> List[str]:
    words = clean_text(text).split()
    return wrap_to_width_words(words, font, size, max_width)


# ============================================================
# Track list layout (fixed indent across all track numbers)
# ============================================================
def can_fit_in_two_columns(
    tracks: List[Tuple[str, str]],
    font: str,
    font_prefix: str,
    fs: float,
    leading: float,
    top_y: float,
    bottom_y: float,
    col1_left: float,
    col1_right: float,
    col2_left: float,
    col2_right: float,
    gutter_titles_times: float,
) -> bool:
    avail = top_y - bottom_y
    col_idx = 0
    used = 0.0

    max_prefix = f"{len(tracks)}. "
    fixed_prefix_w = pdfmetrics.stringWidth(max_prefix, font_prefix, fs)

    for _i, (name, _dur) in enumerate(tracks, start=1):
        col_left, col_right = (col1_left, col1_right) if col_idx == 0 else (col2_left, col2_right)
        max_title_w = (col_right - col_left) - gutter_titles_times

        title_w = max_title_w - fixed_prefix_w
        if title_w < 10:
            title_w = 10

        title_lines = wrap_text(name, font, fs, title_w)
        needed = len(title_lines) * leading + 1.0

        if used + needed <= avail:
            used += needed
        else:
            col_idx += 1
            if col_idx >= 2:
                return False
            used = needed
            if used > avail:
                return False

    return True

def draw_tracklist_two_columns_autoshrink(
    c: canvas.Canvas,
    tracks: List[Tuple[str, str]],
    font_title: str,
    font_time: str,
    fs_start: float,
    leading_factor: float,
    top_y: float,
    bottom_y: float,
    col1_left: float,
    col1_right: float,
    col2_left: float,
    col2_right: float,
    gutter_titles_times: float,
) -> None:
    fs = fs_start
    min_fs = 4.5

    while fs >= min_fs:
        leading = fs * leading_factor
        if can_fit_in_two_columns(
            tracks, font_title, font_time, fs, leading, top_y, bottom_y,
            col1_left, col1_right, col2_left, col2_right,
            gutter_titles_times
        ):
            _draw_tracklist_two_columns(
                c, tracks, font_title, font_time, fs, leading,
                top_y, bottom_y,
                col1_left, col1_right, col2_left, col2_right,
                gutter_titles_times,
            )
            return
        fs -= 0.2

    leading = min_fs * leading_factor
    _draw_tracklist_two_columns(
        c, tracks, font_title, font_time, min_fs, leading,
        top_y, bottom_y,
        col1_left, col1_right, col2_left, col2_right,
        gutter_titles_times,
    )

def _draw_tracklist_two_columns(
    c: canvas.Canvas,
    tracks: List[Tuple[str, str]],
    font_title: str,
    font_time: str,
    fs: float,
    leading: float,
    top_y: float,
    bottom_y: float,
    col1_left: float,
    col1_right: float,
    col2_left: float,
    col2_right: float,
    gutter_titles_times: float,
) -> None:
    cols = [(col1_left, col1_right), (col2_left, col2_right)]
    col_idx = 0
    x_left, x_right = cols[col_idx]
    y = top_y

    # Fixed indent based on the widest prefix (e.g. "24. "), measured in the prefix font
    max_prefix = f"{len(tracks)}. "
    fixed_prefix_w = pdfmetrics.stringWidth(max_prefix, font_time, fs)

    for i, (name, dur) in enumerate(tracks, start=1):
        if y < bottom_y:
            col_idx += 1
            if col_idx >= 2:
                return
            x_left, x_right = cols[col_idx]
            y = top_y

        max_title_w = (x_right - x_left) - gutter_titles_times
        title_w = max_title_w - fixed_prefix_w
        if title_w < 10:
            title_w = 10

        prefix = f"{i}. "
        title_lines = wrap_text(name, font_title, fs, title_w)

        # ---- first line: prefix then title starting at fixed indent ----
        c.setFont(font_time, fs)
        c.setFillGray(0.2)
        c.drawString(x_left, y, prefix)
        c.setFont(font_title, fs)
        c.setFillColorRGB(0, 0, 0)
        if title_lines:
            c.drawString(x_left + fixed_prefix_w, y, title_lines[0])

        c.setFont(font_time, fs * 0.85)
        c.setFillGray(0.2)
        c.drawRightString(x_right, y, dur)
        c.setFillGray(0.0)

        y -= leading

        # ---- continuation lines ----
        c.setFont(font_title, fs)
        c.setFillColorRGB(0, 0, 0)
        for cont in title_lines[1:]:
            if y < bottom_y:
                col_idx += 1
                if col_idx >= 2:
                    return
                x_left, x_right = cols[col_idx]
                y = top_y
                max_title_w = (x_right - x_left) - gutter_titles_times
                title_w = max_title_w - fixed_prefix_w
                if title_w < 10:
                    title_w = 10
            c.drawString(x_left + fixed_prefix_w, y, cont)
            y -= leading

        y -= 1.0


# ============================================================
# Coordinate helper
# ============================================================
def y_from_top(H: float, y_top: float) -> float:
    return H - y_top


# ============================================================
# Template marks (fold/cut)
# ============================================================
def draw_template_marks(c: canvas.Canvas, W: float, H: float) -> None:
    # Fold/score markers around hinge line (your existing positions)
    y1 = y_from_top(H, 294.5)
    y2 = y_from_top(H, 303.5)

    # Grey “tabs” left and right (no stroke)
    c.setFillColorRGB(0.85, 0.85, 0.85)
    c.rect(0, y_from_top(H, 294 + 10), 30, 10, stroke=0, fill=1)
    c.rect(W - 30, y_from_top(H, 294 + 10), 30, 10, stroke=0, fill=1)

    # Black fold lines
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(1)
    c.line(0.5, y1, 20.5, y1)
    c.line(0.5, y2, 20.5, y2)
    c.line(W - 20.5, y1, W - 0.5, y1)
    c.line(W - 20.5, y2, W - 0.5, y2)


def draw_thick_outer_border(c: canvas.Canvas, W: float, H: float) -> None:
    # Inner white area MUST be exactly this
    inner_x, inner_y_top, inner_w, inner_h = 30.0, 78.0, 360.0, 442.08
    inner_y = y_from_top(H, inner_y_top + inner_h)

    lw = 9.0
    half = lw / 2.0

    c.setStrokeColorRGB(0.85, 0.85, 0.85)
    c.setLineWidth(lw)

    # Draw border OUTSIDE so it doesn't steal area from the inner white
    c.rect(inner_x - half, inner_y - half, inner_w + lw, inner_h + lw, stroke=1, fill=0)


# ============================================================
# Front metadata block (7 lines total)
# ============================================================
def draw_front_metadata_block(
    c: canvas.Canvas,
    album: Album,
    x_left: float,
    x_right: float,
    y_top: float,
    max_lines: int = 7,
) -> None:
    base_title = 10.08
    base_artist = 10.08
    base_meta = 9.5

    title_txt = clean_text(album.title)
    artist_txt = clean_text(album.artist)
    meta_txt = clean_text(f"{album.genre} · {album.year}")

    for scale in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
        fs_title = base_title * scale
        fs_artist = base_artist * scale
        fs_meta = base_meta * scale

        w = x_right - x_left

        title_lines = wrap_text(title_txt, "Inter-SemiBold", fs_title, w)
        artist_lines = wrap_text(artist_txt, "Inter-Regular", fs_artist, w)
        meta_lines = wrap_text(meta_txt, "Inter-Light", fs_meta, w)

        total = len(title_lines) + len(artist_lines) + len(meta_lines)
        if total <= max_lines:
            y = y_top
            lead_title = fs_title * 1.15
            lead_artist = fs_artist * 1.15
            lead_meta = fs_meta * 1.15

            c.setFillColorRGB(0, 0, 0)
            c.setFont("Inter-SemiBold", fs_title)
            for line in title_lines:
                c.drawRightString(x_right, y, line)
                y -= lead_title

            c.setFont("Inter-Regular", fs_artist)
            for line in artist_lines:
                c.drawRightString(x_right, y, line)
                y -= lead_artist

            c.setFont("Inter-Light", fs_meta)
            c.setFillGray(0.2)
            for line in meta_lines:
                c.drawRightString(x_right, y, line)
                y -= lead_meta
            c.setFillGray(0.0)
            return

    fs_title = base_title * 0.7
    fs_artist = base_artist * 0.7
    fs_meta = base_meta * 0.7
    w = x_right - x_left

    artist_lines = wrap_text(artist_txt, "Inter-Regular", fs_artist, w)
    meta_lines = wrap_text(meta_txt, "Inter-Light", fs_meta, w)

    reserved = 2
    title_budget = max(1, max_lines - reserved)

    raw_title_lines = wrap_text(title_txt, "Inter-SemiBold", fs_title, w)
    if len(raw_title_lines) > title_budget:
        kept = raw_title_lines[:title_budget]
        kept[-1] = ellipsize(kept[-1], "Inter-SemiBold", fs_title, w)
        title_lines = kept
    else:
        title_lines = raw_title_lines

    y = y_top
    lead_title = fs_title * 1.15
    lead_artist = fs_artist * 1.15
    lead_meta = fs_meta * 1.15

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Inter-SemiBold", fs_title)
    for line in title_lines:
        c.drawRightString(x_right, y, line)
        y -= lead_title

    c.setFont("Inter-Regular", fs_artist)
    c.drawRightString(x_right, y, artist_lines[0] if artist_lines else artist_txt)
    y -= lead_artist

    c.setFont("Inter-Light", fs_meta)
    c.setFillGray(0.2)
    c.drawRightString(x_right, y, meta_lines[0] if meta_lines else meta_txt)
    c.setFillGray(0.0)


# ============================================================
# PDF drawing
# ============================================================
ART_DROP_SHADOW = False   # set False to remove the drop shadow


def draw_album_tile_pdf(album: Album, out_pdf: Path) -> None:
    W, H = A5

    QR_NUDGE_X = 2.0
    QR_NUDGE_Y = -2.0

    c = canvas.Canvas(str(out_pdf), pagesize=A5)
    draw_template_marks(c, W, H)
    draw_thick_outer_border(c, W, H)

    card_x, card_y_top, card_w, card_h = 30, 78, 360, 442.08
    card_y = y_from_top(H, card_y_top + card_h)

    back_h = 216.0
    hinge_h = 9.84
    back_y_top = 78.0
    hinge_y_top = 294.12

    c.setStrokeColorRGB(0.95, 0.95, 0.95)
    c.setFillColorRGB(1, 1, 1)
    c.setLineWidth(0.5)
    c.rect(30.12, y_from_top(H, hinge_y_top + hinge_h), 359.76, hinge_h, stroke=1, fill=1)

    spine_raw = clean_text(f"{album.artist} · {album.title} · {album.year}").upper()
    spine_x = 40.0
    spine_right = 30.12 + 359.76 - 4.0
    spine_text = ellipsize(spine_raw, "Inter-SemiBold", 5.1, spine_right - spine_x)

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Inter-SemiBold", 5.1)
    c.drawString(spine_x, y_from_top(H, 300.85), spine_text)

    # ---------- FRONT ----------
    art_x, art_y_top, art_s = 40.0, 314.0, 196.0
    art_y = y_from_top(H, art_y_top + art_s)
    art = square_center_crop(album.artwork).resize((900, 900), Image.LANCZOS)
    if ART_DROP_SHADOW:
        shadow, pad_left_pt, pad_bottom_pt = make_art_shadow(
            art, art_s,
            offset_x_pt=4.0, offset_y_pt=6.0,
            blur_pt=6.0, spread_pt=0.0, opacity=0.09,
        )
        pt_per_px = art_s / art.width
        shadow_w_pt = shadow.width  * pt_per_px
        shadow_h_pt = shadow.height * pt_per_px
        c.drawImage(ImageReader(shadow),
                    art_x - pad_left_pt, art_y - pad_bottom_pt,
                    width=shadow_w_pt, height=shadow_h_pt, mask="auto")
    c.drawImage(ImageReader(art), art_x, art_y, width=art_s, height=art_s, mask=None)
    # Subtle inset border so white-cover art doesn't bleed into background
    _lw = 0.375
    c.setStrokeColorRGB(0xE8/255, 0xE8/255, 0xE8/255)
    c.setLineWidth(_lw)
    c.rect(art_x + _lw / 2, art_y + _lw / 2, art_s - _lw, art_s - _lw, stroke=1, fill=0)
    c.setLineWidth(1.0)  # restore default

    qr_s = 92.0
    inner_margin = art_x - card_x
    qr_x = (card_x + card_w) - inner_margin - qr_s + QR_NUDGE_X
    qr_y_top = 418.0
    qr_y = y_from_top(H, qr_y_top + qr_s) + QR_NUDGE_Y

    qr_img = make_rounded_qr_png(album.qr_url, opacity=0.8)
    c.drawImage(ImageReader(qr_img), qr_x, qr_y, width=qr_s, height=qr_s, mask="auto")

    meta_x_left = 246.0
    meta_x_right = 380.0
    meta_y_top = y_from_top(H, art_y_top)

    draw_front_metadata_block(
        c=c,
        album=album,
        x_left=meta_x_left,
        x_right=meta_x_right,
        y_top=meta_y_top - 7.5,  # offset ≈ cap height (7.33pt) so text top aligns with art top
        max_lines=7,
    )

    # ---------- BACK ----------
    back_x = card_x
    back_y = y_from_top(H, back_y_top + back_h)

    c.saveState()
    c.translate(back_x + card_w, back_y + back_h)
    c.rotate(180)

    left_margin = 10.0
    right_edge = 350.0
    gap = 10.0

    back_artist = clean_text(album.artist)
    back_title_raw = clean_text(album.title)

    c.setFont("Inter-Regular", 10.08)
    artist_w = pdfmetrics.stringWidth(back_artist, "Inter-Regular", 10.08)
    title_max_w = max(10.0, (right_edge - artist_w - gap) - left_margin)

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Inter-SemiBold", 10.08)
    back_title = ellipsize(back_title_raw, "Inter-SemiBold", 10.08, title_max_w)
    c.drawString(left_margin, back_h - 20, back_title)

    c.setFont("Inter-Regular", 10.08)
    c.drawRightString(right_edge, back_h - 20, back_artist)

    c.setFont("Inter-Light", 9.5)
    c.setFillGray(0.2)
    c.drawString(left_margin, back_h - 34, clean_text(f"{album.genre} · {album.year}"))
    c.setFillGray(0.0)

    top_y = back_h - 50
    bottom_y = 10

    gutter_between_cols = 16.0
    gutter_titles_times = 18.0

    mid = 180.0
    left_col_left = 10.0
    left_col_right = mid - gutter_between_cols / 2.0
    right_col_left = mid + gutter_between_cols / 2.0
    right_col_right = 350.0

    draw_tracklist_two_columns_autoshrink(
        c=c,
        tracks=album.tracks,
        font_title="Inter-Regular",
        font_time="Inter-Light-Tabular",
        fs_start=7.92,
        leading_factor=1.15,
        top_y=top_y,
        bottom_y=bottom_y,
        col1_left=left_col_left,
        col1_right=left_col_right,
        col2_left=right_col_left,
        col2_right=right_col_right,
        gutter_titles_times=gutter_titles_times,
    )

    c.restoreState()
    c.showPage()
    c.save()


# ============================================================
# CLI
# ============================================================
def slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-") or "album"

def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print('Usage:\n  python albumtile.py "<album.link | music.apple.com URL | id>"\n', file=sys.stderr)
        return 2

    register_fonts()

    input_arg = argv[1].strip()
    collection_id = extract_itunes_collection_id(input_arg)

    qr_url = f"https://album.link/i/{collection_id}"

    country = extract_storefront_country(input_arg)
    album = fetch_itunes_album(collection_id, qr_url=qr_url, country=country)

    out_dir = Path.cwd() / "albumtile_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = out_dir / f"{slug(album.artist)}-{slug(album.title)}-{album.year}.a5.pdf"
    draw_album_tile_pdf(album, out_pdf)

    print("Wrote:", out_pdf)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
