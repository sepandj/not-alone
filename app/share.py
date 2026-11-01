"""Share links and share cards.

A person can share their own thought as a link (/f/<id>) and as an image card.
Privacy rules:
  * Only the author can create a share (they hold an HMAC token for their
    thought that nobody else gets).
  * The words appear publicly (share page, link preview image) only if the
    author opted in AND moderation has marked the thought 'ok'. Until then
    the link shows the live count without the words.
  * Crisis thoughts can't be shared at all.
  * The story card the author downloads is rendered from their own words and
    returned only to them, so it doesn't wait for moderation.
"""
import html
import io
import os
import time
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import settings

FONT_DIR = os.path.join(os.path.dirname(__file__), "static", "fonts")
SERIF = os.path.join(FONT_DIR, "EBGaramond.ttf")
SERIF_IT = os.path.join(FONT_DIR, "EBGaramond-Italic.ttf")
TYPE = os.path.join(FONT_DIR, "CourierPrime-Regular.ttf")

PAPER = (231, 225, 211)
INK = (49, 43, 34)
MUTED = (109, 100, 83)
FAINT = (168, 157, 136)
LINE = (202, 191, 170)

SIZES = {"og": (1200, 630), "story": (1080, 1920)}

DOMAIN = settings.public_url.split("://", 1)[-1]


# ---- live counts (cached briefly: share pages can get a lot of traffic) ----
_count_cache: dict[int, tuple[float, int]] = {}
COUNT_TTL = 60


def cached_count(thought_id: int, compute) -> int:
    hit = _count_cache.get(thought_id)
    now = time.monotonic()
    if hit and now - hit[0] < COUNT_TTL:
        return hit[1]
    n = compute()
    if len(_count_cache) > 5000:
        _count_cache.clear()
    _count_cache[thought_id] = (now, n)
    return n


def headline(total: int) -> tuple[str, str]:
    """(big line, small line) for a thought felt by `total` people incl. author."""
    if total >= 2:
        return f"{total:,} people have felt this.", "you are not alone."
    return "I was the first to say this.", "do you feel it too?"


# ---- image rendering --------------------------------------------------------
@lru_cache(maxsize=4)
def _paper(size: tuple[int, int]) -> Image.Image:
    """Aged paper with faint grain, matching the site background."""
    w, h = size
    rng = np.random.default_rng(7)
    base = np.empty((h, w, 3), dtype=np.float32)
    base[:] = PAPER
    grain = rng.normal(0, 4.0, (h, w, 1)).astype(np.float32)
    # gentle vignette
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    vign = (1 - 0.06 * np.clip(d, 0, 1.4) ** 2)[..., None]
    img = np.clip((base + grain) * vign, 0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGB")


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _fit(draw, text, path, max_w, max_lines, start, smallest):
    size = start
    while size > smallest:
        f = _font(path, size)
        lines = _wrap(draw, text, f, max_w)
        if len(lines) <= max_lines:
            return f, lines
        size -= 4
    f = _font(path, smallest)
    lines = _wrap(draw, text, f, max_w)[:max_lines]
    if lines:
        lines[-1] = lines[-1].rstrip(".,;:") + " [...]"
    return f, lines


def _centered(draw, y, lines, font, fill, gap):
    w = draw.im.size[0]
    for ln in lines:
        lw = draw.textlength(ln, font=font)
        draw.text(((w - lw) / 2, y), ln, font=font, fill=fill)
        y += _lh(font) + gap
    return y


def _lh(font) -> int:
    # EB Garamond's ascent/descent are generous; a fixed ratio sets tighter.
    return int(font.size * 1.18)


def _block_height(lines, font, gap):
    return len(lines) * (_lh(font) + gap) - gap if lines else 0


@lru_cache(maxsize=256)
def render_card(words: str | None, total: int, kind: str) -> bytes:
    w, h = SIZES[kind]
    img = _paper((w, h)).copy()
    d = ImageDraw.Draw(img)
    story = kind == "story"
    margin = int(w * (0.12 if story else 0.09))
    max_w = w - 2 * margin

    big, small = headline(total)

    brand_f = _font(TYPE, 30 if story else 22)
    brand = "Y O U   A R E   N O T   A L O N E"
    head_f, head_lines = _fit(d, big, SERIF, max_w, 3, 96 if story else 64, 48 if story else 36)
    sub_f = _font(SERIF_IT, 52 if story else 34)
    sub_lines = _wrap(d, small, sub_f, max_w)

    if words:
        q = f"“{words}”"
        n = len(words)
        start = (92 if n < 40 else 76 if n < 90 else 60) if story else (54 if n < 50 else 42)
        words_f, words_lines = _fit(d, q, SERIF_IT, max_w, 9 if story else 3, start,
                                    40 if story else 30)
    else:
        words_f, words_lines = None, []

    gap_s = 8 if story else 4
    rule_gap = 70 if story else 28
    parts_h = _block_height(head_lines, head_f, gap_s) + 18 + _block_height(sub_lines, sub_f, gap_s)
    if words_lines:
        parts_h += _block_height(words_lines, words_f, gap_s) + 2 * rule_gap

    # brand at the top, domain at the bottom, message centered between
    top = int(h * (0.10 if story else 0.09))
    bw = d.textlength(brand, font=brand_f)
    d.text(((w - bw) / 2, top), brand, font=brand_f, fill=MUTED)

    y = (h - parts_h) / 2 + (20 if story else 14)
    if words_lines:
        y = _centered(d, y, words_lines, words_f, MUTED, gap_s)
        y += rule_gap - gap_s
        d.line([(w / 2 - 50, y), (w / 2 + 50, y)], fill=LINE, width=3 if story else 2)
        y += rule_gap
    y = _centered(d, y, head_lines, head_f, INK, gap_s)
    y += 18 - gap_s
    _centered(d, y, sub_lines, sub_f, MUTED, gap_s)

    foot_f = _font(TYPE, 34 if story else 24)
    fw = d.textlength(DOMAIN, font=foot_f)
    asc, desc = foot_f.getmetrics()
    fy = h - int(h * (0.09 if story else 0.08)) - asc
    d.text(((w - fw) / 2, fy), DOMAIN, font=foot_f, fill=INK)

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


# ---- share page -------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title} · Not Alone</title>
<meta name="description" content="{desc}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Not Alone">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{image}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{desc}">
<meta name="twitter:image" content="{image}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Courier+Prime:wght@400;700&display=swap">
<style>
  :root{{--bg:#e7e1d3;--ink:#312b22;--muted:#6d6453;--faint:#a89d88;--line:#cabfaa;
    --serif:"EB Garamond",Garamond,Georgia,serif;--type:"Courier Prime","Courier New",monospace;color-scheme:light}}
  @media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1b1812;--ink:#e8e1cf;--muted:#a89d86;--faint:#6b6353;--line:#352f27;color-scheme:dark}}}}
  :root[data-theme="dark"]{{--bg:#1b1812;--ink:#e8e1cf;--muted:#a89d86;--faint:#6b6353;--line:#352f27;color-scheme:dark}}
  *{{box-sizing:border-box}}
  body{{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);font-family:var(--serif);
    display:flex;align-items:center;justify-content:center;text-align:center;padding:6vh 20px;
    -webkit-font-smoothing:antialiased}}
  main{{max-width:500px;width:100%}}
  .brand{{font-family:var(--type);font-size:12px;letter-spacing:.34em;text-transform:uppercase;color:var(--muted);margin-bottom:40px}}
  .words{{font-style:italic;font-size:clamp(22px,6.5vw,27px);line-height:1.45;color:var(--muted);text-wrap:balance;margin:0 0 30px}}
  h1{{font-weight:400;font-size:clamp(30px,9vw,40px);line-height:1.18;margin:0 0 10px;text-wrap:balance}}
  .sub{{font-size:20px;color:var(--muted);margin:0}}
  .go{{display:inline-block;margin-top:40px;font-family:var(--type);font-size:15px;letter-spacing:.06em;
    color:var(--bg);background:var(--ink);border-radius:4px;padding:14px 34px;text-decoration:none}}
  .alt{{display:block;margin-top:22px;font-family:var(--type);font-size:12px;letter-spacing:.08em;color:var(--muted)}}
</style>
</head>
<body>
<main>
  <div class="brand">you are not alone</div>
  {words_html}
  <h1>{big}</h1>
  <p class="sub">{small}</p>
  <a class="go" href="/?from={sid}">{cta}</a>
  {alt}
</main>
</body>
</html>"""


def page(share_id: str, words: str | None, total: int) -> str:
    big, small = headline(total)
    url = f"{settings.public_url}/f/{share_id}"
    if words:
        title = f"“{words}”"
        desc = f"{big} {small}"
        words_html = f'<p class="words">“{html.escape(words)}”</p>'
        cta = "I feel this too"
        alt = '<a class="alt" href="/">or say what’s on your mind</a>'
    else:
        title = big
        desc = "Someone said what was on their mind and found they weren’t alone."
        words_html = ""
        cta = "say what’s on your mind"
        alt = ""
    return PAGE.format(
        title=html.escape(title), desc=html.escape(desc), url=url,
        image=f"{settings.public_url}/f/{share_id}.png?n={total}",
        words_html=words_html, big=html.escape(big), small=html.escape(small),
        sid=html.escape(share_id), cta=cta, alt=alt,
    )
