"""Discoverability: robots.txt, sitemap, home link-preview image, icons, IndexNow.

Share pages (/f/<id>) are deliberately kept out of search results (noindex):
they stay shareable, but someone's anonymous words shouldn't surface when a
person searches for a phrase or a name.
"""
import io
from datetime import date
from functools import lru_cache

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, Response
from PIL import Image, ImageDraw

from .config import settings
from . import share as s

router = APIRouter()
BASE = settings.public_url.rstrip("/")

# IndexNow key: not a secret, it only proves we control the host (served below).
INDEXNOW_KEY = "6f1d2c8b9a4e47b3a0c5e7d91f2b8a64"

ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="14" fill="#e7e1d3"/>
<circle cx="27" cy="27" r="14" fill="none" stroke="#312b22" stroke-width="5"/>
<line x1="37.5" y1="37.5" x2="51" y2="51" stroke="#312b22" stroke-width="7" stroke-linecap="round"/>
</svg>"""


@router.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return f"User-agent: *\nAllow: /\nDisallow: /api/\n\nSitemap: {BASE}/sitemap.xml\n"


@router.get("/sitemap.xml")
def sitemap():
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
           f"  <url><loc>{BASE}/</loc><lastmod>{date.today().isoformat()}</lastmod>"
           "<changefreq>daily</changefreq><priority>1.0</priority></url>\n"
           "</urlset>\n")
    return Response(xml, media_type="application/xml")


@router.get(f"/{INDEXNOW_KEY}.txt", response_class=PlainTextResponse)
def indexnow_key():
    return INDEXNOW_KEY


@lru_cache(maxsize=1)
def _home_card() -> bytes:
    w, h = s.SIZES["og"]
    img = s._paper((w, h)).copy()
    d = ImageDraw.Draw(img)
    max_w = w - 2 * int(w * 0.09)
    brand_f = s._font(s.TYPE, 22)
    brand = "Y O U   A R E   N O T   A L O N E"
    head_f, head_lines = s._fit(d, "What's on your mind?", s.SERIF, max_w, 2, 76, 48)
    sub_f = s._font(s.SERIF_IT, 36)
    sub_lines = s._wrap(d, "say it, and see how many others feel it too.", sub_f, max_w)
    bw = d.textlength(brand, font=brand_f)
    d.text(((w - bw) / 2, int(h * 0.09)), brand, font=brand_f, fill=s.MUTED)
    block = s._block_height(head_lines, head_f, 4) + 22 + s._block_height(sub_lines, sub_f, 4)
    y = (h - block) / 2 + 10
    y = s._centered(d, y, head_lines, head_f, s.INK, 4)
    s._centered(d, y + 22 - 4, sub_lines, sub_f, s.MUTED, 4)
    foot_f = s._font(s.TYPE, 24)
    fw = d.textlength(s.DOMAIN, font=foot_f)
    d.text(((w - fw) / 2, h - int(h * 0.08) - foot_f.getmetrics()[0]), s.DOMAIN,
           font=foot_f, fill=s.INK)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85, optimize=True, progressive=True)
    return buf.getvalue()


@router.get("/og.jpg")
def og_image():
    return Response(_home_card(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.get("/icon.svg")
def icon_svg():
    return Response(ICON_SVG, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=604800"})


@lru_cache(maxsize=4)
def _icon_png(size: int) -> Image.Image:
    """Raster version of ICON_SVG (paper tile + magnifier), drawn at 4x then reduced."""
    k = size * 4 / 64
    img = Image.new("RGBA", (size * 4, size * 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size * 4 - 1, size * 4 - 1], radius=int(14 * k), fill=s.PAPER)
    d.ellipse([13 * k, 13 * k, 41 * k, 41 * k], outline=s.INK, width=max(1, int(5 * k)))
    d.line([(37.5 * k, 37.5 * k), (51 * k, 51 * k)], fill=s.INK, width=max(1, int(7 * k)))
    r = 3.5 * k
    d.ellipse([51 * k - r, 51 * k - r, 51 * k + r, 51 * k + r], fill=s.INK)
    return img.resize((size, size), Image.LANCZOS)


@router.get("/favicon.ico")
def favicon():
    buf = io.BytesIO()
    _icon_png(48).save(buf, "ICO", sizes=[(16, 16), (32, 32), (48, 48)])
    return Response(buf.getvalue(), media_type="image/x-icon",
                    headers={"Cache-Control": "public, max-age=604800"})


@router.get("/apple-touch-icon.png")
def apple_icon():
    buf = io.BytesIO()
    _icon_png(180).convert("RGB").save(buf, "PNG", optimize=True)
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=604800"})
