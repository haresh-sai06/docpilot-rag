"""Headless screenshot of the React UI served by the API (python scripts/screenshot_ui.py --url http://localhost:8000)."""
import argparse
import os

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--out", default="docs/img/ui.png")
ap.add_argument("--questions", nargs="+", default=["How do I add CORS middleware to my FastAPI app?"])
a = ap.parse_args()
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.getenv("CHROME_PATH") or None, headless=True)
    page = b.new_page(viewport={"width": 1000, "height": 900})
    page.goto(a.url)
    for q in a.questions:
        page.fill("input", q)
        page.click("button")
        page.wait_for_selector(f"article h3:text-is('{q}')", timeout=300000)
    page.screenshot(path=a.out, full_page=True)
    b.close()
print("saved", a.out)
