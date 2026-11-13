"""Capture README screenshots from the running app (http://localhost:8000, qwen backend).

Uses the locally installed Google Chrome through Playwright and only images created by this project
(bundled synthetic samples), never AI2D images (their licence does not allow redistribution).

    EXTRACTOR_BACKEND=qwen python -m uvicorn studylens.app:app --port 8000 &
    python scripts/make_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
URL = "http://localhost:8000/"
SAMPLE = "diagram_06.png"  # Web service architecture
QUESTION = "How does a request from the client reach the database?"
LONG = 15 * 60 * 1000


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        page.goto(URL)
        page.wait_for_function("document.querySelectorAll('#samples option').length > 1")
        page.select_option("#samples", SAMPLE)
        page.wait_for_function("!document.getElementById('extractBtn').disabled")
        page.click("#extractBtn")
        page.wait_for_function("document.getElementById('plabel').textContent.startsWith('Done')", timeout=LONG)
        page.locator("main#tab-ask > section").nth(0).screenshot(path=str(OUT / "ui_1_extraction.png"))
        print("extraction", flush=True)

        page.fill("#q", QUESTION)
        page.click("#askBtn")
        page.wait_for_selector("#exAnswer .extext", timeout=LONG)
        page.locator("#answerPanel").screenshot(path=str(OUT / "ui_2_answer.png"))
        print("answer", flush=True)

        # click the first flow reference (animated along graph arrows), else the first part reference
        flow = page.locator("#exAnswer button.ref.flow, #exAnswer button.ref.arrow").first
        if flow.count():
            flow.click()
            page.wait_for_timeout(4000)  # let the animation finish
            page.locator("#stage").screenshot(path=str(OUT / "ui_3_flow.png"))
            print("flow", flush=True)
        part = page.locator("#exAnswer button.ref.part").nth(1)
        if part.count():
            part.click()
            page.wait_for_timeout(600)
            page.locator("#stage").screenshot(path=str(OUT / "ui_4_spotlight.png"))
            print("spotlight", flush=True)

        page.click("#compareBox summary")
        page.click("#compareBtn")
        page.wait_for_function("!document.getElementById('pa-E1').hidden", timeout=LONG)
        page.locator("#compareBox").screenshot(path=str(OUT / "ui_5_compare.png"))
        print("compare", flush=True)

        page.click("nav .tab[data-tab='eval']")
        page.wait_for_selector("#evalBody table", timeout=60000)
        page.wait_for_timeout(800)
        page.locator("#tab-eval .panel").screenshot(path=str(OUT / "ui_6_evaluation.png"))
        print("evaluation", flush=True)
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
