r"""Check the dashboard's layout on mobile and desktop viewports.

For every page, at a phone viewport (390x844) and a desktop one (1280x900), it checks that
the page does not scroll horizontally, that no chart legend covers the plot area, and that
there are no JavaScript errors. Optionally saves full-page screenshots.

Runs in the Playwright image against a running dashboard (``plab serve``)::

    docker run --rm --network host -v "$PWD/scripts:/scripts" \
        mcr.microsoft.com/playwright/python:v1.49.0-noble \
        bash -c "pip install -q playwright==1.49.0 && \
                 python /scripts/check_layout.py http://127.0.0.1:8100"

Exits non-zero if any check fails.
"""

import argparse
import re
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

VIEWPORTS = {"mobile": (390, 844), "desktop": (1280, 900)}

# Share of each plot's area covered by its legend, in percent (null if no legend).
LEGEND_OVERLAP_JS = """() => [...document.querySelectorAll('.js-plotly-plot')].map(plot => {
    const legend = plot.querySelector('.legend'), area = plot.querySelector('.nsewdrag');
    if (!legend || !area) return null;
    const l = legend.getBoundingClientRect(), a = area.getBoundingClientRect();
    const w = Math.max(0, Math.min(l.right, a.right) - Math.max(l.left, a.left));
    const h = Math.max(0, Math.min(l.bottom, a.bottom) - Math.max(l.top, a.top));
    return Math.round(100 * w * h / (a.width * a.height));
})"""


def pages(base: str) -> list[str]:
    """Paths to check: the fixed pages plus the first run's detail page."""
    runs = urllib.request.urlopen(f"{base}/runs").read().decode()
    first_run = re.findall(r'href="(/runs/[^"#]+)"', runs)[:1]
    return ["/", "/runs", *first_run, "/status", "/about"]


def main() -> int:
    """Check every page at every viewport; print a line per check and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", nargs="?", default="http://127.0.0.1:8100")
    parser.add_argument("--screenshots", type=Path, help="directory for full-page screenshots")
    args = parser.parse_args()

    failures = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for device, (width, height) in VIEWPORTS.items():
            mobile = device == "mobile"
            context = browser.new_context(
                viewport={"width": width, "height": height}, is_mobile=mobile, has_touch=mobile
            )
            for path in pages(args.base):
                page = context.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
                page.goto(args.base + path, wait_until="networkidle")
                page.wait_for_timeout(700)
                overflow = page.evaluate(
                    "document.documentElement.scrollWidth - document.documentElement.clientWidth"
                )
                overlap = [o for o in page.evaluate(LEGEND_OVERLAP_JS) if o]
                ok = overflow <= 0 and not overlap and not errors
                failures += not ok
                print(
                    f"{'ok  ' if ok else 'FAIL'} {device:7} {path[:40]:40} overflow={overflow}px"
                    f" legend_overlap={overlap} js_errors={len(errors)}"
                )
                if args.screenshots:
                    name = re.sub(r"\W+", "_", path.strip("/")) or "overview"
                    page.screenshot(path=args.screenshots / f"{device}_{name}.png", full_page=True)
            context.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
