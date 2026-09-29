r"""Check the dashboard's layout on mobile and desktop viewports.

For every page, at a phone viewport (390x844) and a desktop one (1280x900), it checks that
the page does not scroll horizontally, that no chart legend covers the plot area, and that
there are no JavaScript errors. On the phone it also taps a data point on the first chart
and checks that the hover label appears and then goes away by itself. Optionally saves
full-page screenshots.

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

# Page coordinates of the middle point of the first chart's first trace (null if no chart).
FIRST_POINT_JS = """() => {
    const el = document.querySelector('.js-plotly-plot');
    if (!el || !el._fullData.length) return null;
    const trace = el._fullData[0], xa = el._fullLayout.xaxis, ya = el._fullLayout.yaxis;
    const k = Math.floor(trace.x.length / 2), box = el.getBoundingClientRect();
    return {
        x: box.left + xa._offset + xa.l2p(xa.d2l(trace.x[k])),
        y: box.top + ya._offset + ya.l2p(ya.d2l(trace.y[k])),
    };
}"""
HOVER_LABELS_JS = "() => document.querySelectorAll('.hoverlayer .hovertext').length"
#: Longer than the page's auto-hide delay for touch hover labels.
HOVER_WAIT_MS = 3500


def hover_clears(page) -> str:
    """Tap a data point; return ``ok``, ``no-label`` or ``stuck`` (``-`` without a chart)."""
    point = page.evaluate(FIRST_POINT_JS)
    if point is None or point["y"] > page.viewport_size["height"]:
        page.evaluate("p => window.scrollTo(0, p.y - 200)", point) if point else None
        point = page.evaluate(FIRST_POINT_JS)
    if point is None:
        return "-"
    page.touchscreen.tap(point["x"], point["y"])
    page.wait_for_timeout(300)
    if not page.evaluate(HOVER_LABELS_JS):
        return "no-label"
    page.wait_for_timeout(HOVER_WAIT_MS)
    return "stuck" if page.evaluate(HOVER_LABELS_JS) else "ok"


def pages(base: str) -> list[str]:
    """Paths to check: the fixed pages plus each strategy configuration's latest run.

    Run pages differ by strategy (explanations, worked-example columns), so all are checked.
    """
    runs = urllib.request.urlopen(f"{base}/runs").read().decode()
    run_pages = list(dict.fromkeys(re.findall(r'href="(/runs/[^"#]+)"', runs)))
    return ["/", "/runs", *run_pages, "/signals", "/context", "/status", "/about"]


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
                hover = hover_clears(page) if mobile else "-"
                ok = overflow <= 0 and not overlap and not errors and hover in ("ok", "-")
                failures += not ok
                print(
                    f"{'ok  ' if ok else 'FAIL'} {device:7} {path[:40]:40} overflow={overflow}px"
                    f" legend_overlap={overlap} js_errors={len(errors)} touch_hover={hover}"
                )
                if args.screenshots:
                    name = re.sub(r"\W+", "_", path.strip("/")) or "overview"
                    page.screenshot(path=args.screenshots / f"{device}_{name}.png", full_page=True)
            context.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
