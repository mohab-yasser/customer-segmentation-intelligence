"""Optional browser QA: run against an already running app on localhost:8501.

Requires Playwright and its Chromium browser in the development environment.
Not needed by the deployed application or the unittest suite.
"""
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

SCREENSHOTS = Path(__file__).resolve().parents[1] / "assets" / "screenshots"


def ready(page):
    expect(page.get_by_text("CUSTOMER SEGMENTATION INTELLIGENCE  /  Historical patterns, informed experiments.", exact=True)).to_be_visible(timeout=30000)
    expect(page.get_by_test_id("stException")).to_have_count(0)


def main():
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8501")
        ready(page)
        expect(page.get_by_text("Overview", exact=True)).to_be_visible()
        expect(page.locator(".js-plotly-plot")).to_have_count(1)
        page.screenshot(path=str(SCREENSHOTS / "overview-desktop.png"), full_page=True)
        page.get_by_role("button", name="Analyze a customer", exact=False).click()
        expect(page.get_by_role("heading", name="Analyze a customer", exact=True)).to_be_visible()
        ready(page)
        page.get_by_role("button", name="Load Example Customer", exact=True).click()
        expect(page.get_by_role("spinbutton", name="Fresh", exact=True)).to_have_value("4113.00")
        page.get_by_role("button", name="Analyze Customer", exact=True).click()
        expect(page.get_by_role("heading", name="Segment 0", exact=False)).to_be_visible()
        ready(page)
        page.screenshot(path=str(SCREENSHOTS / "customer-desktop.png"), full_page=True)
        with page.expect_download() as download_info:
            page.get_by_role("button", name="Download Customer Analysis", exact=False).click()
        download = download_info.value
        assert download.suggested_filename == "customer_analysis.csv"
        page.get_by_text("What-if analysis", exact=True).click()
        expect(page.get_by_text("Spending multiplier", exact=True)).to_be_visible()
        pages = {"Segment Explorer": "Two segments. Different priorities.",
                 "Data Insights": "Explore the evidence", "PCA & Clustering": "Six dimensions, one readable map",
                 "Model Performance": "Model quality, without shortcuts", "About Project": "Creating Customer Segments"}
        for name, title in pages.items():
            page.get_by_text(name, exact=True).click()
            expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible()
            ready(page)
            if name == "Model Performance":
                page.screenshot(path=str(SCREENSHOTS / "performance-desktop.png"), full_page=True)
            print(f"PASS: {name}")
        page.get_by_text("Customer Segmentation", exact=True).click()
        expect(page.get_by_role("heading", name="Analyze a customer", exact=True)).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        # On mobile the sidebar starts collapsed after a fresh visit.
        page.reload()
        ready(page)
        page.screenshot(path=str(SCREENSHOTS / "overview-mobile.png"), full_page=True)
        # Layout overflow is checked independently of any scrollable chart/table internals.
        overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1")
        assert not overflow, "Mobile page has horizontal overflow"
        page.get_by_role("button", name="Analyze a customer", exact=False).click()
        expect(page.get_by_role("heading", name="Analyze a customer", exact=True)).to_be_visible()
        page.get_by_role("button", name="Analyze Customer", exact=True).click()
        expect(page.get_by_role("heading", name="Segment 1", exact=False)).to_be_visible()
        ready(page)
        page.screenshot(path=str(SCREENSHOTS / "customer-mobile.png"), full_page=True)
        assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1")
        assert not errors, errors
        browser.close()
        print("PASS: predictions, CSV download, all pages, mobile overflow, and browser exceptions")


if __name__ == "__main__":
    main()
