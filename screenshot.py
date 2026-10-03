"""Full-page dashboard screenshot -> assets/dashboard.png (starts and stops streamlit itself)."""
import subprocess
import sys
import time
from playwright.sync_api import sync_playwright

PORT = 8602
srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                        "--server.port", str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 2600})
        for _ in range(30):  # wait for the server to accept connections
            try:
                page.goto(f"http://localhost:{PORT}")
                break
            except Exception:
                time.sleep(1)
        page.wait_for_selector('[data-testid="stApp"]')
        page.wait_for_selector(".js-plotly-plot", timeout=60000)
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="hidden", timeout=60000)
        time.sleep(5)
        assert page.locator('[data-testid="stException"]').count() == 0, "app raised an exception"
        page.screenshot(path="assets/dashboard.png", full_page=True)
        b.close()
finally:
    srv.terminate()
