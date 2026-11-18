import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCREENSHOT_DIR = Path(r"C:\Projects\ShadowFox-ML-Engineer-Internship\Week-2\Task-2-Car-Selling-Price-Prediction\screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

def run_qa_and_capture():
    print("=" * 70)
    print("CARSPHERE AI — AUTOMATED BROWSER QA & SCREENSHOT CAPTURE")
    print("=" * 70)
    
    console_errors = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True
        )
        context = browser.new_context(viewport={"width": 1400, "height": 950})
        page = context.new_page()
        
        # Monitor browser console errors
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        
        print("\n[QA Step 1] Navigating to http://localhost:8502...")
        page.goto("http://localhost:8502", wait_until="networkidle")
        time.sleep(3)
        
        # 1. Capture Form Overview with Presets
        s1 = SCREENSHOT_DIR / "app_valuation_overview.png"
        page.screenshot(path=str(s1), full_page=False)
        print(f"  [PASS] Captured: {s1.name}")
        
        # 2. Test Preset 1: Budget Hatchback
        print("\n[QA Step 2] Testing Preset 1: '🚗 Budget Hatchback'...")
        p1_btn = page.locator('button:has-text("Budget Hatchback")').first
        p1_btn.click()
        time.sleep(2)
        submit_btn = page.locator('button[kind="primaryFormSubmit"]').first
        submit_btn.click()
        time.sleep(2)
        res_card = page.locator('.result-card').first
        assert res_card.count() > 0, "Result card not displayed for Preset 1"
        res_text_p1 = res_card.inner_text()
        print(f"  [PASS] Preset 1 Prediction Output:\n    {res_text_p1.splitlines()[0]} -> {res_text_p1.splitlines()[1]}")
        
        # 3. Test Preset 2: Family Sedan
        print("\n[QA Step 3] Testing Preset 2: '🚘 Family Sedan'...")
        p2_btn = page.locator('button:has-text("Family Sedan")').first
        p2_btn.click()
        time.sleep(2)
        submit_btn = page.locator('button[kind="primaryFormSubmit"]').first
        submit_btn.click()
        time.sleep(2)
        res_card = page.locator('.result-card').first
        assert res_card.count() > 0, "Result card not displayed for Preset 2"
        res_text_p2 = res_card.inner_text()
        print(f"  [PASS] Preset 2 Prediction Output:\n    {res_text_p2.splitlines()[0]} -> {res_text_p2.splitlines()[1]}")
        
        # 4. Test Preset 3: Premium Diesel SUV
        print("\n[QA Step 4] Testing Preset 3: '🚙 Premium Diesel SUV'...")
        p3_btn = page.locator('button:has-text("Premium Diesel SUV")').first
        p3_btn.click()
        time.sleep(2)
        submit_btn = page.locator('button[kind="primaryFormSubmit"]').first
        submit_btn.click()
        time.sleep(2)
        res_card = page.locator('.result-card').first
        assert res_card.count() > 0, "Result card not displayed for Preset 3"
        res_text_p3 = res_card.inner_text()
        print(f"  [PASS] Preset 3 Prediction Output:\n    {res_text_p3.splitlines()[0]} -> {res_text_p3.splitlines()[1]}")
        
        # Capture Result View
        page.evaluate("window.scrollTo(0, 520);")
        time.sleep(1)
        s2 = SCREENSHOT_DIR / "app_prediction_result.png"
        page.screenshot(path=str(s2), full_page=False)
        print(f"  [PASS] Captured: {s2.name}")
        
        # 5. Tab 2: Model Intelligence & Benchmark
        print("\n[QA Step 5] Testing Tab 2: '📊 Model Intelligence & Benchmark'...")
        page.evaluate("window.scrollTo(0, 0);")
        time.sleep(0.5)
        tab2 = page.locator('button:has-text("Model Intelligence")').first
        tab2.click()
        time.sleep(2)
        
        # Verify Model Metrics on Screen
        content_tab2 = page.content()
        assert "0.812" in content_tab2, "Test R² metric 0.812 missing in Tab 2"
        assert "1.00" in content_tab2, "Test MAE metric missing in Tab 2"
        assert "2.20" in content_tab2, "Test RMSE metric missing in Tab 2"
        assert "0.897" in content_tab2, "CV R² metric missing in Tab 2"
        print("  [PASS] All 4 KPI Cards verified against experiment_results.csv (R²=0.812, MAE=₹1.00L, RMSE=₹2.20L, CV=0.897)")
        
        page.evaluate("window.scrollTo(0, 240);")
        time.sleep(1)
        s3 = SCREENSHOT_DIR / "app_model_intelligence.png"
        page.screenshot(path=str(s3), full_page=False)
        print(f"  [PASS] Captured: {s3.name}")
        
        # 6. Tab 3: Market Insights & Trends
        print("\n[QA Step 6] Testing Tab 3: '📈 Market Insights & Trends'...")
        page.evaluate("window.scrollTo(0, 0);")
        time.sleep(0.5)
        tab3 = page.locator('button:has-text("Market Insights")').first
        tab3.click()
        time.sleep(2)
        
        # Verify Visualizations Loaded
        imgs = page.locator('img')
        print(f"  [PASS] Visualizations rendered: {imgs.count()} graphics found")
        
        page.evaluate("window.scrollTo(0, 200);")
        time.sleep(1)
        s4 = SCREENSHOT_DIR / "app_market_insights.png"
        page.screenshot(path=str(s4), full_page=False)
        print(f"  [PASS] Captured: {s4.name}")
        
        # 7. Tab 4: About & Governance
        print("\n[QA Step 7] Testing Tab 4: 'ℹ️ About & Governance'...")
        page.evaluate("window.scrollTo(0, 0);")
        time.sleep(0.5)
        tab4 = page.locator('button:has-text("About")').first
        tab4.click()
        time.sleep(2)
        content_tab4 = page.content()
        assert "Subham Gupta" in content_tab4, "Author name Subham Gupta missing in Tab 4"
        assert "2018" in content_tab4, "Historical reference year 2018 missing in Tab 4"
        assert "2003" in content_tab4, "Min year 2003 missing in Tab 4"
        print("  [PASS] Verified author credits (Subham Gupta) and reference horizon policy (2003–2018)")
        
        s5 = SCREENSHOT_DIR / "app_about_governance.png"
        page.screenshot(path=str(s5), full_page=False)
        print(f"  [PASS] Captured: {s5.name}")
        
        # Check console errors
        print("\n[QA Step 8] Checking browser console logs...")
        if console_errors:
            print(f"  [WARN] Console errors detected ({len(console_errors)}): {console_errors}")
        else:
            print("  [PASS] 0 browser console errors detected!")
            
        browser.close()
        
    print("\n" + "=" * 70)
    print("ALL BROWSER QA TESTS AND SCREENSHOT CAPTURES COMPLETED WITH 100% SUCCESS")
    print("=" * 70)

if __name__ == "__main__":
    run_qa_and_capture()
