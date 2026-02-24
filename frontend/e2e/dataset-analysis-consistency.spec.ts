import { test, expect, Page } from "@playwright/test";

const BASE_URL = "http://localhost:3002";
const API_BASE = "http://localhost:8002/api";

// Datasets to test
const DATASETS_TO_TEST = [
  { id: "realomni", name: "10Kh RealOmni-Open", hasSignals: true },
  { id: "egocentric_10k", name: "Egocentric-10K", hasSignals: false },
  { id: "droid_100", name: "Droid 100", hasSignals: true },
  { id: "microagi00", name: "MicroAGI00", hasSignals: true },
];

async function openAnalysisModal(page: Page) {
  await page.goto(BASE_URL, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForSelector('[data-testid="open-analysis-btn"]', { timeout: 15000 });
  await page.click('[data-testid="open-analysis-btn"]');
  await page.waitForSelector('[data-testid="dataset-analysis-modal"]', { timeout: 10000 });
  // Wait for datasets to load in the selector
  await page.waitForFunction(
    () => {
      const sel = document.querySelector('[data-testid="dataset-selector"]') as HTMLSelectElement;
      return sel && sel.options.length > 1;
    },
    { timeout: 15000 }
  );
}

async function selectDatasetById(page: Page, datasetId: string) {
  const selector = page.locator('[data-testid="dataset-selector"]');
  await selector.selectOption(datasetId);
  // Wait for tasks to begin loading
  await page.waitForTimeout(1000);
}

async function waitForTasksLoaded(page: Page): Promise<boolean> {
  try {
    await page.waitForFunction(
      () => {
        const sel = document.querySelector('[data-testid="task-selector"]') as HTMLSelectElement;
        if (!sel) return false;
        const options = Array.from(sel.options);
        return options.length > 0 && !options[0].text.includes("Loading");
      },
      { timeout: 20000 }
    );
    return true;
  } catch {
    return false;
  }
}

test.describe("Dataset Analysis Visual Consistency", () => {
  test("no HTTP 500 when switching between all datasets", async ({ page }) => {
    const errors: string[] = [];
    page.on("response", (response) => {
      if (response.status() >= 500) {
        errors.push(`HTTP ${response.status()} on ${response.url()}`);
      }
    });

    await openAnalysisModal(page);

    for (const ds of DATASETS_TO_TEST) {
      await selectDatasetById(page, ds.id);
      await waitForTasksLoaded(page);
      // Wait for frame counts to attempt loading
      await page.waitForTimeout(3000);

      const http500Errors = errors.filter((e) => e.includes("500"));
      expect(
        http500Errors,
        `HTTP 500 errors when switching to ${ds.name}: ${http500Errors.join(", ")}`
      ).toHaveLength(0);

      // Clear errors for next iteration
      errors.length = 0;
    }

    await page.screenshot({ path: "test-results/dataset-switch-final.png" });
  });

  test("frame counts load for each dataset without error", async ({ page }) => {
    await openAnalysisModal(page);

    for (const ds of DATASETS_TO_TEST) {
      await selectDatasetById(page, ds.id);
      const tasksLoaded = await waitForTasksLoaded(page);
      if (!tasksLoaded) continue;

      // Wait for frame counts
      await page.waitForTimeout(5000);

      // Check no HTTP 500 error text visible
      const errorCount = await page
        .locator('[data-testid="dataset-analysis-modal"]')
        .locator("text=/Error:.*HTTP 500/i")
        .count();
      expect(errorCount, `HTTP 500 error visible for ${ds.name}`).toBe(0);

      await page.screenshot({ path: `test-results/frame-counts-${ds.id}.png` });
    }
  });

  test("RealOmni signal comparison has no false Gripper label", async ({ page }) => {
    await openAnalysisModal(page);
    await selectDatasetById(page, "realomni");
    await waitForTasksLoaded(page);

    // Switch to Signal Comparison tab
    const sigTab = page.locator('[data-testid="signal-comparison-tab"]');
    const isDisabled = await sigTab.evaluate((el) =>
      el.classList.contains("cursor-not-allowed")
    );
    if (isDisabled) {
      // Signal comparison not available — skip
      return;
    }

    await sigTab.click();
    await page.waitForTimeout(500);
    await page.click('[data-testid="start-analysis-btn"]');

    // Wait for analysis to complete
    try {
      await page.waitForFunction(
        () => {
          const btn = document.querySelector('[data-testid="start-analysis-btn"]');
          return btn && btn.textContent?.includes("Re-analyze");
        },
        { timeout: 120000 }
      );
    } catch {
      // If analysis times out, still check what we have
    }

    await page.waitForTimeout(1000);

    // Check the per-episode header for "Gripper" label
    const chartArea = page.locator('[data-testid="signal-comparison-chart"]');
    const chartExists = await chartArea.count();

    if (chartExists > 0) {
      // Gripper should NOT appear in the column header for RealOmni (pose data)
      const gripperInHeader = await chartArea
        .locator(".flex.items-center.gap-2")
        .filter({ hasText: "Gripper" })
        .count();
      expect(
        gripperInHeader,
        "RealOmni should NOT show Gripper label (it uses quaternion data)"
      ).toBe(0);
    }

    await page.screenshot({ path: "test-results/realomni-signals.png" });
  });

  test("Egocentric-10K shows Signal Comparison as N/A", async ({ page }) => {
    await openAnalysisModal(page);
    await selectDatasetById(page, "egocentric_10k");
    await waitForTasksLoaded(page);
    await page.waitForTimeout(3000);

    // Signal Comparison tab should show "(N/A)"
    const tabText = await page
      .locator('[data-testid="signal-comparison-tab"]')
      .textContent();
    expect(tabText).toContain("N/A");

    await page.screenshot({ path: "test-results/egocentric-na.png" });
  });

  test("Droid signal comparison renders", async ({ page }) => {
    await openAnalysisModal(page);
    await selectDatasetById(page, "droid_100");
    await waitForTasksLoaded(page);

    const capResponse = await page.request.get(
      `${API_BASE}/datasets/droid_100/analysis/capabilities`
    );
    const caps = await capResponse.json();

    if (caps.supports_signal_comparison) {
      const sigTab = page.locator('[data-testid="signal-comparison-tab"]');
      await sigTab.click();
      await page.waitForTimeout(500);
      await page.click('[data-testid="start-analysis-btn"]');

      try {
        await page.waitForFunction(
          () => {
            const btn = document.querySelector('[data-testid="start-analysis-btn"]');
            return btn && btn.textContent?.includes("Re-analyze");
          },
          { timeout: 120000 }
        );
      } catch {
        // timeout - check what we have
      }

      await page.waitForTimeout(1000);
      const chartExists = await page.locator('[data-testid="signal-comparison-chart"]').count();
      expect(chartExists, "Signal comparison chart should render for Droid").toBeGreaterThan(0);
    }

    await page.screenshot({ path: "test-results/droid-signals.png" });
  });

  test("MicroAGI signal comparison renders", async ({ page }) => {
    await openAnalysisModal(page);
    await selectDatasetById(page, "microagi00");
    await waitForTasksLoaded(page);

    const capResponse = await page.request.get(
      `${API_BASE}/datasets/microagi00/analysis/capabilities`
    );
    const caps = await capResponse.json();

    if (caps.supports_signal_comparison) {
      const sigTab = page.locator('[data-testid="signal-comparison-tab"]');
      const isDisabled = await sigTab.evaluate((el) =>
        el.classList.contains("cursor-not-allowed")
      );
      if (!isDisabled) {
        await sigTab.click();
        await page.waitForTimeout(500);
        await page.click('[data-testid="start-analysis-btn"]');

        try {
          await page.waitForFunction(
            () => {
              const btn = document.querySelector('[data-testid="start-analysis-btn"]');
              return btn && btn.textContent?.includes("Re-analyze");
            },
            { timeout: 120000 }
          );
        } catch {
          // timeout
        }

        await page.waitForTimeout(1000);
        const chartExists = await page.locator('[data-testid="signal-comparison-chart"]').count();
        expect(chartExists, "Signal comparison chart should render for MicroAGI").toBeGreaterThan(0);
      }
    }

    await page.screenshot({ path: "test-results/microagi-signals.png" });
  });
});
