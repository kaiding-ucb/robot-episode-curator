import { test, expect, Page } from "@playwright/test";

const BASE_URL = "http://localhost:3002";
const API_BASE = "http://localhost:8002/api";

async function openAnalysisModal(page: Page) {
  await page.goto(BASE_URL, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForSelector('[data-testid="open-analysis-btn"]', { timeout: 15000 });
  await page.click('[data-testid="open-analysis-btn"]');
  await page.waitForSelector('[data-testid="dataset-analysis-modal"]', { timeout: 10000 });
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

async function getTaskOptions(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const sel = document.querySelector('[data-testid="task-selector"]') as HTMLSelectElement;
    if (!sel) return [];
    return Array.from(sel.options).map((o) => o.value).filter((v) => v);
  });
}

test.describe("Signal Comparison Performance", () => {
  test("benchmark signal comparison analysis timing for RealOmni", async ({ page }) => {
    // Track SSE event timing
    const sseEvents: { type: string; time: number }[] = [];
    let sseStartTime = 0;

    page.on("response", (response) => {
      const url = response.url();
      if (url.includes("/analysis/signals")) {
        if (sseStartTime === 0) sseStartTime = Date.now();
      }
    });

    await openAnalysisModal(page);
    await selectDatasetById(page, "realomni");
    const tasksLoaded = await waitForTasksLoaded(page);
    expect(tasksLoaded, "Tasks should load for RealOmni").toBe(true);

    // Get all tasks and pick a random one
    const tasks = await getTaskOptions(page);
    expect(tasks.length, "RealOmni should have tasks").toBeGreaterThan(0);

    const randomIndex = Math.floor(Math.random() * tasks.length);
    const selectedTask = tasks[randomIndex];
    console.log(`Selected task: "${selectedTask}" (index ${randomIndex} of ${tasks.length})`);

    // Select the random task
    const taskSelector = page.locator('[data-testid="task-selector"]');
    await taskSelector.selectOption(selectedTask);
    await page.waitForTimeout(500);

    // Switch to Signal Comparison tab
    const sigTab = page.locator('[data-testid="signal-comparison-tab"]');
    const isDisabled = await sigTab.evaluate((el) =>
      el.classList.contains("cursor-not-allowed")
    );
    expect(isDisabled, "Signal comparison should be available for RealOmni").toBe(false);
    await sigTab.click();
    await page.waitForTimeout(500);

    // --- Run 1: First (uncached) analysis ---
    console.log("--- Run 1: First analysis (potentially uncached) ---");
    const run1Start = Date.now();

    // Intercept SSE messages for timing
    const episodeTimings: number[] = [];
    await page.evaluate(() => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__sseTimings = [];
    });

    await page.click('[data-testid="start-analysis-btn"]');

    // Wait for "Re-analyze" to appear (signals analysis complete)
    try {
      await page.waitForFunction(
        () => {
          const btn = document.querySelector('[data-testid="start-analysis-btn"]');
          return btn && btn.textContent?.includes("Re-analyze");
        },
        { timeout: 180000 }
      );
    } catch {
      console.log("WARNING: Analysis timed out after 180s");
    }

    const run1End = Date.now();
    const run1Duration = (run1End - run1Start) / 1000;
    console.log(`Run 1 total time: ${run1Duration.toFixed(1)}s`);

    // Verify chart rendered
    const chartExists = await page.locator('[data-testid="signal-comparison-chart"]').count();
    expect(chartExists, "Signal comparison chart should render").toBeGreaterThan(0);

    // Count episodes rendered
    const episodeCount = await page.locator('[data-testid="signal-comparison-chart"]')
      .locator(".border-b.border-gray-100")
      .count();
    console.log(`Episodes rendered: ${episodeCount}`);

    await page.screenshot({ path: "test-results/perf-run1.png" });

    // --- Run 2: Repeat same task (should benefit from caching) ---
    console.log("--- Run 2: Repeat analysis (cached) ---");
    const run2Start = Date.now();

    await page.click('[data-testid="start-analysis-btn"]');

    try {
      await page.waitForFunction(
        () => {
          const btn = document.querySelector('[data-testid="start-analysis-btn"]');
          return btn && btn.textContent?.includes("Re-analyze");
        },
        { timeout: 180000 }
      );
    } catch {
      console.log("WARNING: Repeat analysis timed out after 180s");
    }

    const run2End = Date.now();
    const run2Duration = (run2End - run2Start) / 1000;
    console.log(`Run 2 total time: ${run2Duration.toFixed(1)}s`);

    // Verify chart still rendered
    const chartExists2 = await page.locator('[data-testid="signal-comparison-chart"]').count();
    expect(chartExists2, "Signal comparison chart should render on repeat").toBeGreaterThan(0);

    await page.screenshot({ path: "test-results/perf-run2.png" });

    // --- Summary ---
    console.log("=== Performance Summary ===");
    console.log(`Task: "${selectedTask}"`);
    console.log(`Run 1 (first): ${run1Duration.toFixed(1)}s`);
    console.log(`Run 2 (repeat): ${run2Duration.toFixed(1)}s`);
    if (run1Duration > 0) {
      const speedup = run1Duration / run2Duration;
      console.log(`Speedup: ${speedup.toFixed(1)}x`);
    }
    console.log("===========================");
  });
});
