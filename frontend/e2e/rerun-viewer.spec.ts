/**
 * End-to-end tests for Rerun viewer integration.
 * Tests verify the Rerun viewer toggle and RRD generation.
 */

import { test, expect } from "@playwright/test";

// =============================================================================
// Rerun Viewer API Tests
// =============================================================================
test.describe("Rerun API", () => {
  test("rerun status endpoint returns available", async ({ request }) => {
    const response = await request.get("http://localhost:8001/api/rerun/status");
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data.available).toBe(true);
    expect(data.version).toBeDefined();
  });

  test("RRD generation for LIBERO episode", async ({ request }) => {
    // Use a known LIBERO episode
    const response = await request.post(
      "http://localhost:8001/api/rerun/generate/libero_spatial%2Fpick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo%2Fdemo_0?dataset_id=libero&max_frames=10"
    );

    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data.status).toMatch(/generated|cached/);
    expect(data.rrd_url).toContain(".rrd");
    expect(data.episode_id).toBeDefined();
    expect(data.dataset_id).toBe("libero");
  });

  test("RRD file can be served", async ({ request }) => {
    // First generate an RRD
    const genResponse = await request.post(
      "http://localhost:8001/api/rerun/generate/libero_spatial%2Fpick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo%2Fdemo_0?dataset_id=libero&max_frames=10"
    );
    const genData = await genResponse.json();

    // Then fetch the file
    const fileResponse = await request.get(
      `http://localhost:8001${genData.rrd_url}`
    );

    expect(fileResponse.status()).toBe(200);
    // RRD files are binary
    const buffer = await fileResponse.body();
    expect(buffer.length).toBeGreaterThan(0);
  });
});

// =============================================================================
// Rerun Viewer UI Tests
// =============================================================================
test.describe("Rerun Viewer UI", () => {
  test("viewer mode toggle appears when episode is selected", async ({ page }) => {
    await page.goto("/");

    // Navigate to dataset -> task -> episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="task-item-"]', { timeout: 15000 });
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="task-item-"]') as HTMLElement;
      btn?.click();
    });
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 15000 });
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    // Wait for viewer to load
    await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 10000 });

    // Viewer mode toggle should be visible
    await expect(page.getByTestId("viewer-mode-standard")).toBeVisible();
    await expect(page.getByTestId("viewer-mode-rerun")).toBeVisible();
  });

  test("standard viewer is default", async ({ page }) => {
    await page.goto("/");

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="task-item-"]', { timeout: 15000 });
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="task-item-"]') as HTMLElement;
      btn?.click();
    });
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 15000 });
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    // Wait for viewer
    await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 10000 });

    // Standard button should be active (has shadow/highlight)
    const standardBtn = page.getByTestId("viewer-mode-standard");
    await expect(standardBtn).toHaveClass(/bg-white|shadow/);

    // Episode viewer (standard) should be visible
    await expect(page.getByTestId("episode-viewer")).toBeVisible();
  });

  test("clicking Rerun button switches to Rerun viewer", async ({ page }) => {
    await page.goto("/");

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="task-item-"]', { timeout: 15000 });
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="task-item-"]') as HTMLElement;
      btn?.click();
    });
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 15000 });
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    // Wait for standard viewer
    await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 10000 });

    // Click Rerun button (use JavaScript to bypass click interception)
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid="viewer-mode-rerun"]') as HTMLElement;
      btn?.click();
    });

    // Wait for Rerun viewer to load (may show loading state first)
    // The standard episode-viewer should be hidden
    await expect(page.getByTestId("episode-viewer")).not.toBeVisible({ timeout: 5000 });

    // Either loading or Rerun content should be visible
    // Look for Rerun iframe or loading indicator
    const hasRerunContent = await page.locator('iframe, [class*="rerun"], .bg-gray-900').first().isVisible().catch(() => false);
    expect(hasRerunContent).toBe(true);
  });

  test("can switch back to standard viewer", async ({ page }) => {
    await page.goto("/");

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="task-item-"]', { timeout: 15000 });
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="task-item-"]') as HTMLElement;
      btn?.click();
    });
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 15000 });
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 10000 });

    // Switch to Rerun (use JavaScript to bypass click interception)
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid="viewer-mode-rerun"]') as HTMLElement;
      btn?.click();
    });
    await page.waitForTimeout(1000);

    // Switch back to Standard
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid="viewer-mode-standard"]') as HTMLElement;
      btn?.click();
    });

    // Standard viewer should be visible again
    await expect(page.getByTestId("episode-viewer")).toBeVisible({ timeout: 5000 });
  });
});

// =============================================================================
// RRD Content Verification
// =============================================================================
test.describe("RRD Content", () => {
  test("generated RRD contains expected number of frames", async ({ request }) => {
    const response = await request.post(
      "http://localhost:8001/api/rerun/generate/libero_spatial%2Fpick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo%2Fdemo_0?dataset_id=libero&max_frames=25&force=true"
    );

    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data.num_frames).toBe(25);
    expect(data.file_size_kb).toBeGreaterThan(100); // Should be at least 100KB for 25 frames
  });

  test("cached RRD returns quickly", async ({ request }) => {
    // First request generates
    await request.post(
      "http://localhost:8001/api/rerun/generate/libero_spatial%2Fpick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo%2Fdemo_1?dataset_id=libero&max_frames=10"
    );

    // Second request should be cached
    const start = Date.now();
    const response = await request.post(
      "http://localhost:8001/api/rerun/generate/libero_spatial%2Fpick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo%2Fdemo_1?dataset_id=libero&max_frames=10"
    );
    const elapsed = Date.now() - start;

    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data.status).toBe("cached");
    expect(elapsed).toBeLessThan(500); // Cached response should be fast
  });
});
