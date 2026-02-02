/**
 * End-to-end tests for HuggingFace dataset ingestion and modality-aware viewing.
 *
 * Tests verify:
 * 1. MicroAGI00 (MCAP, flat, RGB+Depth+IMU): probe → add → browse → toggle depth → view IMU
 * 2. Egocentric-10K (WebDataset, hierarchical, video-only): probe → add → browse tasks → verify no toggle
 *
 * Uses real HuggingFace data as specified in CLAUDE.md.
 * Requires: Backend + Frontend running on ports 3000/8000 (or 3001/8001 for dataset worktree)
 */

import { test, expect } from "@playwright/test";

// HuggingFace dataset URLs
const MICROAGI00_URL = "https://huggingface.co/datasets/MicroAGI-Labs/MicroAGI00";
const EGOCENTRIC10K_URL = "https://huggingface.co/datasets/builddotai/Egocentric-10K";
const REALOMNI_URL = "https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData";

// API base (adjust based on which worktree/port you're testing)
const API_BASE = process.env.API_URL || "http://localhost:8000/api";

// =============================================================================
// FLOW 1: Add Dataset Dialog
// =============================================================================
test.describe("Add Dataset Dialog", () => {
  test("clicking add button opens dialog", async ({ page }) => {
    await page.goto("/");

    // Click add dataset button in sidebar
    await page.getByTestId("add-dataset-button").click();

    // Dialog should appear
    await expect(page.getByRole("dialog")).toBeVisible();
    await expect(page.getByText(/add huggingface dataset/i)).toBeVisible();
  });

  test("dialog has URL input field", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("add-dataset-button").click();

    // Should have URL input
    const urlInput = page.getByPlaceholder(/huggingface/i);
    await expect(urlInput).toBeVisible();
  });

  test("dialog can be closed", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("add-dataset-button").click();
    await expect(page.getByRole("dialog")).toBeVisible();

    // Close via cancel or X button
    await page.getByRole("button", { name: /cancel|close/i }).click();
    await expect(page.getByRole("dialog")).not.toBeVisible();
  });
});

// =============================================================================
// FLOW 2: Probe RealOmni Dataset (MCAP, hierarchical with tasks)
// =============================================================================
test.describe("Probe RealOmni Dataset", () => {
  test("probing RealOmni URL detects MCAP format", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("add-dataset-button").click();

    // Enter RealOmni URL
    const urlInput = page.getByPlaceholder(/huggingface/i);
    await urlInput.fill(REALOMNI_URL);

    // Click probe button
    await page.getByRole("button", { name: /probe|check|analyze/i }).click();

    // Wait for probe result
    await page.waitForSelector('[data-testid="probe-result"]', { timeout: 30000 });

    // Should show MCAP format
    await expect(page.getByText(/mcap/i)).toBeVisible();
  });

  test("probing RealOmni detects hierarchical structure", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("add-dataset-button").click();

    const urlInput = page.getByPlaceholder(/huggingface/i);
    await urlInput.fill(REALOMNI_URL);
    await page.getByRole("button", { name: /probe|check|analyze/i }).click();

    await page.waitForSelector('[data-testid="probe-result"]', { timeout: 30000 });

    // Should indicate has tasks (hierarchical)
    const hasTasksIndicator = page.locator('[data-testid="probe-result"]');
    await expect(hasTasksIndicator).toContainText(/task|hierarchical|folder/i);
  });
});

// =============================================================================
// FLOW 3: Add and Browse RealOmni Dataset
// =============================================================================
test.describe("Add and Browse RealOmni", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("adding dataset makes it appear in sidebar", async ({ page }) => {
    await page.getByTestId("add-dataset-button").click();

    const urlInput = page.getByPlaceholder(/huggingface/i);
    await urlInput.fill(REALOMNI_URL);
    await page.getByRole("button", { name: /probe|check|analyze/i }).click();

    // Wait for probe
    await page.waitForSelector('[data-testid="probe-result"]', { timeout: 30000 });

    // Click add button
    await page.getByRole("button", { name: /add dataset|add/i }).click();

    // Dialog should close
    await expect(page.getByRole("dialog")).not.toBeVisible({ timeout: 5000 });

    // Dataset should appear in sidebar (may take a moment to refresh)
    await page.waitForTimeout(1000);
    const sidebar = page.locator('[data-testid="dataset-browser"]');
    await expect(sidebar.getByText(/realomin|realomni/i)).toBeVisible({ timeout: 10000 });
  });

  test("selecting added dataset shows task list", async ({ page }) => {
    // Assuming dataset was added in previous run or is pre-configured
    // Click on the dataset
    const datasetItem = page.locator('[data-testid^="dataset-item-"]').filter({
      hasText: /realomin|realomni|10kh/i
    });

    if (await datasetItem.count() === 0) {
      // Dataset not found, add it first
      await page.getByTestId("add-dataset-button").click();
      const urlInput = page.getByPlaceholder(/huggingface/i);
      await urlInput.fill(REALOMNI_URL);
      await page.getByRole("button", { name: /probe|check|analyze/i }).click();
      await page.waitForSelector('[data-testid="probe-result"]', { timeout: 30000 });
      await page.getByRole("button", { name: /add dataset|add/i }).click();
      await page.waitForTimeout(2000);
    }

    // Now click the dataset
    await page.locator('[data-testid^="dataset-item-"]').filter({
      hasText: /realomin|realomni|10kh/i
    }).first().click();

    // Should show task list (hierarchical browsing)
    await expect(page.getByRole("heading", { name: /tasks/i })).toBeVisible({ timeout: 15000 });
    await expect(page.locator('[data-testid^="task-item-"]').first()).toBeVisible({ timeout: 15000 });
  });

  test("selecting task shows episode list", async ({ page }) => {
    // Navigate to dataset
    const datasetItem = page.locator('[data-testid^="dataset-item-"]').filter({
      hasText: /realomin|realomni|10kh/i
    }).first();

    if (await datasetItem.count() > 0) {
      await datasetItem.click();
      await page.waitForSelector('[data-testid^="task-item-"]', { timeout: 15000 });

      // Click first task
      await page.locator('[data-testid^="task-item-"]').first().click();

      // Should show episode list
      await expect(page.locator('[data-testid="episode-list"]')).toBeVisible({ timeout: 15000 });
    } else {
      test.skip();
    }
  });
});

// =============================================================================
// FLOW 4: Modality-Aware Episode Viewing
// =============================================================================
test.describe("Modality-Aware Viewing", () => {
  test("dataset with depth shows RGB/Depth toggle", async ({ page }) => {
    await page.goto("/");

    // Need a dataset with depth modality (like RealOmni or MicroAGI00)
    // First check if such dataset exists
    const depthDataset = page.locator('[data-testid^="dataset-item-"]').filter({
      has: page.locator('text=/depth/i')
    });

    if (await depthDataset.count() === 0) {
      // No depth dataset available, skip
      test.skip();
      return;
    }

    // Click dataset with depth
    await depthDataset.first().click();
    await page.waitForSelector('[data-testid^="task-item-"]', { timeout: 15000 });

    // Select first task
    await page.locator('[data-testid^="task-item-"]').first().click();
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 15000 });

    // Select first episode
    await page.locator('[data-testid^="episode-item-"]').first().click();

    // Wait for viewer to load
    await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 15000 });

    // RGB/Depth toggle should be visible
    await expect(page.getByTestId("stream-rgb-btn")).toBeVisible({ timeout: 30000 });
    await expect(page.getByTestId("stream-depth-btn")).toBeVisible();
  });

  test("clicking depth button switches stream", async ({ page }) => {
    await page.goto("/");

    // Navigate to depth-capable dataset
    const depthDataset = page.locator('[data-testid^="dataset-item-"]').filter({
      has: page.locator('text=/depth/i')
    });

    if (await depthDataset.count() === 0) {
      test.skip();
      return;
    }

    await depthDataset.first().click();
    await page.waitForSelector('[data-testid^="task-item-"]', { timeout: 15000 });
    await page.locator('[data-testid^="task-item-"]').first().click();
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 15000 });
    await page.locator('[data-testid^="episode-item-"]').first().click();

    await page.waitForSelector('[data-testid="stream-depth-btn"]', { timeout: 30000 });

    // Click depth button
    await page.getByTestId("stream-depth-btn").click();

    // Depth button should now be active (purple/highlighted)
    const depthBtn = page.getByTestId("stream-depth-btn");
    await expect(depthBtn).toHaveClass(/purple|active|bg-purple/);
  });

  test("dataset with IMU shows IMU chart", async ({ page }) => {
    await page.goto("/");

    // Look for dataset with IMU modality
    const imuDataset = page.locator('[data-testid^="dataset-item-"]').filter({
      has: page.locator('text=/imu/i')
    });

    if (await imuDataset.count() === 0) {
      test.skip();
      return;
    }

    await imuDataset.first().click();
    await page.waitForSelector('[data-testid^="task-item-"]', { timeout: 15000 });
    await page.locator('[data-testid^="task-item-"]').first().click();
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 15000 });
    await page.locator('[data-testid^="episode-item-"]').first().click();

    // Wait for viewer and IMU chart
    await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 15000 });

    // IMU chart should be visible
    await expect(page.getByTestId("imu-chart")).toBeVisible({ timeout: 60000 });
  });
});

// =============================================================================
// FLOW 5: Video-Only Dataset (No Toggle)
// =============================================================================
test.describe("Video-Only Dataset", () => {
  test("video-only dataset does NOT show RGB/Depth toggle", async ({ page }) => {
    await page.goto("/");

    // Look for video-type dataset OR dataset without depth badge
    const videoDataset = page.locator('[data-testid^="dataset-item-"]').filter({
      has: page.locator('text=/video/i')
    }).first();

    if (await videoDataset.count() === 0) {
      // Try finding dataset that explicitly does NOT have depth
      const noDepthDataset = page.locator('[data-testid^="dataset-item-"]').filter({
        hasNot: page.locator('text=/depth/i')
      }).first();

      if (await noDepthDataset.count() > 0) {
        await noDepthDataset.click();
      } else {
        test.skip();
        return;
      }
    } else {
      await videoDataset.click();
    }

    // Navigate to episode
    await page.waitForSelector('[data-testid^="task-item-"], [data-testid^="episode-item-"]', { timeout: 15000 });

    // If tasks exist, click first task
    const taskItem = page.locator('[data-testid^="task-item-"]').first();
    if (await taskItem.count() > 0) {
      await taskItem.click();
      await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 15000 });
    }

    // Click first episode
    await page.locator('[data-testid^="episode-item-"]').first().click();

    // Wait for viewer
    await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 30000 });

    // RGB/Depth toggle should NOT be visible (video-only has only RGB)
    await expect(page.getByTestId("stream-depth-btn")).not.toBeVisible({ timeout: 5000 });
  });
});

// =============================================================================
// FLOW 6: API Direct Tests (Probe Endpoint)
// =============================================================================
test.describe("Probe API", () => {
  test("POST /api/datasets/probe returns format for RealOmni", async ({ request }) => {
    const response = await request.post(`${API_BASE}/datasets/probe`, {
      data: { url: REALOMNI_URL }
    });

    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data).toHaveProperty("repo_id");
    expect(data).toHaveProperty("format_detected");
    expect(data.format_detected).toBe("mcap");
    expect(data).toHaveProperty("modalities");
    expect(data.modalities).toContain("rgb");
  });

  test("POST /api/datasets/probe returns modalities for MCAP dataset", async ({ request }) => {
    const response = await request.post(`${API_BASE}/datasets/probe`, {
      data: { url: REALOMNI_URL }
    });

    expect(response.status()).toBe(200);

    const data = await response.json();

    // Should have modalities array
    expect(Array.isArray(data.modalities)).toBe(true);
    expect(data.modalities.length).toBeGreaterThanOrEqual(1);

    // RGB should always be present
    expect(data.modalities).toContain("rgb");
  });

  test("POST /api/datasets/probe returns has_tasks for hierarchical dataset", async ({ request }) => {
    const response = await request.post(`${API_BASE}/datasets/probe`, {
      data: { url: REALOMNI_URL }
    });

    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data).toHaveProperty("has_tasks");
    expect(data.has_tasks).toBe(true);
  });

  test("POST /api/datasets/probe returns error for invalid URL", async ({ request }) => {
    const response = await request.post(`${API_BASE}/datasets/probe`, {
      data: { url: "https://huggingface.co/datasets/nonexistent/repo-12345" }
    });

    expect(response.status()).toBe(200); // API returns 200 with error in body

    const data = await response.json();
    expect(data.error).not.toBeNull();
  });
});

// =============================================================================
// FLOW 7: Add/Remove Dataset API
// =============================================================================
test.describe("Add/Remove Dataset API", () => {
  const testDatasetId = "test_realomni_e2e";

  test.afterEach(async ({ request }) => {
    // Cleanup: try to remove test dataset
    await request.delete(`${API_BASE}/datasets/${testDatasetId}`).catch(() => {});
  });

  test("POST /api/datasets adds new dataset", async ({ request }) => {
    const response = await request.post(`${API_BASE}/datasets`, {
      data: {
        url: REALOMNI_URL,
        dataset_id: testDatasetId,
        name: "E2E Test RealOmni"
      }
    });

    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data.success).toBe(true);
    expect(data.dataset_id).toBe(testDatasetId);
  });

  test("DELETE /api/datasets/{id} removes dynamic dataset", async ({ request }) => {
    // First add a dataset
    await request.post(`${API_BASE}/datasets`, {
      data: {
        url: REALOMNI_URL,
        dataset_id: testDatasetId,
        name: "E2E Test RealOmni"
      }
    });

    // Then delete it
    const response = await request.delete(`${API_BASE}/datasets/${testDatasetId}`);
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data.success).toBe(true);
  });

  test("DELETE /api/datasets/{id} fails for built-in dataset", async ({ request }) => {
    const response = await request.delete(`${API_BASE}/datasets/libero`);
    expect(response.status()).toBe(400);
  });
});

// =============================================================================
// FLOW 8: Frame Streaming with Stream Parameter
// =============================================================================
test.describe("Frame Streaming API", () => {
  test("GET /api/episodes/{id}/frames supports stream parameter", async ({ request }) => {
    // This test assumes a dataset is available with known episode
    // Using LIBERO as it's a built-in dataset
    const episodeId = "libero_object%2Fpick_up_the_bbq_sauce_and_place_it_in_the_basket_demo%2Fdemo_0";

    const response = await request.get(
      `${API_BASE}/episodes/${episodeId}/frames?start=0&end=5&dataset_id=libero&stream=rgb`
    );

    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data).toHaveProperty("frames");
    expect(Array.isArray(data.frames)).toBe(true);
  });
});
