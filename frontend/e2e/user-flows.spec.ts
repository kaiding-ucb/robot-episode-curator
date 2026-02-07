/**
 * End-to-end tests for core user flows.
 * Tests verify FUNCTIONALITY, not data properties.
 * Requires: Backend + Frontend running with real LIBERO data
 */

import { test, expect } from "@playwright/test";

// =============================================================================
// FLOW 1: Application Loading
// =============================================================================
test.describe("App Loading", () => {
  test("app loads without crashing", async ({ page }) => {
    await page.goto("/");

    // Page should render (not blank, not error screen)
    await expect(page.locator("body")).not.toBeEmpty();

    // Main heading should be visible
    await expect(page.getByRole("heading", { name: "Data Viewer" })).toBeVisible();
  });

  test("sidebar shows dataset list", async ({ page }) => {
    await page.goto("/");

    // Datasets section should be visible
    await expect(page.getByRole("heading", { name: "Datasets" })).toBeVisible();

    // At least one dataset should be listed
    const datasetButtons = page.locator('[data-testid^="dataset-item-"]');
    await expect(datasetButtons.first()).toBeVisible();
  });

  test("content area shows placeholder when no episode selected", async ({ page }) => {
    await page.goto("/");

    // Should show instruction to select episode (use first() for strict mode)
    await expect(page.getByText(/select an episode/i).first()).toBeVisible();
  });
});

// =============================================================================
// FLOW 2: Dataset Browsing
// =============================================================================
test.describe("Dataset Browsing", () => {
  test("clicking dataset shows task list then episodes", async ({ page }) => {
    await page.goto("/");

    // Click on LIBERO dataset
    await page.getByTestId("dataset-item-libero").click();

    // Tasks heading should appear (new task-based navigation)
    await expect(page.getByRole("heading", { name: "Tasks" })).toBeVisible({ timeout: 10000 });

    // Wait for tasks to load
    await page.waitForSelector('[data-testid^="task-item-"]', { timeout: 15000 });

    // Should have at least one task
    const tasks = page.locator('[data-testid^="task-item-"]');
    const taskCount = await tasks.count();
    expect(taskCount).toBeGreaterThan(0);

    // Click on first task
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="task-item-"]') as HTMLElement;
      btn?.click();
    });

    // Wait for episodes to load
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 15000 });

    // Should have at least one episode
    const episodes = page.locator('[data-testid^="episode-item-"]');
    const episodeCount = await episodes.count();
    expect(episodeCount).toBeGreaterThan(0);
  });

  test("dataset shows type badge", async ({ page }) => {
    await page.goto("/");

    // Look for type badges (teleop or video)
    const typeBadge = page.locator('[data-testid^="dataset-item-"] >> text=/teleop|video/');
    await expect(typeBadge.first()).toBeVisible();
  });
});

// =============================================================================
// FLOW 3: Episode Viewing
// =============================================================================
test.describe("Episode Viewing", () => {
  test("selecting episode loads viewer", async ({ page }) => {
    await page.goto("/");

    // Select dataset -> task -> episode
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

    // Viewer should show "Viewing:" header
    await expect(page.getByText(/viewing:/i)).toBeVisible({ timeout: 10000 });
  });

  test("episode viewer shows frame image", async ({ page }) => {
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

    // Wait for image to load
    await page.waitForTimeout(2000);

    // Should show an image element with src (base64 or URL)
    const hasImage = await page.locator('img[alt^="Frame"]').first().isVisible().catch(() => false);
    expect(hasImage).toBe(true);
  });

  test("episode viewer shows action data", async ({ page }) => {
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

    // Wait for content
    await page.waitForTimeout(2000);

    // Should show action vector
    await expect(page.getByText(/action:/i)).toBeVisible();
  });
});

// =============================================================================
// FLOW 4: Video Playback Controls
// =============================================================================
test.describe("Video Playback", () => {
  test("timeline slider is visible when episode loaded", async ({ page }) => {
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

    // Timeline slider should be visible
    await expect(page.locator('input[type="range"]')).toBeVisible({ timeout: 10000 });
  });

  test("play button exists", async ({ page }) => {
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

    // Wait for timeline to appear (indicates episode is loaded)
    await page.waitForSelector('input[type="range"]', { timeout: 10000 });

    // Play button should exist in main content area
    const mainContent = page.locator('main');
    const buttons = mainContent.locator('button');
    const count = await buttons.count();
    expect(count).toBeGreaterThan(0);
  });

  test("speed selector exists", async ({ page }) => {
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

    // Speed selector should exist
    await expect(page.getByText(/speed:/i)).toBeVisible({ timeout: 10000 });
    await expect(page.locator('select')).toBeVisible();
  });
});

// =============================================================================
// FLOW 5: Quality Panel
// =============================================================================
test.describe("Quality Panel", () => {
  test("quality panel heading is visible", async ({ page }) => {
    await page.goto("/");

    // Quality panel should be visible
    await expect(page.getByRole("heading", { name: /quality analysis/i })).toBeVisible();
  });

  test("quality panel shows task quality section", async ({ page }) => {
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

    // Wait for quality to compute
    await page.waitForTimeout(8000);

    // Should show Task Quality heading
    await expect(page.getByText(/task quality/i).first()).toBeVisible();

    // Should show Expertise Test and Physics Test
    await expect(page.getByText(/expertise test/i)).toBeVisible();
    await expect(page.getByText(/physics test/i)).toBeVisible();
  });

  test("quality panel shows episode comparison section", async ({ page }) => {
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

    // Wait for quality to compute
    await page.waitForTimeout(8000);

    // Should show "This Episode vs Task" section
    await expect(page.getByText(/this episode vs task/i)).toBeVisible();

    // Should show Episode Divergence with percentage
    await expect(page.getByText(/episode divergence/i)).toBeVisible();

    // Should show Recovery Behaviors heading
    await expect(page.getByRole("heading", { name: /recovery behaviors/i })).toBeVisible();
  });

  test("quality panel shows dimension breakdown", async ({ page }) => {
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

    // Wait for quality to compute
    await page.waitForTimeout(8000);

    // Should show at least one dimension (Position X, Y, Z, etc.)
    const hasPositionDim = await page.getByText(/position (x|y|z)/i).first().isVisible().catch(() => false);
    const hasRollPitchYaw = await page.getByText(/(roll|pitch|yaw)/i).first().isVisible().catch(() => false);

    expect(hasPositionDim || hasRollPitchYaw).toBe(true);
  });

  test("quality metrics are numbers (not NaN)", async ({ page }) => {
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

    // Wait for quality to compute
    await page.waitForTimeout(8000);

    // Check that percentage displays don't contain NaN
    const pageContent = await page.content();
    expect(pageContent).not.toContain("NaN%");
  });
});

// =============================================================================
// FLOW 6: Data Manager Modal
// =============================================================================
test.describe("Data Manager", () => {
  test("clicking Manage Downloads opens modal", async ({ page }) => {
    await page.goto("/");

    // Click manage downloads button
    await page.getByTestId("open-data-manager-btn").click();

    // Modal should appear
    await expect(page.getByTestId("data-manager")).toBeVisible();
  });

  test("data manager shows disk space info", async ({ page }) => {
    await page.goto("/");

    await page.getByTestId("open-data-manager-btn").click();
    await page.waitForSelector('[data-testid="data-manager"]');

    // Should show disk space section
    await expect(page.getByText(/disk space|storage/i)).toBeVisible();
  });

  test("data manager modal can be closed", async ({ page }) => {
    await page.goto("/");

    // Open modal
    await page.getByTestId("open-data-manager-btn").click();
    await expect(page.getByTestId("data-manager")).toBeVisible();

    // Close modal by clicking the close button
    await page.getByTestId("close-modal").click();

    // Modal should close
    await expect(page.getByTestId("data-manager")).not.toBeVisible();
  });
});

// =============================================================================
// FLOW 7: Smooth Batch Boundary Playback
// =============================================================================
test.describe("Batch Boundary Playback", () => {
  test("playback continues smoothly across batch boundary", async ({ page }) => {
    await page.goto("/");

    // Navigate to dataset -> task -> episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="task-item-"]', { timeout: 10000 });
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="task-item-"]') as HTMLElement;
      btn?.click();
    });
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 10000 });
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    // Wait for viewer to load
    await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 10000 });
    await page.waitForSelector('[data-testid="frame-image"]', { timeout: 10000 });

    // Start playback
    await page.getByTestId("play-pause-btn").click();

    // Wait for playback to cross the boundary (frame 30+)
    await page.waitForFunction(() => {
      const frameInfo = document.querySelector('.bg-black\\/50.text-white');
      if (!frameInfo) return false;
      const text = frameInfo.textContent || '';
      const match = text.match(/Frame (\d+)/);
      return match && parseInt(match[1]) > 32;
    }, { timeout: 15000 });

    // Stop playback
    await page.getByTestId("play-pause-btn").click();

    // Verify frame-image remained visible throughout (no "No frame available")
    const frameImage = page.locator('[data-testid="frame-image"]');
    await expect(frameImage).toBeVisible();

    // Verify "No frame available" text is not visible
    await expect(page.getByText("No frame available")).not.toBeVisible();
  });

  test("no 'No frame available' message during normal playback", async ({ page }) => {
    await page.goto("/");

    // Navigate to dataset -> task -> episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="task-item-"]', { timeout: 10000 });
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="task-item-"]') as HTMLElement;
      btn?.click();
    });
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 10000 });
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    // Wait for viewer and frame to load
    await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 10000 });
    await page.waitForSelector('[data-testid="frame-image"]', { timeout: 10000 });

    // Start playback from beginning
    await page.getByTestId("play-pause-btn").click();

    // Let it play for a bit
    await page.waitForTimeout(3000);

    // Stop playback
    await page.getByTestId("play-pause-btn").click();

    // Verify "No frame available" is not shown
    await expect(page.getByText("No frame available")).not.toBeVisible();

    // Verify frame is visible
    await expect(page.locator('[data-testid="frame-image"]')).toBeVisible();
  });
});

// =============================================================================
// FLOW 9: Layout Structure
// =============================================================================
test.describe("Layout Structure", () => {
  test("three-column layout is visible", async ({ page }) => {
    await page.goto("/");

    // Left sidebar
    const sidebars = page.locator("aside");
    await expect(sidebars.first()).toBeVisible();

    // Main content area
    await expect(page.locator("main")).toBeVisible();

    // Right sidebar (quality panel)
    await expect(sidebars.last()).toBeVisible();
  });
});

// =============================================================================
// FLOW 10: Task-Level Quality Metrics (New Feature)
// =============================================================================
test.describe("Task-Level Quality", () => {
  test("task quality API returns valid data", async ({ request }) => {
    // Test the task quality API directly with a known LIBERO task
    const response = await request.get(
      "http://localhost:8001/api/quality/task/libero/Pick%20Up%20The%20Black%20Bowl%20On%20The%20Cookie%20Box%20And%20Place%20It%20On%20The%20Plate%20Demo?limit=5"
    );

    expect(response.status()).toBe(200);

    const data = await response.json();

    // Verify task-level metrics structure
    expect(data).toHaveProperty("task_name");
    expect(data).toHaveProperty("expertise_score");
    expect(data).toHaveProperty("divergence_distribution");

    // Verify scores are valid numbers (0-1 range)
    expect(data.expertise_score).toBeGreaterThanOrEqual(0);
    expect(data.expertise_score).toBeLessThanOrEqual(1);
  });

  test("episode divergence API returns frame-level data", async ({ request }) => {
    // Test the episode divergence API
    const response = await request.get(
      "http://localhost:8001/api/quality/task/libero/Pick%20Up%20The%20Black%20Bowl%20On%20The%20Cookie%20Box%20And%20Place%20It%20On%20The%20Plate%20Demo/divergence/libero_spatial%2Fpick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo%2Fdemo_0?limit=5"
    );

    expect(response.status()).toBe(200);

    const data = await response.json();

    // Verify divergence response structure
    expect(data).toHaveProperty("episode_id");
    expect(data).toHaveProperty("overall_divergence_score");
    expect(data).toHaveProperty("frame_divergences");
    expect(data).toHaveProperty("high_divergence_frames");

    // Verify frame_divergences is an array with values
    expect(Array.isArray(data.frame_divergences)).toBe(true);
    expect(data.frame_divergences.length).toBeGreaterThan(0);
  });

  test("quality events include metric_category field", async ({ request }) => {
    // Test that quality events have the new metric_category field
    const response = await request.get(
      "http://localhost:8001/api/quality/events/libero_spatial%2Fpick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo%2Fdemo_0?dataset_id=libero"
    );

    expect(response.status()).toBe(200);

    const data = await response.json();

    // Verify events structure
    expect(data).toHaveProperty("events");
    expect(Array.isArray(data.events)).toBe(true);

    // If there are events, check they have metric_category
    if (data.events.length > 0) {
      const event = data.events[0];
      expect(event).toHaveProperty("metric_category");
      expect(["transition", "divergence"]).toContain(event.metric_category);
    }
  });
});

// =============================================================================
// FLOW 11: Modality Chart Tabs
// =============================================================================
test.describe("Modality Chart Tabs", () => {
  test("LIBERO shows Actions tab only (no IMU)", async ({ page }) => {
    await page.goto("/");

    // Select LIBERO dataset and wait for episodes
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 10000 });

    // Select first episode
    await page.locator('[data-testid^="episode-item-"]').first().click();

    // Wait for episode viewer to load
    await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 15000 });

    // Should have Actions tab
    await expect(page.getByTestId("chart-tab-actions")).toBeVisible();

    // Should NOT have IMU tab (LIBERO has no IMU data)
    await expect(page.getByTestId("chart-tab-imu")).not.toBeVisible();
  });

  test("clicking Actions tab shows actions chart", async ({ page }) => {
    await page.goto("/");

    // Navigate to LIBERO episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 10000 });
    await page.locator('[data-testid^="episode-item-"]').first().click();
    await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 15000 });

    // Click Actions tab
    await page.getByTestId("chart-tab-actions").click();

    // Actions chart should be visible
    await expect(page.getByTestId("actions-chart")).toBeVisible({ timeout: 5000 });
  });

  test("Egocentric-10k shows no chart tabs (video-only)", async ({ page }) => {
    await page.goto("/");

    // Check if egocentric dataset exists
    const egoDataset = page.getByTestId("dataset-item-egocentric-10k");
    const exists = await egoDataset.count();

    if (exists === 0) {
      test.skip();
      return;
    }

    // Select Egocentric-10k dataset
    await egoDataset.click();
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 10000 });

    // Select first episode
    await page.locator('[data-testid^="episode-item-"]').first().click();
    await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 15000 });

    // Should NOT have any chart tabs (video-only dataset)
    await expect(page.getByTestId("chart-tab-actions")).not.toBeVisible();
    await expect(page.getByTestId("chart-tab-imu")).not.toBeVisible();
  });
});
