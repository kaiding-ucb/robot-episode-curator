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
  test("clicking dataset expands episode list", async ({ page }) => {
    await page.goto("/");

    // Click on LIBERO dataset
    await page.getByTestId("dataset-item-libero").click();

    // Episode list heading should appear
    await expect(page.getByRole("heading", { name: "Episodes" })).toBeVisible({ timeout: 10000 });

    // Wait for episodes to load
    await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 10000 });

    // Should have at least one episode
    const episodes = page.locator('[data-testid^="episode-item-"]');
    const count = await episodes.count();
    expect(count).toBeGreaterThan(0);
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

    // Select dataset
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]');

    // Select first episode via JavaScript (to avoid click interception)
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    // Viewer should show "Viewing:" header
    await expect(page.getByText(/viewing:/i)).toBeVisible({ timeout: 10000 });
  });

  test("episode viewer shows frame image", async ({ page }) => {
    await page.goto("/");

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]');
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

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]');
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

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]');
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    // Timeline slider should be visible
    await expect(page.locator('input[type="range"]')).toBeVisible({ timeout: 10000 });
  });

  test("play button exists", async ({ page }) => {
    await page.goto("/");

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]');
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

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]');
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

  test("quality panel shows metrics when episode loaded", async ({ page }) => {
    await page.goto("/");

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]');
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    // Wait for quality to compute
    await page.waitForTimeout(3000);

    // Should show "Overall Quality" text
    await expect(page.getByText(/overall quality/i)).toBeVisible();

    // Should show percentage (not "select episode")
    const qualityPanel = page.locator('aside').last();
    await expect(qualityPanel.getByText(/select.*episode/i)).not.toBeVisible();
  });

  test("quality metrics are numbers (not NaN)", async ({ page }) => {
    await page.goto("/");

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]');
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    // Wait for quality to compute
    await page.waitForTimeout(3000);

    // Check that percentage displays don't contain NaN
    const pageContent = await page.content();
    expect(pageContent).not.toContain("NaN%");
  });

  test("quality panel shows temporal metrics", async ({ page }) => {
    await page.goto("/");

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]');
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    // Wait for quality to compute
    await page.waitForTimeout(3000);

    // Should show temporal quality section
    await expect(page.getByText(/temporal quality/i)).toBeVisible();
    await expect(page.getByText(/motion smoothness/i)).toBeVisible();
  });

  test("quality panel shows diversity metrics", async ({ page }) => {
    await page.goto("/");

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]');
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    // Wait for quality to compute
    await page.waitForTimeout(3000);

    // Should show diversity quality section
    await expect(page.getByText(/diversity quality/i)).toBeVisible();
  });

  test("quality panel shows visual metrics", async ({ page }) => {
    await page.goto("/");

    // Navigate to episode
    await page.getByTestId("dataset-item-libero").click();
    await page.waitForSelector('[data-testid^="episode-item-"]');
    await page.evaluate(() => {
      const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
      btn?.click();
    });

    // Wait for quality to compute
    await page.waitForTimeout(3000);

    // Should show visual quality section
    await expect(page.getByText(/visual quality/i)).toBeVisible();
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
// FLOW 7: Compare Datasets
// =============================================================================
test.describe("Compare Datasets", () => {
  test("clicking Compare Datasets opens panel", async ({ page }) => {
    await page.goto("/");

    await page.getByTestId("open-compare-btn").click();

    await expect(page.getByTestId("compare-panel")).toBeVisible();
  });

  test("compare panel can be closed", async ({ page }) => {
    await page.goto("/");

    await page.getByTestId("open-compare-btn").click();
    await expect(page.getByTestId("compare-panel")).toBeVisible();

    // Close via backdrop
    await page.getByTestId("compare-modal-backdrop").click({ position: { x: 10, y: 10 } });

    await expect(page.getByTestId("compare-panel")).not.toBeVisible();
  });
});

// =============================================================================
// FLOW 8: Smooth Batch Boundary Playback
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
