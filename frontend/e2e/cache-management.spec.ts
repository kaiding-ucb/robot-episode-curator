/**
 * End-to-end tests for frame caching functionality.
 * Tests verify cache behavior for backward scrubbing, episode switching, and cache management UI.
 * Requires: Backend + Frontend running with real LIBERO data
 */

import { test, expect } from "@playwright/test";

// Helper to select a dataset
async function selectDataset(page: import("@playwright/test").Page, datasetId: string) {
  await page.getByTestId(`dataset-item-${datasetId}`).click();
  await page.waitForSelector('[data-testid^="task-item-"]', { timeout: 15000 });
}

// Helper to select a task
async function selectTask(page: import("@playwright/test").Page) {
  await page.evaluate(() => {
    const btn = document.querySelector('[data-testid^="task-item-"]') as HTMLElement;
    btn?.click();
  });
  await page.waitForSelector('[data-testid^="episode-item-"]', { timeout: 15000 });
}

// Helper to select an episode
async function selectEpisode(page: import("@playwright/test").Page) {
  await page.evaluate(() => {
    const btn = document.querySelector('[data-testid^="episode-item-"]') as HTMLElement;
    btn?.click();
  });
  await page.waitForSelector('[data-testid="episode-viewer"]', { timeout: 10000 });
}

// Helper to wait for frame to load
async function waitForFrameLoad(page: import("@playwright/test").Page) {
  await page.waitForSelector('[data-testid="frame-image"]', { timeout: 10000 });
}

// =============================================================================
// TEST 1: Backward Scrubbing with Cached Episode
// =============================================================================
test.describe("Backward Scrubbing Cache", () => {
  test("cached episode backward scrubbing serves from cache", async ({ page }) => {
    // Navigate to episode
    await page.goto("/");
    await selectDataset(page, "libero");
    await selectTask(page);
    await selectEpisode(page);
    await waitForFrameLoad(page);

    // Play forward to load multiple batches
    await page.getByTestId("play-pause-btn").click();

    // Wait for playback to advance past frame 50
    await page.waitForFunction(
      () => {
        const frameInfo = document.querySelector(".bg-black\\/50.text-white");
        if (!frameInfo) return false;
        const text = frameInfo.textContent || "";
        const match = text.match(/Frame (\d+)/);
        return match && parseInt(match[1]) > 50;
      },
      { timeout: 20000 }
    );

    // Pause playback
    await page.getByTestId("play-pause-btn").click();

    // Now scrub backward - track network requests
    const frameRequests: { url: string; fromCache?: boolean }[] = [];

    page.on("response", async (response) => {
      if (response.url().includes("/frames")) {
        try {
          const json = await response.json();
          frameRequests.push({
            url: response.url(),
            fromCache: json.from_cache,
          });
        } catch {
          frameRequests.push({ url: response.url() });
        }
      }
    });

    // Seek to frame 10 (should be in cache)
    const slider = page.locator('input[type="range"]');
    await slider.evaluate((el: HTMLInputElement) => {
      el.value = "10";
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    });

    // Wait a bit for any network requests
    await page.waitForTimeout(2000);

    // Verify frame displays correctly
    await expect(page.locator('[data-testid="frame-image"]')).toBeVisible();

    // If there were frame requests during backward seek, they should be from cache
    const backwardRequests = frameRequests.filter((r) => r.url.includes("start=0"));
    for (const req of backwardRequests) {
      expect(req.fromCache).toBe(true);
    }
  });
});

// =============================================================================
// TEST 2: Episode Switch and Return with Cache
// =============================================================================
test.describe("Episode Switch Cache", () => {
  test("returning to cached episode loads from cache", async ({ page }) => {
    // Navigate to first episode
    await page.goto("/");
    await selectDataset(page, "libero");
    await selectTask(page);
    await selectEpisode(page);
    await waitForFrameLoad(page);

    // Let frames load
    await page.waitForTimeout(2000);

    // Get all episode buttons
    const episodeButtons = page.locator('[data-testid^="episode-item-"]');
    const count = await episodeButtons.count();

    // Skip test if only one episode
    if (count < 2) {
      test.skip();
      return;
    }

    // Click second episode
    await episodeButtons.nth(1).click();
    await waitForFrameLoad(page);
    await page.waitForTimeout(2000);

    // Track requests when returning to first episode
    const frameRequests: { url: string; fromCache?: boolean }[] = [];

    page.on("response", async (response) => {
      if (response.url().includes("/frames")) {
        try {
          const json = await response.json();
          frameRequests.push({
            url: response.url(),
            fromCache: json.from_cache,
          });
        } catch {
          frameRequests.push({ url: response.url() });
        }
      }
    });

    // Return to first episode
    await episodeButtons.first().click();
    await waitForFrameLoad(page);

    // Wait for request to complete
    await page.waitForTimeout(2000);

    // Verify frame requests came from cache
    expect(frameRequests.length).toBeGreaterThan(0);
    for (const req of frameRequests) {
      expect(req.fromCache).toBe(true);
    }
  });
});

// =============================================================================
// TEST 3: Cache Management UI
// =============================================================================
test.describe("Cache Management UI", () => {
  test("manage cached episodes via Data Manager", async ({ page }) => {
    // First, ensure at least one episode is cached by viewing it
    await page.goto("/");
    await selectDataset(page, "libero");
    await selectTask(page);
    await selectEpisode(page);
    await waitForFrameLoad(page);

    // Wait for caching to happen
    await page.waitForTimeout(3000);

    // Open Data Manager
    await page.getByTestId("open-data-manager-btn").click();
    await page.waitForSelector('[data-testid="data-manager"]');

    // Switch to Cached Episodes tab
    await page.getByTestId("cached-episodes-tab").click();

    // Wait for cached episodes to load
    await page.waitForTimeout(2000);

    // Check if there are cached episodes
    const cacheList = page.locator('[data-testid="cached-episode-list"]');
    const hasCache = await cacheList.isVisible().catch(() => false);

    if (hasCache) {
      // Verify cache size is displayed
      const cacheSize = page.locator('[data-testid="cache-total-size"]');
      await expect(cacheSize).toBeVisible();
      const cacheSizeText = await cacheSize.textContent();
      expect(cacheSizeText).toMatch(/Cache Size:/);

      // Find a delete button and click it
      const deleteBtn = page.locator('[data-testid^="delete-cache-"]').first();
      if (await deleteBtn.isVisible()) {
        await deleteBtn.click();

        // Confirm deletion dialog appears
        await expect(page.getByTestId("delete-confirm-dialog")).toBeVisible();

        // Click confirm
        await page.getByTestId("confirm-delete-button").click();

        // Wait for deletion to process
        await page.waitForTimeout(2000);

        // Verify dialog closed
        await expect(page.getByTestId("delete-confirm-dialog")).not.toBeVisible();
      }
    } else {
      // No cache to manage yet - verify empty state message
      await expect(page.getByText(/No cached episodes/)).toBeVisible();
    }
  });

  test("clear all cache button works", async ({ page }) => {
    // First, cache an episode
    await page.goto("/");
    await selectDataset(page, "libero");
    await selectTask(page);
    await selectEpisode(page);
    await waitForFrameLoad(page);
    await page.waitForTimeout(3000);

    // Open Data Manager and go to cached tab
    await page.getByTestId("open-data-manager-btn").click();
    await page.waitForSelector('[data-testid="data-manager"]');
    await page.getByTestId("cached-episodes-tab").click();
    await page.waitForTimeout(2000);

    // Check for Clear All button
    const clearAllBtn = page.getByTestId("clear-all-cache-btn");

    if (await clearAllBtn.isVisible()) {
      // Click Clear All
      await clearAllBtn.click();

      // Confirm deletion
      await expect(page.getByTestId("delete-confirm-dialog")).toBeVisible();
      await expect(page.getByText(/clear all cached episodes/i)).toBeVisible();

      await page.getByTestId("confirm-delete-button").click();

      // Wait for clearing to complete
      await page.waitForTimeout(2000);

      // Verify cache is cleared
      await expect(page.getByText(/No cached episodes/)).toBeVisible();
    }
  });
});

// =============================================================================
// TEST 4: Browser Refresh Cache Persistence
// =============================================================================
test.describe("Browser Refresh Cache Persistence", () => {
  // Run these tests serially to avoid interference from cache-clearing tests
  test.describe.configure({ mode: "serial" });
  test("cache persists after browser refresh", async ({ page }) => {
    // This test verifies that after loading an episode and refreshing,
    // subsequent requests are served from the persistent disk cache.
    //
    // Note: The first request may or may not be cached depending on
    // whether other tests have run. The key verification is that after
    // we load frames, a browser refresh still serves from cache.

    // Navigate to episode
    await page.goto("/");
    await selectDataset(page, "libero");
    await selectTask(page);
    await selectEpisode(page);
    await waitForFrameLoad(page);

    // Wait for frames to load and be cached
    await page.waitForTimeout(3000);

    // Refresh the page
    await page.reload();

    // Re-select the same episode after refresh
    await selectDataset(page, "libero");
    await selectTask(page);

    // Track requests after refresh - set up listener BEFORE selecting episode
    let postRefreshFromCache: boolean | undefined;
    page.on("response", async (response) => {
      if (response.url().includes("/frames") && postRefreshFromCache === undefined) {
        try {
          const json = await response.json();
          postRefreshFromCache = json.from_cache;
        } catch {
          // ignore
        }
      }
    });

    await selectEpisode(page);
    await waitForFrameLoad(page);

    // Wait for response
    await page.waitForTimeout(2000);

    // Verify request after refresh WAS from cache (persistent cache)
    expect(postRefreshFromCache).toBe(true);
  });

  test("cache survives browser refresh with different batch boundaries", async ({ page }) => {
    // This test verifies that full-episode caching works regardless of batch boundaries
    // Navigate to episode
    await page.goto("/");
    await selectDataset(page, "libero");
    await selectTask(page);
    await selectEpisode(page);
    await waitForFrameLoad(page);

    // Play forward to advance past first batch
    await page.getByTestId("play-pause-btn").click();
    await page.waitForTimeout(3000);
    await page.getByTestId("play-pause-btn").click(); // Pause

    // Track frame requests
    const frameRequests: { url: string; fromCache?: boolean; start?: number }[] = [];
    page.on("response", async (response) => {
      if (response.url().includes("/frames")) {
        try {
          const url = new URL(response.url());
          const start = parseInt(url.searchParams.get("start") || "0");
          const json = await response.json();
          frameRequests.push({
            url: response.url(),
            fromCache: json.from_cache,
            start,
          });
        } catch {
          frameRequests.push({ url: response.url() });
        }
      }
    });

    // Refresh the page
    await page.reload();

    // Re-navigate to same episode
    await selectDataset(page, "libero");
    await selectTask(page);
    await selectEpisode(page);
    await waitForFrameLoad(page);

    // Wait for requests
    await page.waitForTimeout(2000);

    // All requests after refresh should be from cache (full episode cached)
    expect(frameRequests.length).toBeGreaterThan(0);
    for (const req of frameRequests) {
      expect(req.fromCache).toBe(true);
    }
  });
});

// =============================================================================
// TEST 5: Cache API Endpoints
// =============================================================================
test.describe("Cache API", () => {
  test("cache stats endpoint returns valid data", async ({ request }) => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002/api";
    const response = await request.get(`${apiBase}/downloads/cache/stats`);

    expect(response.status()).toBe(200);

    const data = await response.json();

    expect(data).toHaveProperty("total_size_mb");
    expect(data).toHaveProperty("episode_count");
    expect(data).toHaveProperty("frame_cache_size_mb");
    expect(data).toHaveProperty("quality_cache_size_mb");

    expect(typeof data.total_size_mb).toBe("number");
    expect(typeof data.episode_count).toBe("number");
  });

  test("cache episodes endpoint returns list", async ({ request }) => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002/api";
    const response = await request.get(`${apiBase}/downloads/cache/episodes`);

    expect(response.status()).toBe(200);

    const data = await response.json();

    expect(Array.isArray(data)).toBe(true);

    // If there are cached episodes, verify structure
    if (data.length > 0) {
      const episode = data[0];
      expect(episode).toHaveProperty("dataset_id");
      expect(episode).toHaveProperty("episode_id");
      expect(episode).toHaveProperty("size_mb");
      expect(episode).toHaveProperty("cached_at");
      expect(episode).toHaveProperty("batch_count");
    }
  });

  test("frames endpoint returns from_cache field", async ({ request }) => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002/api";
    // First request - may or may not be cached
    const response = await request.get(
      `${apiBase}/episodes/libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo/demo_0/frames?start=0&end=10&dataset_id=libero`
    );

    expect(response.status()).toBe(200);

    const data = await response.json();

    // Verify from_cache field exists
    expect(data).toHaveProperty("from_cache");
    expect(typeof data.from_cache).toBe("boolean");

    // Second request should be cached
    const response2 = await request.get(
      `${apiBase}/episodes/libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo/demo_0/frames?start=0&end=10&dataset_id=libero`
    );

    expect(response2.status()).toBe(200);

    const data2 = await response2.json();
    expect(data2.from_cache).toBe(true);
  });
});
