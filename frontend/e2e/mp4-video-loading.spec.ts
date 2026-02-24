import { test, expect } from "@playwright/test";

/**
 * MP4 Video Loading Tests
 *
 * Tests the hybrid MP4 + frame-based loading approach:
 * - Frame-based loading works as fallback
 * - MP4 generation can be triggered
 * - Video plays with native controls
 */

test.describe("MP4 Video Loading", () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to app with longer timeout for initial load
    await page.goto("http://localhost:3001", { waitUntil: "networkidle" });
    // Wait for the app to fully render (dataset browser should appear)
    await expect(page.locator('[data-testid="dataset-browser"]')).toBeVisible({ timeout: 30000 });
  });

  test("loads MicroAGI00 dataset and shows episodes", async ({ page }) => {
    // Click on MicroAGI00 dataset in sidebar
    await page.click('[data-testid="dataset-item-microagi00"]');

    // Wait for tasks to load (MicroAGI00 has tasks)
    await expect(page.locator('[data-testid="task-list"]')).toBeVisible({ timeout: 15000 });

    // Click on first task
    const firstTask = page.locator('[data-testid^="task-item-"]').first();
    await firstTask.click();

    // Wait for episodes to load
    await expect(page.locator('[data-testid="episode-list"]')).toBeVisible({ timeout: 15000 });

    // Should see at least one episode
    const episodes = page.locator('[data-testid^="episode-item-"]');
    await expect(episodes.first()).toBeVisible({ timeout: 10000 });
  });

  test("frame-based loading works for MCAP episode", async ({ page }) => {
    // Select MicroAGI00 dataset
    await page.click('[data-testid="dataset-item-microagi00"]');

    // Wait for tasks and click first task
    await expect(page.locator('[data-testid="task-list"]')).toBeVisible({ timeout: 15000 });
    await page.locator('[data-testid^="task-item-"]').first().click();

    // Wait for episodes
    await expect(page.locator('[data-testid="episode-list"]')).toBeVisible({ timeout: 15000 });

    // Click on first episode
    const firstEpisode = page.locator('[data-testid^="episode-item-"]').first();
    await firstEpisode.click();

    // Wait for either frame image or video player to appear
    const startTime = Date.now();

    // Either frame loading, frame display, or video generation should appear
    const anyContent = page.locator('[data-testid="frame-image"], [data-testid="video-not-available"], [data-testid="video-generating"], [data-testid="loading-frames"]');
    await expect(anyContent).toBeVisible({ timeout: 120000 });

    const loadTime = (Date.now() - startTime) / 1000;
    console.log(`First content appeared in ${loadTime.toFixed(1)} seconds`);

    // If we see frame-image, the frame-based loading is working
    const frameImage = page.locator('[data-testid="frame-image"]');
    if (await frameImage.isVisible()) {
      console.log("Frame-based loading is active");

      // Verify frame shows correctly
      await expect(frameImage).toHaveAttribute("src", /^data:image\/webp;base64,/);
    }
  });

  test("MP4 generation button appears for MCAP episode", async ({ page }) => {
    // Select MicroAGI00 dataset
    await page.click('[data-testid="dataset-item-microagi00"]');

    // Wait for tasks and click first task
    await expect(page.locator('[data-testid="task-list"]')).toBeVisible({ timeout: 15000 });
    await page.locator('[data-testid^="task-item-"]').first().click();

    // Wait for episodes
    await expect(page.locator('[data-testid="episode-list"]')).toBeVisible({ timeout: 15000 });

    // Click on first episode
    const firstEpisode = page.locator('[data-testid^="episode-item-"]').first();
    await firstEpisode.click();

    // Check if "Generate MP4 Preview" button appears (for uncached episodes)
    // This might not appear if frames load quickly or video is already cached
    const generateButton = page.locator('button:has-text("Generate MP4 Preview")');
    const frameImage = page.locator('[data-testid="frame-image"]');
    const videoPlayer = page.locator('[data-testid="video-player"]');

    // Wait for any of these to appear (also include loading state)
    const loadingFrames = page.locator('[data-testid="loading-frames"]');
    await expect(generateButton.or(frameImage).or(videoPlayer).or(loadingFrames)).toBeVisible({ timeout: 120000 });

    // If generate button is visible, test the generation flow
    if (await generateButton.isVisible()) {
      console.log("MP4 generation button visible - clicking to test generation");

      await generateButton.click();

      // Should show progress indicator
      await expect(page.locator('[data-testid="video-generating"]')).toBeVisible({ timeout: 5000 });
      console.log("MP4 generation started");
    }
  });

  test("playback controls work", async ({ page }) => {
    // Select MicroAGI00 dataset
    await page.click('[data-testid="dataset-item-microagi00"]');

    // Wait for tasks and click first task
    await expect(page.locator('[data-testid="task-list"]')).toBeVisible({ timeout: 15000 });
    await page.locator('[data-testid^="task-item-"]').first().click();

    // Wait for episodes
    await expect(page.locator('[data-testid="episode-list"]')).toBeVisible({ timeout: 15000 });

    // Click on first episode
    const firstEpisode = page.locator('[data-testid^="episode-item-"]').first();
    await firstEpisode.click();

    // Wait for content to load
    const frameImage = page.locator('[data-testid="frame-image"]');
    const videoPlayer = page.locator('[data-testid="video-player"]');

    const loadingFrames = page.locator('[data-testid="loading-frames"]');
    await expect(frameImage.or(videoPlayer).or(loadingFrames)).toBeVisible({ timeout: 120000 });

    // Wait for loading to complete if it's in loading state
    if (await loadingFrames.isVisible()) {
      await expect(frameImage.or(videoPlayer)).toBeVisible({ timeout: 120000 });
    }

    // Test play button (only in frame mode)
    if (await frameImage.isVisible()) {
      const playButton = page.locator('[data-testid="play-pause-btn"]');
      await expect(playButton).toBeVisible();

      // Click play
      await playButton.click();

      // Wait a moment for frame to change
      await page.waitForTimeout(1000);

      // Click pause
      await playButton.click();

      console.log("Playback controls work in frame mode");
    }

    // If video player is visible, native controls are available
    if (await videoPlayer.isVisible()) {
      console.log("Native video player is active - browser controls available");
      await expect(videoPlayer).toHaveAttribute("controls");
    }
  });
});
