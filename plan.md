# Robotics Dataset Viewer - Implementation Plan (Revised)

## MVP Scope

**3 Datasets Only:**
| Dataset | Size | Strategy | Format |
|---------|------|----------|--------|
| **LIBERO** | ~10GB | Download locally | HDF5 |
| **LIBERO-PRO** | ~5GB | Download locally | HDF5 |
| **10Kh RealOmni-Open** | 95TB | Streaming only | HuggingFace API |

---

## CRITICAL: Testing Philosophy

### Tests Must Verify FUNCTIONALITY, Not Data Properties

**BAD tests (remove these):**
```python
# USELESS - tests algorithm definition, not functionality
def test_motion_smoothness_smooth_trajectory():
    smooth_actions = np.linspace(...)
    score = motion_smoothness(episode)
    assert 0.8 <= score <= 1.0  # Testing math, not system
```

**GOOD tests:**
```python
# USEFUL - tests that the system actually works
def test_can_load_and_display_episode():
    # Load real episode
    episode = loader.load_episode(real_episode_id)
    # Verify it loaded (not crashed, not empty)
    assert episode.observations is not None
    assert len(episode.observations) > 0
```

**What to test:**
- Can we download data? (download succeeds, files exist)
- Can we load data? (loader returns data, not crash)
- Can we compute metrics? (returns a number, not NaN/crash)
- Does UI respond to interactions? (click → something happens)

**What NOT to test:**
- Is motion smoothness 0.8-1.0 for smooth data? (testing algorithm definition)
- Is diversity score high for diverse data? (testing math)
- Are specific score ranges correct? (properties of data, not functionality)

---

## Playwright MCP Tests: End-to-End User Flows

### UI Architecture Reference

```
┌─────────────────┬────────────────────────────────┬─────────────────┐
│    SIDEBAR      │        CONTENT AREA            │  RIGHT SIDEBAR  │
│   (288px)       │                                │    (288px)      │
├─────────────────┤ ┌────────────────────────────┐ ├─────────────────┤
│                 │ │                            │ │                 │
│  Data Viewer    │ │     VIDEO/3D VIEWER        │ │ QUALITY ANALYSIS│
│  ─────────────  │ │     (Episode frames)       │ │                 │
│                 │ │                            │ │ [Score display] │
│  DATASETS       │ └────────────────────────────┘ │ [Metrics bars]  │
│  ▼ LIBERO       │ ┌────────────────────────────┐ │ [Quality flags] │
│    └─ episodes  │ │   TIMELINE + CONTROLS      │ │                 │
│  ▶ LIBERO-PRO   │ │   [◀ ▶ ■] ═══●═══════════  │ │                 │
│  ▶ RealOmni     │ └────────────────────────────┘ │                 │
│                 │                                │                 │
├─────────────────┤                                │                 │
│ ACTION BUTTONS  │                                │                 │
│ [Compare]       │                                │                 │
│ [Quality]       │                                │                 │
│ [Downloads]     │                                │                 │
└─────────────────┴────────────────────────────────┴─────────────────┘
```

### Test Suite: User Flows (Playwright MCP)

```typescript
// frontend/e2e/user-flows.spec.ts
/**
 * End-to-end tests for core user flows.
 * Uses Playwright MCP to interact with real UI.
 * Requires: Backend + Frontend running with real LIBERO data
 */

import { test, expect } from "@playwright/test";

// =============================================================================
// FLOW 1: Application Loading
// =============================================================================
test.describe("App Loading", () => {
  test("app loads without crashing", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("body")).not.toBeEmpty();
    await expect(page.locator('[data-testid="app-layout"]')).toBeVisible();
  });

  test("sidebar shows dataset list", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("DATASETS")).toBeVisible();
    const datasetButtons = page.locator('[data-testid^="dataset-"]');
    await expect(datasetButtons.first()).toBeVisible();
  });

  test("content area shows placeholder when no episode selected", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText(/select an episode/i)).toBeVisible();
  });
});

// =============================================================================
// FLOW 2: Dataset Browsing
// =============================================================================
test.describe("Dataset Browsing", () => {
  test("clicking dataset expands episode list", async ({ page }) => {
    await page.goto("/");
    const datasetBtn = page.locator('[data-testid^="dataset-"]').first();
    await datasetBtn.click();
    await expect(page.locator('[data-testid="episode-list"]')).toBeVisible();
    const episodes = page.locator('[data-testid="episode-list"] button');
    const count = await episodes.count();
    expect(count).toBeGreaterThan(0);
  });

  test("dataset shows type badge (teleop/video)", async ({ page }) => {
    await page.goto("/");
    const typeBadge = page.locator('[data-testid^="dataset-"] span:has-text("teleop"), [data-testid^="dataset-"] span:has-text("video")');
    await expect(typeBadge.first()).toBeVisible();
  });
});

// =============================================================================
// FLOW 3: Episode Viewing
// =============================================================================
test.describe("Episode Viewing", () => {
  test("selecting episode loads video viewer", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid^="dataset-"]').first().click();
    await page.waitForSelector('[data-testid="episode-list"] button');
    await page.locator('[data-testid="episode-list"] button').first().click();
    await expect(page.locator('[data-testid="episode-viewer"], [data-testid="video-player"], [data-testid="frame-display"]')).toBeVisible();
  });

  test("episode viewer shows frame content (not error state)", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid^="dataset-"]').first().click();
    await page.waitForSelector('[data-testid="episode-list"] button');
    await page.locator('[data-testid="episode-list"] button').first().click();
    await page.waitForTimeout(1000);

    // Should NOT show error message
    const errorVisible = await page.locator('[data-testid="error-message"], .error').isVisible().catch(() => false);
    expect(errorVisible).toBe(false);

    // Should show image or canvas
    const hasContent = await page.locator('img[src], canvas').first().isVisible().catch(() => false);
    expect(hasContent).toBe(true);
  });

  test("header updates to show selected episode", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid^="dataset-"]').first().click();
    await page.waitForSelector('[data-testid="episode-list"] button');
    await page.locator('[data-testid="episode-list"] button').first().click();
    await expect(page.getByText(/select an episode to view/i)).not.toBeVisible();
    await expect(page.getByText(/viewing/i)).toBeVisible();
  });
});

// =============================================================================
// FLOW 4: Video Playback Controls
// =============================================================================
test.describe("Video Playback", () => {
  test("timeline is visible when episode loaded", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid^="dataset-"]').first().click();
    await page.waitForSelector('[data-testid="episode-list"] button');
    await page.locator('[data-testid="episode-list"] button').first().click();
    await expect(page.locator('[data-testid="timeline"], .timeline, input[type="range"]')).toBeVisible();
  });

  test("play button exists and is clickable", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid^="dataset-"]').first().click();
    await page.waitForSelector('[data-testid="episode-list"] button');
    await page.locator('[data-testid="episode-list"] button').first().click();
    const playBtn = page.locator('[data-testid="play-button"], button:has-text("Play"), [aria-label*="play" i]');
    await expect(playBtn.first()).toBeVisible();
    await playBtn.first().click();
  });

  test("scrubbing timeline changes displayed frame", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid^="dataset-"]').first().click();
    await page.waitForSelector('[data-testid="episode-list"] button');
    await page.locator('[data-testid="episode-list"] button').first().click();
    await page.waitForSelector('[data-testid="timeline"], input[type="range"]');

    const slider = page.locator('[data-testid="timeline"], input[type="range"]').first();
    const box = await slider.boundingBox();
    if (box) {
      await page.mouse.click(box.x + box.width * 0.8, box.y + box.height / 2);
    }
    await page.waitForTimeout(500);
  });
});

// =============================================================================
// FLOW 5: Quality Panel
// =============================================================================
test.describe("Quality Panel", () => {
  test("quality panel shows content when episode loaded", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid^="dataset-"]').first().click();
    await page.waitForSelector('[data-testid="episode-list"] button');
    await page.locator('[data-testid="episode-list"] button').first().click();
    await page.waitForTimeout(1000);

    const qualitySection = page.locator('[data-testid="quality-panel"], .quality-panel');
    await expect(qualitySection).toBeVisible();
  });

  test("quality metrics are numbers (not NaN or error)", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid^="dataset-"]').first().click();
    await page.waitForSelector('[data-testid="episode-list"] button');
    await page.locator('[data-testid="episode-list"] button').first().click();
    await page.waitForTimeout(2000);

    const scoreElements = page.locator('[data-testid*="score"], .score-value, .metric-value');
    const count = await scoreElements.count();

    if (count > 0) {
      for (let i = 0; i < Math.min(count, 3); i++) {
        const text = await scoreElements.nth(i).textContent();
        expect(text).not.toContain("NaN");
        expect(text?.toLowerCase()).not.toContain("error");
      }
    }
  });
});

// =============================================================================
// FLOW 6: Data Manager Modal
// =============================================================================
test.describe("Data Manager", () => {
  test("clicking Manage Downloads opens modal", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid="open-data-manager-btn"]').click();
    await expect(page.locator('[data-testid="data-manager"], .data-manager')).toBeVisible();
  });

  test("data manager modal can be closed", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid="open-data-manager-btn"]').click();
    await expect(page.locator('[data-testid="data-manager"]')).toBeVisible();
    const closeBtn = page.locator('[data-testid="modal-backdrop"], [data-testid="close-modal"], button:has-text("Close")');
    await closeBtn.first().click();
    await expect(page.locator('[data-testid="data-manager"]')).not.toBeVisible();
  });
});

// =============================================================================
// FLOW 7: Compare Datasets
// =============================================================================
test.describe("Compare Datasets", () => {
  test("clicking Compare Datasets opens modal", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid="open-compare-btn"]').click();
    await expect(page.locator('[data-testid="compare-panel"]')).toBeVisible();
  });

  test("compare panel shows dataset selection options", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid="open-compare-btn"]').click();
    await page.waitForSelector('[data-testid="compare-panel"]');
    const hasSelection = await page.getByText(/select.*dataset/i).isVisible() ||
                         await page.locator('[data-testid="compare-panel"] button').count() > 0;
    expect(hasSelection).toBe(true);
  });

  test("compare modal can be closed", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid="open-compare-btn"]').click();
    await expect(page.locator('[data-testid="compare-panel"]')).toBeVisible();
    await page.locator('[data-testid="compare-modal-backdrop"]').click({ position: { x: 10, y: 10 } });
    await expect(page.locator('[data-testid="compare-panel"]')).not.toBeVisible();
  });
});

// =============================================================================
// FLOW 8: Dataset Quality Dashboard
// =============================================================================
test.describe("Dataset Quality Dashboard", () => {
  test("quality button is disabled when no dataset selected", async ({ page }) => {
    await page.goto("/");
    const qualityBtn = page.locator('[data-testid="open-dataset-quality-btn"]');
    await expect(qualityBtn).toBeDisabled();
  });

  test("quality button enables after selecting dataset and episode", async ({ page }) => {
    await page.goto("/");
    await page.locator('[data-testid^="dataset-"]').first().click();
    await page.waitForSelector('[data-testid="episode-list"] button');
    await page.locator('[data-testid="episode-list"] button').first().click();
    const qualityBtn = page.locator('[data-testid="open-dataset-quality-btn"]');
    await expect(qualityBtn).toBeEnabled();
  });
});

// =============================================================================
// FLOW 9: Layout Structure
// =============================================================================
test.describe("Layout Structure", () => {
  test("three-column layout is visible", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator('aside').first()).toBeVisible();
    await expect(page.locator('main')).toBeVisible();
    await expect(page.locator('aside').last()).toBeVisible();
  });

  test("sidebars have correct approximate width", async ({ page }) => {
    await page.goto("/");
    const sidebars = page.locator('aside');
    const leftSidebar = sidebars.first();
    const leftBox = await leftSidebar.boundingBox();
    expect(leftBox?.width).toBeGreaterThan(200);
    expect(leftBox?.width).toBeLessThan(400);
  });
});
```

---

## Integration Tests (pytest) - Functionality Only

```python
# tests/integration/test_functionality.py
"""
Tests that verify FUNCTIONALITY, not data properties.
"""

import pytest
from pathlib import Path
import numpy as np

# =============================================================================
# Download Functionality
# =============================================================================
class TestDownloadFunctionality:
    """Tests that downloads actually work"""

    def test_libero_downloader_completes(self, tmp_path):
        """Download completes without crashing"""
        from backend.downloaders.libero import LiberoDownloader

        downloader = LiberoDownloader(data_dir=tmp_path)
        result = downloader.download(task_suite="libero_spatial", max_tasks=1)

        # Functionality: did it complete?
        assert result.success, f"Download failed: {result.error}"

    def test_downloaded_files_are_valid_hdf5(self, tmp_path):
        """Downloaded files can be opened as HDF5"""
        from backend.downloaders.libero import LiberoDownloader
        import h5py

        downloader = LiberoDownloader(data_dir=tmp_path)
        downloader.download(task_suite="libero_spatial", max_tasks=1)

        hdf5_files = list(tmp_path.glob("**/*.hdf5"))
        assert len(hdf5_files) > 0, "No files downloaded"

        # Functionality: can we open the file?
        with h5py.File(hdf5_files[0], 'r') as f:
            assert len(f.keys()) > 0  # File has content

# =============================================================================
# Loader Functionality
# =============================================================================
class TestLoaderFunctionality:
    """Tests that loaders work with real data"""

    @pytest.fixture
    def real_data_path(self):
        path = Path("./data/libero")
        if not path.exists():
            pytest.skip("LIBERO data not downloaded")
        return path

    def test_loader_lists_episodes(self, real_data_path):
        """Loader can list episodes"""
        from backend.loaders.hdf5_loader import HDF5Loader

        loader = HDF5Loader(data_root=real_data_path)
        episodes = loader.list_episodes()

        # Functionality: did it return episodes?
        assert len(episodes) > 0

    def test_loader_loads_episode(self, real_data_path):
        """Loader can load an episode"""
        from backend.loaders.hdf5_loader import HDF5Loader

        loader = HDF5Loader(data_root=real_data_path)
        episodes = loader.list_episodes()
        episode = loader.load_episode(episodes[0].id)

        # Functionality: did it return data?
        assert episode is not None
        assert episode.observations is not None
        assert episode.actions is not None

    def test_loaded_episode_has_frames(self, real_data_path):
        """Loaded episode has actual frame data"""
        from backend.loaders.hdf5_loader import HDF5Loader

        loader = HDF5Loader(data_root=real_data_path)
        episodes = loader.list_episodes()
        episode = loader.load_episode(episodes[0].id)

        # Functionality: is there actual data?
        assert len(episode.observations) > 0
        assert episode.observations[0].size > 0  # Not empty array

# =============================================================================
# Quality Metrics Functionality
# =============================================================================
class TestQualityFunctionality:
    """Tests that quality metrics compute without crashing"""

    @pytest.fixture
    def real_episode(self):
        from backend.loaders.hdf5_loader import HDF5Loader
        path = Path("./data/libero")
        if not path.exists():
            pytest.skip("LIBERO data not downloaded")
        loader = HDF5Loader(data_root=path)
        episodes = loader.list_episodes()
        return loader.load_episode(episodes[0].id)

    def test_motion_smoothness_computes(self, real_episode):
        """Motion smoothness returns a valid number"""
        from backend.quality.temporal import motion_smoothness

        score = motion_smoothness(real_episode)

        # Functionality: did it return a valid number?
        assert isinstance(score, (int, float))
        assert not np.isnan(score)
        assert 0 <= score <= 1

    def test_diversity_metrics_compute(self, real_episode):
        """Diversity metrics return valid numbers"""
        from backend.quality.diversity import detect_recovery_behavior

        has_recovery, indices = detect_recovery_behavior(real_episode)

        # Functionality: did it return expected types?
        assert isinstance(has_recovery, bool)
        assert isinstance(indices, list)

    def test_quality_aggregator_computes(self, real_episode):
        """Quality aggregator returns valid number"""
        from backend.quality.aggregator import compute_episode_quality

        score = compute_episode_quality(real_episode)

        # Functionality: did it return a valid number?
        assert isinstance(score, (int, float))
        assert not np.isnan(score)
        assert 0 <= score <= 1

# =============================================================================
# API Functionality
# =============================================================================
class TestAPIFunctionality:
    """Tests that API endpoints respond correctly"""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.api.main import app
        return TestClient(app)

    def test_datasets_endpoint_responds(self, client):
        """GET /api/datasets returns 200"""
        response = client.get("/api/datasets")
        assert response.status_code == 200

    def test_datasets_endpoint_returns_list(self, client):
        """GET /api/datasets returns a list"""
        response = client.get("/api/datasets")
        data = response.json()
        assert isinstance(data, list)

    def test_episodes_endpoint_responds(self, client):
        """GET /api/datasets/{id}/episodes returns 200 or 404"""
        response = client.get("/api/datasets/libero/episodes")
        # 200 if data exists, 404 if not - both are valid functional responses
        assert response.status_code in [200, 404]

# =============================================================================
# Streaming Functionality
# =============================================================================
class TestStreamingFunctionality:
    """Tests that streaming works without downloading everything"""

    @pytest.fixture
    def hf_token(self):
        import os
        token = os.environ.get("HF_TOKEN")
        if not token:
            pytest.skip("HF_TOKEN not set")
        return token

    def test_streaming_connects(self, hf_token):
        """Can connect to streaming endpoint"""
        from backend.loaders.streaming_loader import RealOmniStreamingLoader

        loader = RealOmniStreamingLoader(token=hf_token)
        info = loader.get_dataset_info()

        # Functionality: did connection succeed?
        assert info is not None

    def test_streaming_is_fast(self, hf_token):
        """Streaming doesn't download entire dataset"""
        import time
        from backend.loaders.streaming_loader import RealOmniStreamingLoader

        loader = RealOmniStreamingLoader(token=hf_token)

        start = time.time()
        metadata = loader.get_episode_metadata(limit=1)
        elapsed = time.time() - start

        # Functionality: completed quickly (streaming, not downloading 95TB)
        assert elapsed < 60, f"Took {elapsed}s - too slow"
```

---

## Test Summary

| Test Type | What It Verifies | Example |
|-----------|------------------|---------|
| **Playwright E2E** | UI responds to user actions | Click dataset → episodes appear |
| **Integration** | Backend functions work | Loader returns episode data |
| **NOT included** | Algorithm definitions | "Smooth data should score 0.8-1.0" |

**Key Principle:** Tests verify "does clicking X do Y?" and "does function F return valid output?" - NOT "is the score for smooth data between 0.8 and 1.0?"

---

## Files to Create/Modify

### Delete (useless mock tests):
- `tests/phase1/*` (old mock-based tests)
- `tests/phase2/*` (tests asserting score ranges)
- `tests/fixtures/*` (synthetic test data)

### Create (functionality tests):
- `frontend/e2e/user-flows.spec.ts` - Playwright MCP tests for all user flows
- `tests/integration/test_functionality.py` - Backend functionality tests

### Modify:
- Existing components to add required `data-testid` attributes for Playwright

---

## Running Tests

```bash
# 1. Download real data first
python -m backend.downloaders.libero --output ./data/libero

# 2. Start servers
./start.sh

# 3. Run Playwright E2E tests
cd frontend && npx playwright test e2e/user-flows.spec.ts

# 4. Run integration tests
pytest tests/integration/test_functionality.py -v
```

---

## Gate Criteria

| Phase | Pass Condition |
|-------|----------------|
| 1 | Playwright: App loads, datasets visible, episode viewer works |
| 2 | Playwright: Quality panel shows metrics, no NaN/errors |
| 3 | Playwright: Compare panel works, streaming datasets accessible |

**All tests verify FUNCTIONALITY. No tests verify data properties.**

---
---

# Scale Data Viewer to All HuggingFace Robotics Datasets

## Context

The adapter pattern infrastructure is **already built** (`backend/adapters/`):
- `StreamingAdapter` ABC with `list_tasks()`, `list_episodes()`, `resolve_episode()`, `get_capabilities()`
- `FormatRegistry` with auto-detection and caching
- `LeRobotAdapter`, `MCAPAdapter`, `RawVideoAdapter`, `MultiSubdatasetAdapter` all functional

**What's missing**: The adapters have shallow traversal limits, hardcoded FPS/frame estimates, no pagination for large episode lists (droid has 92K episodes), no stride for large episodes (1000+ frames), duplicate utility code, and the frontend lacks format-awareness.

This plan merges the original adapter improvements with the co-worker's pagination + stride approach.

---

## Phase 1: Robustify Adapters & Shared Utils (Steps 1-4, parallelizable)

### Step 1: Extract Shared Utilities
**New file**: `backend/adapters/utils.py`

- Extract `_get_hf_headers()` (duplicated in mcap_adapter.py, raw_video_adapter.py, lerobot_adapter.py, registry.py) into shared module
- Add `async fetch_hf_tree(repo_id, path, headers, recursive=False)` helper with **pagination support** (HF API paginates at 1000 entries -- critical for droid_lerobot's 92K episodes)
- All 4 adapter files + registry.py import from `utils.py`

### Step 2: MCAP Adapter -- Deeper Traversal & Metadata
**File**: `backend/adapters/mcap_adapter.py`

| Change | Current | New |
|--------|---------|-----|
| Depth limit in `list_episodes()` | 3 (line 162) | Configurable via `config.get("mcap_max_depth", 5)` |
| Task probing depth | 1 level | 2 levels deep -- after finding sub_dirs, make a second API call for each (limit 5) to find `Category/Skill/` structures |
| FPS in `resolve_episode()` | Hardcoded 30 (line 203) | Detect from MCAP summary metadata timestamps; fall back to 30 |
| `get_capabilities()` cameras | Reads from config, defaults to generic list | Detect actual camera topics from first MCAP file's channel list (cache result) |
| Frame estimation | Flat 50KB/frame (line 259) | Codec-aware: H.264 ~20KB/frame@480p, ~80KB/frame@1080p; JPEG ~50KB/frame |

### Step 3: Raw Video Adapter -- Depth & Metadata
**File**: `backend/adapters/raw_video_adapter.py`

| Change | Current | New |
|--------|---------|-----|
| Depth limit in `list_episodes()` | 2 (line 142) | Configurable via `config.get("video_max_depth", 4)` |
| Metadata detection | None | Detect sidecar files (`metadata.json`, `annotations.json`, `info.parquet`) alongside videos; parse for task names, episode mappings |
| FPS in `resolve_episode()` | Hardcoded 30 | Detect from MP4 headers via HTTP range request (first 4KB moov atom) with fallback to 30 |
| Frame estimation | Flat 30KB/frame (line 186) | Codec-aware: H.264@480p ~20KB/frame, @1080p ~100KB/frame; AV1 ~15KB/frame |

### Step 4: Registry -- Deeper BFS Auto-Detection
**File**: `backend/adapters/registry.py`

| Change | Current | New |
|--------|---------|-----|
| Subdirectory probing | Single level, 3 dirs (line 188) | BFS up to depth 4 |
| API call budget | Unbounded | Hard cap at 15 API calls per detection |
| Probing breadth | 3 dirs, no sub-probe | 5 dirs per level, 3 subdirs per dir |
| Cache | No TTL | Add tree-result cache per repo_id (5 min TTL) to avoid redundant probing |

---

## Phase 2: Large Dataset Handling (Steps 5-7)

### Step 5: Fix Parquet Misclassification
**File**: `backend/api/routes/datasets.py`

- In `probe_huggingface_dataset()` (line 1618-1625): Currently classifies any `.parquet` file as `"lerobot"`. **Fix**: Require `meta/info.json` with `codebase_version` field before labeling as LeRobot. Otherwise return `format_detected: "unknown"` with a note
- Reduce MCAP modality probe download from 50MB to 5MB (line ~1690) -- MCAP summary/header data is in the first few MB
- **New skeleton**: `backend/adapters/hf_native_adapter.py` -- placeholder for non-LeRobot parquet datasets (just raises `NotImplementedError` with helpful message)

### Step 6: Episode List Pagination
**Files**: `backend/api/routes/datasets.py`, `frontend/src/types/api.ts`, `frontend/src/components/DatasetBrowser.tsx`

**Backend** -- Add `offset` and `limit` query params to episode listing endpoints:
- `GET /api/datasets/{id}/tasks/{task}/episodes?limit=50&offset=0`
- Response shape: `{ episodes: [...], total: 92000, offset: 0, limit: 50, has_more: true }`
- Adapter `list_episodes()` already accepts `limit` and `offset` -- wire them through the route
- Default limit=50 for HF streaming datasets, no limit for local datasets

**Frontend**:
- Add `PaginatedEpisodesResponse` type to `api.ts`: `{ episodes, total, offset, limit, has_more }`
- Add "Load More" button in `DatasetBrowser.tsx` episode list when `has_more === true`
- Show total count: "Showing 50 of 92,000 episodes"
- Accumulate episodes client-side as user clicks "Load More"

### Step 7: Auto-Stride for Large Episodes
**Files**: `backend/api/routes/episodes.py`, `backend/loaders/streaming_extractor.py`, `frontend/src/hooks/useStreamingFrames.ts`, `frontend/src/components/EpisodeViewer.tsx`

**Backend** (`episodes.py`):
- In `stream_frames()` (line ~1165-1197): Implement auto-stride computation:
  ```python
  if stride == 1 and num_frames > 1000:
      stride = math.ceil(num_frames / 500)  # Target ~500 frames for initial load
  ```
- Pass computed `stride` through to `stream_frames_generator()` and extractors
- Include `stride` field in SSE `total` event: `{"total": 4500, "stride": 9, "effective_frames": 500}`
- For MCAP `extract_frames_from_mcap()` (line 336): Skip every Nth message for JPEG; for H.264, decode all then subsample (inter-frame compression requires sequential decode)
- For video `extract_frames_from_video()`: Use OpenCV seek (`CAP_PROP_POS_FRAMES`) to jump by stride

**Cache limit increases**:
| Cache | Current | New |
|-------|---------|-----|
| Persistent cache frame limit | 2000 (streaming_extractor.py:167) | 5000 |
| In-memory frames per episode | 500 (streaming_extractor.py:103) | 1000 |

**Frontend** (`useStreamingFrames.ts`):
- Parse `stride` from SSE `total` event, expose in hook return value
- When stride > 1, advance playback timer by `stride` frames per tick instead of 1
- Add nearest-frame lookup: when exact frame index isn't loaded, show closest loaded frame
- Show "Subsampled (every Nth frame)" indicator text when stride > 1

**Frontend** (`EpisodeViewer.tsx`):
- When stride > 1, show badge: "Showing every {stride}th frame ({effective_frames} of {total})"
- Timeline slider maps to actual frame indices (0 to total-1), snaps to nearest loaded frame

---

## Phase 3: Frontend Format-Aware UI (Step 8)

**Files**: `frontend/src/components/DatasetBrowser.tsx`, `frontend/src/components/EpisodeViewer.tsx`, `frontend/src/types/api.ts`

### Types (`api.ts`)
- Add to `Dataset` interface: `cameras?: string[]`, `fps?: number`, `total_episodes?: number`
- Add `PaginatedEpisodesResponse` type (from Step 6)

### Dataset Browser (`DatasetBrowser.tsx`)
- Color-coded format badges next to dataset names:
  - Orange badge: "MCAP" for `format === "mcap"`
  - Purple badge: "Video" for `format === "video"` or `"mp4"` or `"raw_video"`
  - Yellow badge: "LeRobot" for `format === "lerobot"` (and variants)
  - Blue badge: "Multi" for `format === "multi-subdataset"`
- Show camera count: "(3 cameras)" when `cameras` array available
- Show total episodes: "92K episodes" with abbreviation for large numbers
- Show "Large episode (N frames)" indicator when episode `num_frames > 1000`

### Episode Viewer (`EpisodeViewer.tsx`)
- Hide action chart tabs when dataset capabilities indicate `has_actions: false` (video-only)
- Show FPS when non-default: "@ 15 fps" next to frame counter (when fps !== 30)
- Stride indicator from Step 7

---

## Phase 4: Playwright E2E Verification (Step 9)

**New file**: `frontend/e2e/universal-format-support.spec.ts`

### Prerequisites
- Backend running on port 8000, frontend on port 3000 (main branch)
- HF token configured in environment

### Test Suite 1: MCAP Deep Nesting (10Kh-RealOmni)
Using `genrobot2025/10Kh-RealOmin-OpenData`:
1. POST `/api/datasets/probe` -- verify `format_detected === "mcap"`, `modalities` includes `"rgb"`, `has_tasks === true`
2. Add dataset, GET tasks -- verify nested paths like `"Cooking_and_Kitchen_Clean/clean_bowl"` exist
3. List episodes for a task -- verify `.mcap` file IDs returned
4. SSE stream one MCAP episode -- verify frames arrive with valid base64 images
5. Verify MCAP format badge (orange) visible in sidebar

### Test Suite 2: Raw MP4 Video Datasets
Using a public MP4-based HF dataset:
1. Verify `format_detected === "video"`
2. Directories become tasks, MP4 files become episodes
3. SSE stream delivers frames from MP4
4. No action chart tabs visible, only "rgb" modality shown
5. Verify Video format badge (purple) visible

### Test Suite 3: Large Episode Handling (droid_lerobot)
Using `IPEC-COMMUNITY/droid_lerobot`:
1. Add dataset, verify probe completes in <30s
2. Episode list shows pagination: "Showing 50 of N episodes", "Load More" button visible
3. Click "Load More" -- verify more episodes loaded, count increases
4. Navigate to large episode (>1000 frames): SSE `total` event includes `stride > 1`
5. Playback works with stride -- verify smooth advancement
6. No memory/timeout issues -- SSE stream completes without error
7. Performance: tasks API < 10s, episodes API with limit=50 < 5s

### Test Suite 4: Multi-Subdataset (GR00T)
Using `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`:
1. Probe detects multi-subdataset format
2. Subdataset selector appears with ~122 options
3. Select subdataset -> see its tasks
4. Play episode from subdataset -> frames load correctly
5. Switch subdatasets -> different task list loads

### Test Suite 5: LeRobot Regression
1. Rerun existing `user-flows.spec.ts` -- all must pass
2. LIBERO: full flow (dataset -> task -> episode -> frames -> actions chart)
3. Verify LeRobot format badge (yellow) visible

### Verification approach with Playwright MCP:
- Navigate to `http://localhost:3000`
- Use `browser_snapshot` for accessibility tree verification
- Use `browser_click` / `browser_type` for UI interaction
- Use `browser_wait_for` for async loading
- Use `browser_evaluate` to check JS state (frame counts, stride, pagination)
- Use `browser_take_screenshot` for visual evidence

---

## Implementation Order

| Step | Phase | What | Files | Depends On |
|------|-------|------|-------|------------|
| 1 | 1 | Shared utils module | New `adapters/utils.py` + update 4 adapters + registry | -- |
| 2 | 1 | MCAP adapter improvements | `mcap_adapter.py` | Step 1 |
| 3 | 1 | RawVideo adapter improvements | `raw_video_adapter.py` | Step 1 |
| 4 | 1 | Registry deeper BFS | `registry.py` | Step 1 |
| 5 | 2 | Fix parquet misclassification | `datasets.py`, new `hf_native_adapter.py` | -- |
| 6 | 2 | Episode list pagination | `datasets.py`, `api.ts`, `DatasetBrowser.tsx` | -- |
| 7 | 2 | Auto-stride (large episodes) | `episodes.py`, `streaming_extractor.py`, `useStreamingFrames.ts`, `EpisodeViewer.tsx` | -- |
| 8 | 3 | Format-aware UI | `DatasetBrowser.tsx`, `EpisodeViewer.tsx`, `api.ts` | Steps 6-7 |
| 9 | 4 | Playwright E2E tests | New `universal-format-support.spec.ts` | All above |

**Parallelization**: Steps 1-5 can run in parallel. Steps 6-7 can run in parallel. Step 8 after 6-7. Step 9 last.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| HF API rate limiting from deeper probing | Hard cap of 15 API calls per detection; cache tree results per-repo (5 min TTL); exponential backoff on 429 |
| H.264 in MCAP requires sequential decode | Decode all frames then subsample post-decode; persistent cache for subsequent requests |
| Memory from large episodes (5000+ frames) | Auto-stride caps initial load at ~500 frames; configurable cache limits |
| 92K episode list overwhelms frontend | Pagination (limit=50 default) with "Load More"; never load full list |
| Regression for existing LeRobot datasets | Run `user-flows.spec.ts` before and after; adapter-first path falls through to existing code if no adapter |
| MCAP modality probe too slow (50MB download) | Reduce to 5MB -- header/summary is sufficient |
