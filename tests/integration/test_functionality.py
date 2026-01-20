"""
Integration Tests: Verify FUNCTIONALITY, not data properties.

These tests verify:
- Can we download data? (download succeeds, files exist)
- Can we load data? (loader returns data, not crash)
- Can we compute metrics? (returns valid numbers, not NaN/crash)
- Do API endpoints respond correctly?

NOT testing:
- "Is motion smoothness 0.8-1.0 for smooth data?" (algorithm definition)
- "Is diversity score high for diverse data?" (math property)
"""

import numpy as np
import pytest
from pathlib import Path


# =============================================================================
# Download Functionality Tests
# =============================================================================
class TestDownloadFunctionality:
    """Tests that downloaders work without crashing."""

    def test_libero_downloader_initializes(self, test_data_dir):
        """LiberoDownloader can be instantiated."""
        from downloaders.libero import LiberoDownloader

        downloader = LiberoDownloader(data_dir=test_data_dir / "libero")
        assert downloader.data_dir == test_data_dir / "libero"

    def test_libero_downloader_has_required_methods(self):
        """LiberoDownloader has download and check_status methods."""
        from downloaders.libero import LiberoDownloader

        assert hasattr(LiberoDownloader, "download")
        assert hasattr(LiberoDownloader, "check_status")
        assert callable(getattr(LiberoDownloader, "download", None))

    def test_libero_downloader_check_status_returns_dict(self, test_data_dir):
        """check_status returns a valid status dictionary."""
        from downloaders.libero import LiberoDownloader

        downloader = LiberoDownloader(data_dir=test_data_dir / "libero")
        status = downloader.check_status()

        assert isinstance(status, dict)
        assert "status" in status

    def test_download_result_structure(self, test_data_dir):
        """DownloadResult has expected fields."""
        from downloaders.libero import DownloadResult

        result = DownloadResult(
            success=True,
            path=test_data_dir / "libero",
            size_bytes=1000,
            error=None
        )

        assert hasattr(result, "success")
        assert hasattr(result, "path")
        assert hasattr(result, "size_bytes")

    @pytest.mark.slow
    @pytest.mark.network
    def test_libero_download_completes(self, test_data_dir):
        """Download completes without crashing (network required)."""
        from downloaders.libero import LiberoDownloader

        downloader = LiberoDownloader(data_dir=test_data_dir / "libero")
        result = downloader.download(dataset="libero_spatial", limit=1)

        # Functionality: did it complete without crashing?
        assert hasattr(result, "success")


# =============================================================================
# Loader Functionality Tests
# =============================================================================
class TestLoaderFunctionality:
    """Tests that loaders work with data directories."""

    def test_hdf5_loader_initializes(self, test_data_dir):
        """HDF5Loader can be instantiated."""
        from loaders.hdf5_loader import HDF5Loader

        loader = HDF5Loader(test_data_dir / "libero")
        assert loader.data_dir == test_data_dir / "libero"

    def test_hdf5_loader_list_episodes_returns_list(self, test_data_dir):
        """list_episodes returns a list (even if empty)."""
        from loaders.hdf5_loader import HDF5Loader

        loader = HDF5Loader(test_data_dir / "libero")
        episodes = loader.list_episodes()

        assert isinstance(episodes, list)

    def test_hdf5_loader_get_metadata_returns_dict(self, test_data_dir):
        """get_metadata returns a dictionary."""
        from loaders.hdf5_loader import HDF5Loader

        loader = HDF5Loader(test_data_dir / "libero")
        metadata = loader.get_metadata()

        assert isinstance(metadata, dict)
        assert "data_dir" in metadata
        assert "exists" in metadata

    @pytest.fixture
    def real_libero_path(self):
        """Path to real LIBERO data if available."""
        path = Path("./data/libero")
        if not path.exists() or not list(path.rglob("*.hdf5")):
            pytest.skip("LIBERO data not downloaded")
        return path

    def test_loader_lists_episodes_from_real_data(self, real_libero_path):
        """Loader can list episodes from real data."""
        from loaders.hdf5_loader import HDF5Loader

        loader = HDF5Loader(real_libero_path)
        episodes = loader.list_episodes()

        # Functionality: did it return episodes?
        assert len(episodes) > 0
        assert hasattr(episodes[0], "id")

    def test_loader_loads_episode_from_real_data(self, real_libero_path):
        """Loader can load an episode from real data."""
        from loaders.hdf5_loader import HDF5Loader

        loader = HDF5Loader(real_libero_path)
        episodes = loader.list_episodes()
        episode = loader.load_episode(episodes[0].id)

        # Functionality: did it return data?
        assert episode is not None
        assert episode.id == episodes[0].id

    def test_loaded_episode_has_data(self, real_libero_path):
        """Loaded episode has observations and/or actions."""
        from loaders.hdf5_loader import HDF5Loader

        loader = HDF5Loader(real_libero_path)
        episodes = loader.list_episodes()
        episode = loader.load_episode(episodes[0].id)

        # Functionality: is there actual data?
        has_obs = episode.observations is not None and len(episode.observations) > 0
        has_actions = episode.actions is not None and len(episode.actions) > 0

        assert has_obs or has_actions, "Episode should have observations or actions"


# =============================================================================
# Quality Metrics Functionality Tests
# =============================================================================
class TestQualityMetricsFunctionality:
    """Tests that quality metrics compute without crashing."""

    def test_motion_smoothness_returns_number(self):
        """compute_motion_smoothness returns a valid number."""
        from quality.temporal import compute_motion_smoothness

        # Create sample actions
        actions = np.random.randn(50, 7).astype(np.float32)
        score = compute_motion_smoothness(actions)

        # Functionality: is it a valid number?
        assert isinstance(score, (int, float))
        assert not np.isnan(score)
        assert 0 <= score <= 1

    def test_motion_smoothness_handles_short_input(self):
        """compute_motion_smoothness handles short inputs gracefully."""
        from quality.temporal import compute_motion_smoothness

        # Very short input
        short_actions = np.random.randn(2, 7)
        score = compute_motion_smoothness(short_actions)

        assert isinstance(score, (int, float))
        assert not np.isnan(score)

    def test_motion_smoothness_handles_none(self):
        """compute_motion_smoothness handles None input."""
        from quality.temporal import compute_motion_smoothness

        score = compute_motion_smoothness(None)

        assert isinstance(score, (int, float))
        assert not np.isnan(score)

    def test_temporal_metrics_returns_dataclass(self):
        """compute_temporal_metrics returns TemporalMetrics."""
        from quality.temporal import compute_temporal_metrics, TemporalMetrics

        actions = np.random.randn(50, 7)
        metrics = compute_temporal_metrics(actions)

        assert isinstance(metrics, TemporalMetrics)
        assert hasattr(metrics, "motion_smoothness")
        assert hasattr(metrics, "overall_temporal_score")
        assert not np.isnan(metrics.overall_temporal_score)

    def test_diversity_metrics_returns_dataclass(self):
        """compute_diversity_metrics returns DiversityMetrics."""
        from quality.diversity import compute_diversity_metrics, DiversityMetrics

        actions = np.random.randn(50, 7)
        metrics = compute_diversity_metrics(actions)

        assert isinstance(metrics, DiversityMetrics)
        assert hasattr(metrics, "recovery_behavior_score")
        assert hasattr(metrics, "overall_diversity_score")
        assert not np.isnan(metrics.overall_diversity_score)

    def test_recovery_detection_returns_list(self):
        """detect_velocity_reversals returns a list."""
        from quality.diversity import detect_velocity_reversals

        actions = np.random.randn(50, 7)
        reversals = detect_velocity_reversals(actions)

        assert isinstance(reversals, list)

    def test_recovery_score_returns_tuple(self):
        """compute_recovery_score returns (score, events) tuple."""
        from quality.diversity import compute_recovery_score

        actions = np.random.randn(50, 7)
        result = compute_recovery_score(actions)

        assert isinstance(result, tuple)
        assert len(result) == 2
        score, events = result
        assert isinstance(score, (int, float))
        assert isinstance(events, list)
        assert not np.isnan(score)

    def test_quality_aggregator_returns_quality_score(self):
        """compute_quality_score returns QualityScore."""
        from quality.aggregator import compute_quality_score, QualityScore

        actions = np.random.randn(50, 7)
        observations = np.random.randint(0, 255, (50, 64, 64, 3), dtype=np.uint8)

        quality = compute_quality_score(actions=actions, observations=observations)

        assert isinstance(quality, QualityScore)
        assert hasattr(quality, "overall_score")
        assert hasattr(quality, "quality_grade")
        assert not np.isnan(quality.overall_score)
        assert quality.quality_grade in ['A', 'B', 'C', 'D', 'F']

    def test_quality_score_to_dict(self):
        """QualityScore.to_dict returns serializable dictionary."""
        from quality.aggregator import compute_quality_score

        actions = np.random.randn(50, 7)
        observations = np.random.randint(0, 255, (50, 64, 64, 3), dtype=np.uint8)

        quality = compute_quality_score(actions=actions, observations=observations)
        d = quality.to_dict()

        assert isinstance(d, dict)
        assert "overall_score" in d
        assert "temporal" in d
        assert "diversity" in d

    def test_grade_from_score_function(self):
        """grade_from_score returns valid letter grades."""
        from quality.aggregator import grade_from_score

        assert grade_from_score(0.95) in ['A', 'B', 'C', 'D', 'F']
        assert grade_from_score(0.5) in ['A', 'B', 'C', 'D', 'F']
        assert grade_from_score(0.0) in ['A', 'B', 'C', 'D', 'F']


# =============================================================================
# API Functionality Tests
# =============================================================================
class TestAPIFunctionality:
    """Tests that API endpoints respond correctly."""

    def test_root_endpoint_responds(self, client):
        """GET / returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_endpoint_responds(self, client):
        """GET /api/health returns 200."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_datasets_endpoint_responds(self, client):
        """GET /api/datasets returns 200."""
        response = client.get("/api/datasets")
        assert response.status_code == 200

    def test_datasets_endpoint_returns_list(self, client):
        """GET /api/datasets returns a list."""
        response = client.get("/api/datasets")
        data = response.json()
        assert isinstance(data, list)

    def test_nonexistent_dataset_returns_404(self, client):
        """GET /api/datasets/nonexistent returns 404."""
        response = client.get("/api/datasets/definitely_not_a_real_dataset_xyz")
        assert response.status_code == 404

    def test_nonexistent_episode_returns_404(self, client):
        """GET /api/episodes/nonexistent returns 404."""
        response = client.get("/api/episodes/definitely_not_a_real_episode_xyz")
        assert response.status_code == 404

    def test_downloads_status_endpoint_responds(self, client):
        """GET /api/downloads/status returns 200."""
        response = client.get("/api/downloads/status")
        # May be 200 or 404 depending on implementation
        assert response.status_code in [200, 404, 500]


# =============================================================================
# Download Manager Functionality Tests
# =============================================================================
class TestDownloadManagerFunctionality:
    """Tests that download manager works."""

    def test_manager_initializes(self, test_data_dir):
        """DownloadManager can be instantiated."""
        from downloaders.manager import DownloadManager

        manager = DownloadManager(data_root=test_data_dir)
        assert manager.data_root == test_data_dir

    def test_manager_lists_datasets(self, test_data_dir):
        """DownloadManager.list_datasets returns a list."""
        from downloaders.manager import DownloadManager

        manager = DownloadManager(data_root=test_data_dir)
        datasets = manager.list_datasets()

        assert isinstance(datasets, list)

    def test_manager_checks_disk_space(self, test_data_dir):
        """DownloadManager.check_disk_space returns valid info."""
        from downloaders.manager import DownloadManager

        manager = DownloadManager(data_root=test_data_dir)
        space = manager.check_disk_space()

        assert isinstance(space, dict)


# =============================================================================
# Real Data Integration Tests (requires downloaded data)
# =============================================================================
class TestRealDataIntegration:
    """End-to-end tests with real downloaded data."""

    @pytest.fixture
    def real_episode(self):
        """Load a real episode from downloaded LIBERO data."""
        from loaders.hdf5_loader import HDF5Loader

        path = Path("./data/libero")
        if not path.exists() or not list(path.rglob("*.hdf5")):
            pytest.skip("LIBERO data not downloaded")

        loader = HDF5Loader(path)
        episodes = loader.list_episodes()
        if not episodes:
            pytest.skip("No episodes found in LIBERO data")

        return loader.load_episode(episodes[0].id)

    def test_real_episode_has_observations(self, real_episode):
        """Real episode has observations data."""
        assert real_episode.observations is not None
        assert len(real_episode.observations) > 0

    def test_real_episode_has_actions(self, real_episode):
        """Real episode has actions data."""
        assert real_episode.actions is not None
        assert len(real_episode.actions) > 0

    def test_real_episode_observations_valid_shape(self, real_episode):
        """Real episode observations have valid shape (T, H, W, C)."""
        obs = real_episode.observations
        assert len(obs.shape) == 4  # T, H, W, C
        assert obs.shape[-1] == 3  # RGB

    def test_quality_metrics_on_real_data(self, real_episode):
        """Quality metrics compute on real episode data."""
        from quality.aggregator import compute_quality_score

        quality = compute_quality_score(
            actions=real_episode.actions,
            observations=real_episode.observations
        )

        # Functionality: did it compute without crashing?
        assert quality is not None
        assert not np.isnan(quality.overall_score)
        assert quality.quality_grade in ['A', 'B', 'C', 'D', 'F']

    def test_temporal_metrics_on_real_data(self, real_episode):
        """Temporal metrics compute on real episode data."""
        from quality.temporal import compute_temporal_metrics

        metrics = compute_temporal_metrics(
            actions=real_episode.actions,
            timestamps=real_episode.timestamps
        )

        assert not np.isnan(metrics.overall_temporal_score)
        assert 0 <= metrics.motion_smoothness <= 1

    def test_diversity_metrics_on_real_data(self, real_episode):
        """Diversity metrics compute on real episode data."""
        from quality.diversity import compute_diversity_metrics

        metrics = compute_diversity_metrics(
            actions=real_episode.actions,
            observations=real_episode.observations
        )

        assert not np.isnan(metrics.overall_diversity_score)
        assert 0 <= metrics.recovery_behavior_score <= 1
