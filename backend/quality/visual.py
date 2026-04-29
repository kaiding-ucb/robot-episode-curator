"""
Visual Quality Metrics for robotics datasets.

These metrics assess the visual quality of observations:
- Resolution quality
- Blur detection (Laplacian variance)
- Exposure quality (histogram analysis)
- Contrast quality
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class VisualMetrics:
    """Container for visual quality metrics."""
    resolution_score: float  # 0-1, based on image dimensions
    blur_score: float  # 0-1, higher is sharper
    exposure_score: float  # 0-1, higher is better exposed
    contrast_score: float  # 0-1, higher is better contrast
    overall_visual_score: float  # Weighted combination


def compute_resolution_score(height: int, width: int,
                            reference_height: int = 1080,
                            reference_width: int = 1920) -> float:
    """
    Compute resolution quality score.

    Args:
        height: Image height
        width: Image width
        reference_height: Reference height (1080p)
        reference_width: Reference width (1080p)

    Returns:
        Resolution score between 0 and 1
    """
    actual_pixels = height * width
    reference_pixels = reference_height * reference_width

    # Score based on pixel count ratio, capped at 1.0
    score = min(1.0, actual_pixels / reference_pixels)

    # Boost score slightly for very low res (some data is intentionally low res)
    if score < 0.1:
        score = 0.1 + score * 0.5  # Min 0.1, gradual increase

    return float(score)


def compute_laplacian_variance(image: np.ndarray) -> float:
    """
    Compute Laplacian variance for blur detection.

    Higher variance indicates sharper image.

    Args:
        image: Image array of shape (H, W) or (H, W, C)

    Returns:
        Laplacian variance value
    """
    if image is None:
        return 0.0

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        # Simple grayscale conversion
        gray = np.mean(image, axis=2).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    # Simple Laplacian kernel
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    # Manual convolution (avoiding cv2 dependency)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0

    # Pad image
    padded = np.pad(gray, 1, mode='edge')

    # Apply kernel
    laplacian = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            laplacian += padded[i:i+h, j:j+w] * laplacian_kernel[i, j]

    return float(np.var(laplacian))


def compute_blur_score(image: np.ndarray) -> float:
    """
    Compute blur score from Laplacian variance.

    Args:
        image: Image array

    Returns:
        Blur score between 0 and 1 (1 = sharp, 0 = blurry)
    """
    variance = compute_laplacian_variance(image)

    # Typical variance ranges:
    # < 100: Very blurry
    # 100-500: Moderately sharp
    # > 500: Sharp

    if variance < 50:
        score = variance / 100  # 0 to 0.5
    elif variance < 500:
        score = 0.5 + (variance - 50) / 900  # 0.5 to 1.0
    else:
        score = 1.0

    return float(np.clip(score, 0, 1))


def compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Compute histogram of image intensity.

    Args:
        image: Image array
        bins: Number of histogram bins

    Returns:
        Histogram array
    """
    if image is None:
        return np.zeros(bins)

    # Flatten and normalize to 0-255 range
    if image.dtype == np.float32 or image.dtype == np.float64:
        flat = (image.flatten() * 255).astype(np.int32)
    else:
        flat = image.flatten().astype(np.int32)

    flat = np.clip(flat, 0, 255)

    # Compute histogram
    hist, _ = np.histogram(flat, bins=bins, range=(0, 255))

    return hist


def compute_exposure_score(image: np.ndarray) -> float:
    """
    Compute exposure quality score.

    Checks for:
    - Clipping at highlights (255)
    - Clipping at shadows (0)
    - Mean brightness (ideal around 128)

    Args:
        image: Image array

    Returns:
        Exposure score between 0 and 1
    """
    if image is None:
        return 0.5

    hist = compute_histogram(image)
    total_pixels = np.sum(hist)

    if total_pixels == 0:
        return 0.5

    # Check for clipping
    shadow_clip = np.sum(hist[:5]) / total_pixels  # Very dark pixels
    highlight_clip = np.sum(hist[-5:]) / total_pixels  # Very bright pixels

    # Penalize clipping (> 5% is bad)
    clip_penalty = min(1.0, (shadow_clip + highlight_clip) * 10)

    # Check mean brightness
    if image.dtype == np.uint8:
        mean_brightness = np.mean(image) / 255.0
    else:
        mean_brightness = np.mean(image)

    # Ideal brightness is around 0.4-0.6
    brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2

    # Combined score
    score = 0.6 * brightness_score + 0.4 * (1.0 - clip_penalty)

    return float(np.clip(score, 0, 1))


def compute_contrast_score(image: np.ndarray) -> float:
    """
    Compute contrast quality score.

    Uses standard deviation of pixel values as a measure of contrast.

    Args:
        image: Image array

    Returns:
        Contrast score between 0 and 1
    """
    if image is None:
        return 0.5

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # Normalize to 0-1
    if gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255.0

    # Standard deviation as contrast measure
    std = np.std(gray)

    # Ideal std is around 0.2-0.3 for good contrast
    # Very low std = flat/washed out image
    # Very high std = harsh contrast

    if std < 0.1:
        score = std / 0.1 * 0.5  # Low contrast
    elif std < 0.25:
        score = 0.5 + (std - 0.1) / 0.15 * 0.5  # Good range
    elif std < 0.4:
        score = 1.0  # Ideal
    else:
        score = max(0.5, 1.0 - (std - 0.4) * 2)  # Too harsh

    return float(np.clip(score, 0, 1))


def compute_visual_metrics_single_frame(image: np.ndarray) -> VisualMetrics:
    """
    Compute visual metrics for a single frame.

    Args:
        image: Image array of shape (H, W, C) or (H, W)

    Returns:
        VisualMetrics dataclass
    """
    if image is None:
        return VisualMetrics(
            resolution_score=0.0,
            blur_score=0.0,
            exposure_score=0.0,
            contrast_score=0.0,
            overall_visual_score=0.0
        )

    h, w = image.shape[:2]
    resolution = compute_resolution_score(h, w)
    blur = compute_blur_score(image)
    exposure = compute_exposure_score(image)
    contrast = compute_contrast_score(image)

    overall = 0.25 * resolution + 0.35 * blur + 0.25 * exposure + 0.15 * contrast

    return VisualMetrics(
        resolution_score=resolution,
        blur_score=blur,
        exposure_score=exposure,
        contrast_score=contrast,
        overall_visual_score=overall
    )


def compute_visual_metrics(observations: np.ndarray,
                          sample_frames: int = 10) -> VisualMetrics:
    """
    Compute visual metrics for an episode by sampling frames.

    Args:
        observations: Array of shape (T, H, W, C) containing images
        sample_frames: Number of frames to sample for analysis

    Returns:
        VisualMetrics dataclass with averaged metrics
    """
    if observations is None or len(observations) == 0:
        return VisualMetrics(
            resolution_score=0.0,
            blur_score=0.0,
            exposure_score=0.0,
            contrast_score=0.0,
            overall_visual_score=0.0
        )

    # Sample frames evenly across the episode
    num_frames = len(observations)
    indices = np.linspace(0, num_frames - 1, min(sample_frames, num_frames), dtype=int)

    metrics_list = []
    for idx in indices:
        frame_metrics = compute_visual_metrics_single_frame(observations[idx])
        metrics_list.append(frame_metrics)

    # Average metrics
    avg_resolution = np.mean([m.resolution_score for m in metrics_list])
    avg_blur = np.mean([m.blur_score for m in metrics_list])
    avg_exposure = np.mean([m.exposure_score for m in metrics_list])
    avg_contrast = np.mean([m.contrast_score for m in metrics_list])
    avg_overall = np.mean([m.overall_visual_score for m in metrics_list])

    return VisualMetrics(
        resolution_score=float(avg_resolution),
        blur_score=float(avg_blur),
        exposure_score=float(avg_exposure),
        contrast_score=float(avg_contrast),
        overall_visual_score=float(avg_overall)
    )
