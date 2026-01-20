"""
Temporal Quality Metrics for robotics datasets.

These metrics assess the temporal characteristics of trajectories:
- Motion smoothness (velocity/acceleration consistency)
- Frame rate consistency
- Action consistency
- Trajectory completeness
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class TemporalMetrics:
    """Container for temporal quality metrics."""
    motion_smoothness: float  # 0-1, higher is smoother
    frame_rate_consistency: float  # 0-1, higher is more consistent
    action_consistency: float  # 0-1, higher is more consistent
    trajectory_completeness: float  # 0-1, 1 means complete
    sync_score: float  # 0-1, action-observation synchronization
    overall_temporal_score: float  # Weighted combination


def compute_motion_smoothness(actions: np.ndarray) -> float:
    """
    Compute motion smoothness from action sequences.

    Uses jerk-based analysis (third derivative) which is more robust to
    different action scales and better captures smoothness in robotics data.
    Returns a score between 0-1 where 1 is perfectly smooth.

    Args:
        actions: Array of shape (T, action_dim) containing actions

    Returns:
        Smoothness score between 0 and 1
    """
    if actions is None or len(actions) < 4:
        return 0.5  # Default for insufficient data

    # Normalize actions to have unit variance per dimension for scale invariance
    action_std = np.std(actions, axis=0)
    action_std = np.where(action_std < 1e-8, 1.0, action_std)  # Avoid division by zero
    normalized_actions = actions / action_std

    # Compute velocity (first derivative)
    velocity = np.diff(normalized_actions, axis=0)

    # Compute acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0)

    # Compute jerk (third derivative) - key indicator of smoothness
    jerk = np.diff(acceleration, axis=0)

    # Smoothness metric 1: Jerk magnitude (lower = smoother)
    jerk_magnitude = np.sqrt(np.sum(jerk ** 2, axis=1))
    mean_jerk = np.mean(jerk_magnitude)
    # Typical normalized jerk ranges from 0 (smooth) to 1+ (jerky)
    jerk_score = np.clip(1.0 - mean_jerk / 0.5, 0, 1)

    # Smoothness metric 2: Acceleration sign changes (fewer = smoother)
    # Count direction reversals in acceleration
    if len(acceleration) > 1:
        accel_signs = np.sign(acceleration)
        sign_changes = np.sum(np.abs(np.diff(accel_signs, axis=0)) > 0, axis=0)
        max_possible_changes = len(acceleration) - 1
        reversal_ratio = np.mean(sign_changes / max_possible_changes) if max_possible_changes > 0 else 0
        reversal_score = 1.0 - reversal_ratio
    else:
        reversal_score = 0.5

    # Smoothness metric 3: Spectral smoothness (low frequency dominance)
    # High frequency content indicates jerkiness
    try:
        fft_result = np.fft.rfft(velocity, axis=0)
        power = np.abs(fft_result) ** 2
        total_power = np.sum(power)
        if total_power > 1e-10:
            # Low frequency is first half of spectrum
            low_freq_cutoff = len(power) // 2
            low_freq_power = np.sum(power[:low_freq_cutoff])
            spectral_score = low_freq_power / total_power
        else:
            spectral_score = 0.5
    except Exception:
        spectral_score = 0.5

    # Combine metrics with weighted average
    smoothness = (
        0.4 * jerk_score +
        0.3 * reversal_score +
        0.3 * spectral_score
    )

    return float(np.clip(smoothness, 0, 1))


def compute_frame_rate_consistency(timestamps: Optional[np.ndarray] = None,
                                   num_frames: int = 0,
                                   expected_fps: float = 30.0) -> float:
    """
    Compute frame rate consistency.

    Args:
        timestamps: Optional array of timestamps
        num_frames: Number of frames if timestamps not available
        expected_fps: Expected frames per second

    Returns:
        Consistency score between 0 and 1
    """
    if timestamps is not None and len(timestamps) > 1:
        # Compute time deltas
        deltas = np.diff(timestamps)
        expected_delta = 1.0 / expected_fps

        # Coefficient of variation of deltas
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)

        if mean_delta < 1e-8:
            return 0.5

        cv = std_delta / mean_delta
        # Also check deviation from expected frame rate
        rate_deviation = abs(mean_delta - expected_delta) / expected_delta

        consistency = np.clip(1.0 - cv - rate_deviation * 0.5, 0, 1)
        return float(consistency)

    # Without timestamps, assume consistent
    return 0.8 if num_frames > 10 else 0.5


def compute_action_consistency(actions: np.ndarray) -> float:
    """
    Compute action consistency - how physically plausible the actions are.

    Checks for:
    - Sudden large jumps in action space
    - Actions within reasonable bounds
    - Gripper state transitions (should be gradual or binary)

    Args:
        actions: Array of shape (T, action_dim) containing actions

    Returns:
        Consistency score between 0 and 1
    """
    if actions is None or len(actions) < 2:
        return 0.5

    # Check for sudden jumps
    action_deltas = np.abs(np.diff(actions, axis=0))

    # For 7-DoF actions: xyz (0-2), rotation (3-5), gripper (6)
    # Position changes should be small (< 0.1 typically)
    # Rotation changes should be small (< 0.5 rad typically)

    if actions.shape[1] >= 7:
        # Position consistency (first 3 dims)
        pos_deltas = action_deltas[:, :3]
        large_pos_jumps = np.sum(np.any(pos_deltas > 0.2, axis=1))
        pos_consistency = 1.0 - (large_pos_jumps / len(pos_deltas))

        # Rotation consistency (dims 3-5)
        rot_deltas = action_deltas[:, 3:6]
        large_rot_jumps = np.sum(np.any(rot_deltas > 1.0, axis=1))
        rot_consistency = 1.0 - (large_rot_jumps / len(rot_deltas))

        # Gripper consistency (dim 6) - should change smoothly or be binary
        grip_deltas = action_deltas[:, 6]
        # Allow either smooth or binary transitions
        grip_consistency = 1.0 - np.mean(np.clip(grip_deltas, 0, 0.5) / 0.5) * 0.5

        consistency = 0.4 * pos_consistency + 0.3 * rot_consistency + 0.3 * grip_consistency
    else:
        # Generic consistency check
        max_delta = np.max(action_deltas, axis=1)
        large_jumps = np.sum(max_delta > 0.5)
        consistency = 1.0 - (large_jumps / len(max_delta))

    return float(np.clip(consistency, 0, 1))


def compute_sync_score(actions: np.ndarray,
                       observations: Optional[np.ndarray] = None) -> float:
    """
    Compute action-observation synchronization score.

    Measures the cross-correlation between optical flow (computed from
    observations) and end-effector velocity (from actions). High correlation
    means actions and visual changes are well synchronized.

    For datasets like Open X, the synchronization between camera and robot
    action is often slightly off. If action happens before/after visual change,
    the data is problematic for training.

    Args:
        actions: Array of shape (T, action_dim) containing actions
        observations: Array of shape (T, H, W, C) containing images

    Returns:
        Sync score between 0 and 1 (1 = perfectly synchronized)
    """
    if actions is None or len(actions) < 3:
        return 0.5  # Default for insufficient data

    # Compute action velocity magnitude (end-effector velocity proxy)
    # Typically first 3 dimensions are xyz position
    action_dim = min(3, actions.shape[1])
    position_actions = actions[:, :action_dim]
    action_velocity = np.diff(position_actions, axis=0)
    action_vel_magnitude = np.sqrt(np.sum(action_velocity ** 2, axis=1))

    if observations is None or len(observations) < 3:
        # Without observations, estimate sync from action signal consistency
        # Well-synchronized actions should have consistent velocity profiles
        if len(action_vel_magnitude) < 2:
            return 0.5
        # Check for autocorrelation - well-synced should have smooth transitions
        autocorr = np.correlate(action_vel_magnitude, action_vel_magnitude, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # Keep positive lags
        if autocorr[0] > 1e-10:
            autocorr = autocorr / autocorr[0]  # Normalize
            # Measure how quickly autocorrelation decays
            decay_rate = np.sum(autocorr[:min(10, len(autocorr))]) / min(10, len(autocorr))
            return float(np.clip(decay_rate, 0, 1))
        return 0.5

    # Compute optical flow magnitude from observations
    # Use simple frame differencing as optical flow proxy
    if observations.ndim == 4:  # (T, H, W, C)
        # Convert to grayscale if needed
        if observations.shape[-1] == 3:
            gray = np.mean(observations, axis=-1)
        else:
            gray = observations[..., 0] if observations.shape[-1] > 0 else observations.squeeze(-1)

        # Compute frame differences (optical flow proxy)
        frame_diff = np.abs(np.diff(gray.astype(np.float32), axis=0))
        optical_flow_magnitude = np.mean(frame_diff, axis=(1, 2))
    else:
        return 0.5

    # Ensure same length
    min_len = min(len(action_vel_magnitude), len(optical_flow_magnitude))
    action_vel_magnitude = action_vel_magnitude[:min_len]
    optical_flow_magnitude = optical_flow_magnitude[:min_len]

    if min_len < 3:
        return 0.5

    # Normalize signals
    def normalize(x):
        x = x - np.mean(x)
        std = np.std(x)
        return x / std if std > 1e-10 else x

    action_norm = normalize(action_vel_magnitude)
    flow_norm = normalize(optical_flow_magnitude)

    # Compute cross-correlation to find sync
    cross_corr = np.correlate(action_norm, flow_norm, mode='full')
    cross_corr = cross_corr / min_len  # Normalize

    # Find peak correlation and its lag
    center = len(cross_corr) // 2
    # Check within reasonable lag range (±5 frames for 30fps = ±166ms)
    lag_range = min(5, min_len // 2)
    search_region = cross_corr[center - lag_range:center + lag_range + 1]

    if len(search_region) == 0:
        return 0.5

    peak_corr = np.max(search_region)
    peak_lag = np.argmax(search_region) - lag_range

    # Score based on:
    # 1. Peak correlation magnitude (higher = better match)
    # 2. Peak lag (closer to 0 = better sync)
    correlation_score = np.clip((peak_corr + 1) / 2, 0, 1)  # Map [-1,1] to [0,1]
    lag_penalty = 1.0 - (abs(peak_lag) / (lag_range + 1))  # 0 lag = no penalty

    sync_score = 0.7 * correlation_score + 0.3 * lag_penalty

    return float(np.clip(sync_score, 0, 1))


def compute_trajectory_completeness(actions: np.ndarray,
                                   success_label: Optional[bool] = None,
                                   min_length: int = 10) -> float:
    """
    Compute trajectory completeness score.

    Args:
        actions: Array of actions
        success_label: Optional success label from metadata
        min_length: Minimum expected trajectory length

    Returns:
        Completeness score between 0 and 1
    """
    if success_label is not None:
        # If we have explicit success labels, use them
        return 1.0 if success_label else 0.3

    if actions is None:
        return 0.0

    # Heuristics for completeness without explicit labels
    length_score = min(1.0, len(actions) / min_length)

    # Check if trajectory ends in a "settled" state (low velocity)
    if len(actions) > 5:
        final_velocity = np.mean(np.abs(np.diff(actions[-5:], axis=0)))
        settled_score = np.clip(1.0 - final_velocity * 5, 0, 1)
    else:
        settled_score = 0.5

    completeness = 0.6 * length_score + 0.4 * settled_score
    return float(completeness)


def compute_temporal_metrics(actions: np.ndarray,
                            timestamps: Optional[np.ndarray] = None,
                            observations: Optional[np.ndarray] = None,
                            success_label: Optional[bool] = None,
                            expected_fps: float = 30.0) -> TemporalMetrics:
    """
    Compute all temporal quality metrics for an episode.

    Args:
        actions: Array of shape (T, action_dim) containing actions
        timestamps: Optional array of timestamps
        observations: Optional array of shape (T, H, W, C) for sync analysis
        success_label: Optional success label
        expected_fps: Expected frame rate

    Returns:
        TemporalMetrics dataclass with all metrics
    """
    smoothness = compute_motion_smoothness(actions)
    frame_consistency = compute_frame_rate_consistency(timestamps,
                                                       len(actions) if actions is not None else 0,
                                                       expected_fps)
    action_consistency = compute_action_consistency(actions)
    completeness = compute_trajectory_completeness(actions, success_label)
    sync = compute_sync_score(actions, observations)

    # Weighted overall score
    overall = (
        0.30 * smoothness +
        0.10 * frame_consistency +
        0.20 * action_consistency +
        0.15 * completeness +
        0.25 * sync  # Sync score is critical for training quality
    )

    return TemporalMetrics(
        motion_smoothness=smoothness,
        frame_rate_consistency=frame_consistency,
        action_consistency=action_consistency,
        trajectory_completeness=completeness,
        sync_score=sync,
        overall_temporal_score=overall
    )
