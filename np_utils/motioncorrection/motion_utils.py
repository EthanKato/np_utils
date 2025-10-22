"""
Reusable utilities for motion correction analysis.

Contains functions for:
- Running MEDiCINE algorithm
- Detecting and localizing peaks
- Plotting drift maps, traces, and before/after comparisons
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.core.motion import Motion
import json
import shutil

try:
    from medicine import run_medicine
except ImportError:
    run_medicine = None


def get_peaks_medicine(
    recording,
    cache_dir: Path,
    detect_threshold: float = 5.0,
    job_kwargs: dict = None,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect and localize peaks using MEDiCINE-compatible parameters.
    
    Caches results to disk to avoid recomputation across multiple presets.
    
    Args:
        recording: SpikeInterface recording object
        cache_dir: Directory to cache peak detection results
        detect_threshold: Detection threshold for peaks
        job_kwargs: Job parallelization kwargs (n_jobs, chunk_duration, progress_bar)
        overwrite: Force recomputation even if cache exists
    
    Returns:
        tuple: (peaks, peak_locations) as structured numpy arrays
    """
    if job_kwargs is None:
        job_kwargs = {}
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    peaks_npy = cache_dir / f"peaks_medicine_thr{detect_threshold}.npy"
    locs_npy = cache_dir / f"peak_locs_medicine_thr{detect_threshold}.npy"

    if peaks_npy.exists() and locs_npy.exists() and not overwrite:
        peaks = np.load(peaks_npy)
        locs = np.load(locs_npy)
        return peaks, locs

    detect_kwargs = dict(
        peak_sign="neg",
        detect_threshold=detect_threshold,
        exclude_sweep_ms=0.8,
        radius_um=80.0,
        method="locally_exclusive",
    )
    peaks = detect_peaks(recording, **detect_kwargs, **job_kwargs)

    loc_kwargs = dict(
        method="monopolar_triangulation",
        radius_um=75.0,
        max_distance_um=100.0,
    )

    peak_locs = localize_peaks(recording, peaks, **loc_kwargs, **job_kwargs)

    np.save(peaks_npy, peaks)
    np.save(locs_npy, peak_locs)

    return peaks, peak_locs


def compute_medicine_external(
    recording,
    output_folder: Path,
    num_depth_bins: int,
    peaks: np.ndarray,
    peak_locations: np.ndarray,
) -> Tuple[Motion, Dict[str, Any]]:
    """
    Run external MEDiCINE algorithm and return Motion object.
    
    Args:
        recording: SpikeInterface recording object
        output_folder: Directory to save MEDiCINE outputs
        num_depth_bins: Number of depth bins for MEDiCINE
        peaks: Detected peaks (must have 'amplitude' and 'sample_index' fields)
        peak_locations: Peak locations (must have 'y' field for depth)
    
    Returns:
        tuple: (Motion object, info dict)
    """
    if run_medicine is None:
        raise ImportError("MEDiCINE package not found. Install with: pip install medicine")
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    fs = recording.get_sampling_frequency()
    run_medicine(
        peak_amplitudes=peaks["amplitude"],
        peak_depths=peak_locations["y"],
        peak_times=peaks["sample_index"] / fs,
        output_dir=output_folder,
        num_depth_bins=num_depth_bins,
    )
    
    # Load MEDiCINE outputs
    disp = np.load(output_folder / "motion.npy").squeeze()
    tb = np.load(output_folder / "time_bins.npy").squeeze()
    yb = np.load(output_folder / "depth_bins.npy").squeeze()
    
    # Handle shape issues
    if disp.ndim == 3 and disp.shape[0] == 1:
        disp = disp[0]
    if tb.size == disp.shape[0] + 1:
        tb = 0.5 * (tb[:-1] + tb[1:])
    
    motion = Motion(displacement=disp, temporal_bins_s=tb, spatial_bins_um=yb)
    info = {"method": "medicine_external", "num_depth_bins": int(num_depth_bins)}
    
    with open(output_folder / "motion_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    return motion, info


def load_motion_from_folder(preset_folder: Path) -> Motion:
    """
    Load Motion object from preset folder (handles both SI and MEDiCINE formats).
    
    Args:
        preset_folder: Folder containing motion correction outputs
    
    Returns:
        Motion object
    """
    preset_folder = Path(preset_folder)
    
    if (preset_folder / 'motion').exists():
        # SpikeInterface format
        motion_path = preset_folder / 'motion'
        motion = np.load(motion_path / 'displacement_seg0.npy')
        time_bins = np.load(motion_path / 'temporal_bins_s_seg0.npy')
        depth_bins = np.load(motion_path / 'spatial_bins_um.npy')
    else:
        # MEDiCINE format
        motion = np.load(preset_folder / 'motion.npy')
        time_bins = np.load(preset_folder / 'time_bins.npy')
        depth_bins = np.load(preset_folder / 'depth_bins.npy')

    return Motion(
        displacement=motion,
        temporal_bins_s=time_bins,
        spatial_bins_um=depth_bins
    )


def save_peak_map(
    recording,
    peaks: np.ndarray,
    peak_locations: np.ndarray,
    stable_range: Tuple[float, float],
    output_path: Path,
    rec_id: str,
    probe_id: str,
    preset_name: str,
    alpha: float = None,
    copy_to_paper: bool = True,
) -> Path:
    """
    Save scatter plot of detected peaks in time/depth space.
    
    Args:
        recording: SpikeInterface recording object
        peaks: Detected peaks
        peak_locations: Peak locations
        stable_range: (t0, t1) time range in seconds
        output_path: Where to save the figure
        rec_id: Recording ID for title
        probe_id: Probe ID for title
        preset_name: Preset name for title
        alpha: Scatter plot transparency (auto-determined if None)
        copy_to_paper: Also save copy to paper directory
    
    Returns:
        Path to saved figure
    """
    if alpha is None:
        # Auto-select alpha based on preset
        if preset_name in {"dredge_th6", "medicine", "medicine_ndb2", "medicine_ndb4"}:
            alpha = 0.05
        else:
            alpha = 0.5
    
    fs = float(recording.get_sampling_frequency())
    t0, t1 = stable_range
    t_abs = peaks["sample_index"].astype(float) / fs + t0
    y_um = peak_locations["y"].astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.scatter(t_abs, y_um, color="k", alpha=alpha, s=0.2, rasterized=True)
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 8000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance from probe tip (µm)")
    ax.set_title(f"Peak map — {preset_name}, {probe_id}, stable = {t0:.2f}–{t1:.2f} s")
    ax.set_xticks(np.linspace(t0, t1, 6))
    fig.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    
    if copy_to_paper:
        paper_path = Path("/userdata/ekato/stabilizer_paper/motion_traces") / rec_id / f"{preset_name}_spike_map.png"
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_path, dpi=150)
    
    plt.close(fig)
    return output_path


def save_motion_traces(
    recording,
    motion: Motion,
    peaks: np.ndarray,
    peak_locations: np.ndarray,
    stable_range: Tuple[float, float],
    output_path: Path,
    rec_id: str,
    probe_id: str,
    preset_name: str,
    n_interp: int = 16,
    sign: int = +1,
    alpha: float = 0.08,
    interpolate: bool = None,
    copy_to_paper: bool = True,
) -> Path:
    """
    Save motion traces overlaid on peak scatter plot.
    
    Args:
        recording: SpikeInterface recording object
        motion: Motion object
        peaks: Detected peaks
        peak_locations: Peak locations
        stable_range: (t0, t1) time range in seconds
        output_path: Where to save the figure
        rec_id: Recording ID for title
        probe_id: Probe ID for title
        preset_name: Preset name for title
        n_interp: Number of interpolated traces
        sign: Sign convention for displacement (+1 or -1)
        alpha: Scatter plot transparency
        interpolate: Whether to add interpolated traces (auto-determined if None)
        copy_to_paper: Also save copy to paper directory
    
    Returns:
        Path to saved figure
    """
    if interpolate is None:
        # Auto-determine based on preset
        interpolate = ('medicine' in preset_name.lower() and 'si' not in preset_name.lower())
    
    fs = float(recording.get_sampling_frequency())
    t0, t1 = stable_range
    t_abs = peaks["sample_index"].astype(float) / fs + t0
    y_um = peak_locations["y"].astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.scatter(t_abs, y_um, color="k", alpha=alpha, s=0.2, rasterized=True)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance from probe tip (µm)")
    ax.set_title(f"Motion traces — {preset_name}, {probe_id}, stable = {t0:.2f}–{t1:.2f} s")
    ax.set_xticks(np.linspace(t0, t1, 6))

    # Unpack motion
    disp = np.asarray(motion.displacement).squeeze()
    tb = np.asarray(motion.temporal_bins_s).squeeze()
    yb = np.asarray(motion.spatial_bins_um).squeeze()

    if tb.size == disp.shape[0] + 1:
        tb = 0.5 * (tb[:-1] + tb[1:])
    
    # Adjust time bins based on preset
    if 'medicine' in preset_name.lower():
        tb_abs = tb + t0
    else:
        tb_abs = tb

    ax.set_ylim(-500, 8000)

    # Interpolated bands (optional)
    if interpolate and disp.shape[1] >= 2:
        y_dense = np.linspace(yb.min(), yb.max(), int(n_interp))
        disp_i = np.vstack([np.interp(y_dense, yb, disp[k, :]) for k in range(disp.shape[0])])
        for j, y0 in enumerate(y_dense):
            y_tr = y0 + sign * disp_i[:, j]
            ax.plot(tb_abs, y_tr, lw=1, alpha=1, color="C0")

    # Control/true bands (always plot)
    for d, y0 in enumerate(yb):
        y_tr = y0 + sign * disp[:, d]
        ax.plot(tb_abs, y_tr, lw=1, alpha=1, color="C1")

    fig.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    
    if copy_to_paper:
        paper_path = Path("/userdata/ekato/stabilizer_paper/motion_traces") / rec_id / f"{preset_name}_motion_traces.png"
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_path, dpi=150)
    
    plt.close(fig)
    return output_path


def plot_drift_maps_before_after(
    recording,
    motion: Motion,
    peaks: np.ndarray,
    peak_locations: np.ndarray,
    stable_range: Tuple[float, float],
    output_path: Path,
    rec_id: str,
    probe_id: str,
    preset_name: str,
    sign: int = +1,
    decimate: int = None,
    color_amplitude: bool = False,
    amplitude_cmap: str = "inferno",
    alpha: float = 0.15,
    copy_to_paper: bool = True,
) -> Path:
    """
    Plot drift map before/after correction by warping peak depths with Motion field.
    
    Args:
        recording: SpikeInterface recording object
        motion: Motion object
        peaks: Detected peaks
        peak_locations: Peak locations
        stable_range: (t0, t1) time range in seconds
        output_path: Where to save the figure
        rec_id: Recording ID for title
        probe_id: Probe ID for title
        preset_name: Preset name for title
        sign: Sign convention for displacement (+1 or -1)
        decimate: Downsample peaks by this factor (None = no decimation)
        color_amplitude: Color by peak amplitude
        amplitude_cmap: Colormap for amplitudes
        alpha: Scatter plot transparency
        copy_to_paper: Also save copy to paper directory
    
    Returns:
        Path to saved figure
    """
    t0, t1 = map(float, stable_range)

    # Peaks to absolute time/depth
    fs = float(recording.get_sampling_frequency())
    t_peak = peaks["sample_index"].astype(float) / fs + t0
    y_peak = peak_locations["y"].astype(float)
    
    if decimate and decimate > 1:
        idx = np.arange(t_peak.size)[::decimate]
        t_peak, y_peak = t_peak[idx], y_peak[idx]
        amp = peaks["amplitude"][idx] if ("amplitude" in peaks.dtype.names) else None
    else:
        amp = peaks["amplitude"] if ("amplitude" in peaks.dtype.names) else None

    # Unpack Motion
    disp = np.asarray(motion.displacement).squeeze()
    tb = np.asarray(motion.temporal_bins_s).squeeze()
    yb = np.asarray(motion.spatial_bins_um).squeeze()

    if tb.size == disp.shape[0] + 1:
        tb = 0.5 * (tb[:-1] + tb[1:])
    
    if 'medicine' in preset_name.lower():
        tb_abs = tb + t0
    else:
        tb_abs = tb

    # Sort depth bins for interpolation
    order = np.argsort(yb)
    yb = yb[order]
    disp = disp[:, order]

    # Bilinear interpolation Δ(t_peak, y_peak)
    disp_t = np.stack([np.interp(t_peak, tb_abs, disp[:, d]) for d in range(disp.shape[1])], axis=1)
    delta = np.array([np.interp(y_peak[i], yb, disp_t[i, :]) for i in range(t_peak.size)])

    # Corrected depths
    y_corr = y_peak - sign * delta

    # Plot
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), dpi=150, sharex=True, sharey=True)

    # BEFORE
    if color_amplitude and (amp is not None):
        sc0 = ax0.scatter(t_peak, y_peak, c=amp, s=0.2, alpha=alpha, cmap=amplitude_cmap, rasterized=True)
        plt.colorbar(sc0, ax=ax0, label="Peak amplitude")
    else:
        ax0.scatter(t_peak, y_peak, color="k", s=0.2, alpha=alpha, rasterized=True)
    ax0.set_title("Drift map (before)")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Distance from probe tip (µm)")
    ax0.set_xlim(t0, t1)

    # AFTER (corrected)
    if color_amplitude and (amp is not None):
        sc1 = ax1.scatter(t_peak, y_corr, c=amp, s=0.2, alpha=alpha, cmap=amplitude_cmap, rasterized=True)
        plt.colorbar(sc1, ax=ax1, label="Peak amplitude")
    else:
        ax1.scatter(t_peak, y_corr, color="k", s=0.2, alpha=alpha, rasterized=True)
    ax1.set_title("Drift map (after correction)")
    ax1.set_xlabel("Time (s)")
    ax1.set_xlim(t0, t1)

    fig.suptitle(f"Drift maps — {preset_name} ({rec_id}, {probe_id})")
    fig.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)

    if copy_to_paper:
        paper_path = Path("/userdata/ekato/stabilizer_paper/motion_traces") / rec_id / f"{preset_name}_corrected_drift_maps.png"
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_path, dpi=150)

    plt.close(fig)
    return output_path

def reorganize_motion_traces(motion_traces_path, dry_run=True):
    """
    Reorganize motion traces directory structure.
    
    For each algorithm subdirectory in motion_traces_path, creates a new
    subdirectory named NPXXX_BXX_imec0 and moves all existing files into it.
    
    Parameters
    ----------
    motion_traces_path : str or Path
        Path to the motion_traces directory (e.g., /path/to/NP0123_B01/motion_traces)
    dry_run : bool, default=True
        If True, only print what would be done without making changes
    
    Example
    -------
    >>> reorganize_motion_traces('/path/to/NP0123_B01/motion_traces', dry_run=True)
    >>> reorganize_motion_traces('/path/to/NP0123_B01/motion_traces', dry_run=False)
    """
    motion_traces_path = Path(motion_traces_path)
    
    # Extract the session name (NPXXX_BXX) from the parent directory
    session_name = motion_traces_path.parent.name
    new_subdir_name = f"{session_name}_imec0"
    
    print(f"Session: {session_name}")
    print(f"Motion traces path: {motion_traces_path}")
    print(f"New subdirectory name: {new_subdir_name}")
    print(f"Dry run: {dry_run}")
    print("-" * 80)
    
    if not motion_traces_path.exists():
        print(f"ERROR: Path does not exist: {motion_traces_path}")
        return
    
    # Get all subdirectories (algorithm directories)
    algo_dirs = [d for d in motion_traces_path.iterdir() if d.is_dir()]
    
    if not algo_dirs:
        print(f"No subdirectories found in {motion_traces_path}")
        return
    
    print(f"Found {len(algo_dirs)} algorithm directories:")
    for algo_dir in algo_dirs:
        print(f"  - {algo_dir.name}")
    print()
    
    # Process each algorithm directory
    for algo_dir in algo_dirs:
        print(f"\nProcessing: {algo_dir.name}/")
        
        # Get all files and directories in this algorithm directory
        contents = list(algo_dir.iterdir())
        
        if not contents:
            print(f"  └─ (empty, skipping)")
            continue
        
        # Path for the new subdirectory
        new_subdir = algo_dir / new_subdir_name
        
        # Check if new subdirectory already exists
        if new_subdir.exists():
            print(f"  └─ WARNING: {new_subdir_name}/ already exists, skipping")
            continue
        
        print(f"  └─ Creating: {new_subdir_name}/")
        if not dry_run:
            new_subdir.mkdir(exist_ok=True)
        
        # Move all contents into the new subdirectory
        for item in contents:
            rel_name = item.name
            dest = new_subdir / rel_name
            
            print(f"     └─ Moving: {rel_name} → {new_subdir_name}/{rel_name}")
            
            if not dry_run:
                shutil.move(str(item), str(dest))
    
    print("\n" + "=" * 80)
    if dry_run:
        print("DRY RUN COMPLETE - No changes were made")
        print("Set dry_run=False to execute the reorganization")
    else:
        print("REORGANIZATION COMPLETE")

