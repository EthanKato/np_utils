from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.core import get_noise_levels, fix_job_kwargs
from spikeinterface.preprocessing import bandpass_filter, motion

import numpy as np

def detect_peaks_for_visualization(
    rec, 
    detect_threshold: float = 5.5, 
    max_peaks: int = 500000,
    localize: bool = True,
    job_kwargs: dict = None,
    preset: str = 'rigid_fast'
):
    """
    Detect and localize spike peaks for visualization overlay on LFP heatmaps.
    
    This function applies bandpass filtering (300-6000 Hz), detects peaks using
    SpikeInterface's locally_exclusive method, and optionally localizes them for
    accurate depth estimation. Results are subsampled if needed for visualization.
    
    Used primarily for overlaying spike activity on LFP heatmaps to help identify
    the pial surface. Below the surface, peaks form organized patterns. Above,
    they become sparse or noisy.
    
    Parameters
    ----------
    rec : RecordingExtractor
        Recording to detect peaks from (typically AP band at 30 kHz)
    detect_threshold : float, optional
        Detection threshold in MAD (median absolute deviation) units
        Default: 5.5 (fairly stringent)
        Lower values (3-4) detect more peaks but include more noise
        Higher values (6-8) detect fewer, higher-amplitude peaks
    max_peaks : int, optional
        Maximum number of peaks to return for visualization (default: 500000)
        If more peaks detected, randomly subsample to this number
        Keeps visualization responsive
    localize : bool, optional
        If True, use monopolar_triangulation for accurate depth (default: True)
        If False, use channel y-coordinate (faster but less accurate)
        Localization recommended for pial surface identification
    job_kwargs : dict, optional
        Job parameters for parallel processing
        Default: {'chunk_duration': '1s', 'n_jobs': 8, 'progress_bar': True}
    preset : str, optional
        Motion detection preset for detection/localization parameters
        Default: 'rigid_fast'
        See spikeinterface.preprocessing.motion.motion_options_preset
    
    Returns
    -------
    peak_times : np.ndarray
        Peak times in seconds, shape (n_peaks,)
    peak_channels : np.ndarray
        Channel indices for each peak, shape (n_peaks,)
    peak_depths : np.ndarray
        Depth in μm for each peak, shape (n_peaks,)
        Either localized (if localize=True) or channel y-coordinate
    peak_locations : np.ndarray
        Full localization array with x, y coordinates and other metrics
        Shape (n_peaks,) structured array
        Fields depend on localization method
    
    Examples
    --------
    >>> import spikeinterface.full as si
    >>> from np_utils.spikeinterface import detect_peaks_for_visualization
    >>> 
    >>> # Load AP band
    >>> rec = si.read_spikeglx(folder, stream_id='imec0.ap')
    >>> 
    >>> # Detect with default parameters
    >>> peak_times, peak_chans, peak_depths, peak_locs = detect_peaks_for_visualization(rec)
    >>> 
    >>> # More sensitive detection
    >>> peak_times, _, peak_depths, _ = detect_peaks_for_visualization(
    ...     rec, detect_threshold=3.5, max_peaks=100000
    ... )
    >>> 
    >>> # Fast detection without accurate localization
    >>> peak_times, _, peak_depths, _ = detect_peaks_for_visualization(
    ...     rec, localize=False, job_kwargs={'n_jobs': 1, 'progress_bar': False}
    ... )
    
    Notes
    -----
    - Applies bandpass filter (300-6000 Hz) before detection
    - Uses 'locally_exclusive' detection method from motion preset
    - Localization uses 'monopolar_triangulation' from preset
    - Random subsampling preserves temporal distribution
    - Peak detection parameters come from motion correction presets
    
    See Also
    --------
    plot_lfp_heatmap : Main function that uses this for peak overlay
    plot_lfp_heatmap_plotly : Lower-level plotting with peak overlay
    """
    
    if job_kwargs is None:
        job_kwargs = dict(chunk_duration='1s', n_jobs=8, progress_bar=True)
    job_kwargs = fix_job_kwargs(job_kwargs)

    rec = bandpass_filter(rec, freq_min=300.0, freq_max=6000.0)
    params = motion.motion_options_preset[preset]
    detect_kwargs = dict(detect_threshold=detect_threshold)
    detect_kwargs = dict(params["detect_kwargs"], **detect_kwargs)

    noise_levels = get_noise_levels(rec, return_scaled=False)
    # Detect peaks
    peaks = detect_peaks(recording=rec, 
    noise_levels=noise_levels, 
    pipeline_nodes=None, 
    **job_kwargs, 
    **detect_kwargs)
    
    # Extract times and channels
    fs = rec.get_sampling_frequency()
    peak_times = peaks['sample_index'].astype(float) / fs
    peak_channels = peaks['channel_index']
    
    # Get depths
    if localize:
        # Accurate localization
        localize_peaks_kwargs = dict(params["localize_peaks_kwargs"])
        peak_locations = localize_peaks(rec, peaks, **localize_peaks_kwargs, **job_kwargs)
        peak_depths = peak_locations['y']
    else:
        # Fast: use channel y-coordinate
        channel_locs = rec.get_channel_locations()
        peak_depths = channel_locs[peak_channels, 1]
    
    # Subsample if too many peaks
    if len(peak_times) > max_peaks:
        idx = np.random.choice(len(peak_times), max_peaks, replace=False)
        peak_times = peak_times[idx]
        peak_channels = peak_channels[idx]
        peak_depths = peak_depths[idx]
    
    return peak_times, peak_channels, peak_depths, peak_locations