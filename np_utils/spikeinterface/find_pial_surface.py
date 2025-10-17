"""
LFP visualization and pial surface identification utilities.

This module provides tools for visualizing LFP data to identify the pial surface
in Neuropixels recordings. It emulates MTracer's approach by filtering and decimating
LFP data, then displaying it as an interactive heatmap.

Key Features
------------
- MTracer-compatible LFP decimation (10th-order elliptic filter + downsampling)
- Interactive plotly-based heatmap visualization
- Optional splicing to recording endpoints for faster preview
- Channel sorting by depth for proper visualization

Main Functions
--------------
plot_lfp_heatmap : function
    Main entry point for LFP visualization
decimate_like_mtracer_fast : function
    Apply MTracer-style filtering and decimation
splice_recording_to_ends : function
    Extract start and end segments of recording
"""

import spikeinterface.full as si
import spikeinterface.preprocessing as spre
from spikeinterface import concatenate_recordings
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from typing import Union
from pathlib import Path

from ..core import get_neural_binary_folder, parse_binary_path, read_stable_range, extract_rec_id_from_path


def splice_recording_to_ends(rec, t0, t1, epsilon=50):
    """
    Extract and concatenate start and end segments of a recording.
    
    Useful for quickly visualizing LFP across the full recording duration
    without processing the entire stable period. Takes epsilon seconds from
    the start and end, providing a preview of the full recording.
    
    Parameters
    ----------
    rec : RecordingExtractor
        SpikeInterface recording object
    t0 : float
        Start time of stable period (seconds)
    t1 : float
        End time of stable period (seconds)
    epsilon : float, optional
        Duration in seconds to extract from each end (default: 50)
    
    Returns
    -------
    concatenated_rec : RecordingExtractor
        Concatenated recording with [0, t0+epsilon] + [t1-epsilon, end]
    
    Examples
    --------
    >>> rec = si.read_spikeglx(folder, stream_id='imec0.lf')
    >>> rec_splice = splice_recording_to_ends(rec, 100, 1200, epsilon=50)
    >>> # Now rec_splice contains first 150s and last 50s of recording
    """
    rec1 = rec.time_slice(0, t0 + epsilon)
    rec2 = rec.time_slice(t1 - epsilon, None)
    return concatenate_recordings([rec1, rec2])

def decimate_like_mtracer_fast(data, r=50, n=None):
    """
    Apply MTracer-style filtering and decimation to LFP data.
    
    Emulates MTracer's MMath.Decimate function using a 10th-order elliptic
    IIR lowpass filter followed by simple decimation. The filter is applied
    bidirectionally (zero-phase) using filtfilt, matching MTracer exactly.
    
    Filter Specifications
    ---------------------
    - Order: 10th-order elliptic IIR
    - Passband ripple: 0.01 dB
    - Stopband attenuation: 80 dB
    - Cutoff frequency: fs/(2*r) (Nyquist of output rate)
    - Direction: Forward-backward (zero-phase)
    
    For typical Neuropixels LFP (fs=2500 Hz, r=50):
    - Cutoff: 25 Hz
    - Output rate: 50 Hz
    
    Parameters
    ----------
    data : np.ndarray
        Input LFP data, shape (time, channels)
    r : int, optional
        Decimation factor (default: 50)
        New sampling rate = original_fs / r
    n : int, optional
        Sample offset for downsampling (default: r//2)
        Takes samples at indices [n, n+r, n+2r, ...]
        Using r//2 centers samples within each bin
    
    Returns
    -------
    decimated : np.ndarray
        Filtered and decimated data, shape (time//r, channels)
    
    Notes
    -----
    This function processes all channels simultaneously (vectorized) for
    speed, unlike MTracer which loops over channels. Results are identical
    but processing is much faster.
    
    Examples
    --------
    >>> lfp = raw_rec.get_traces()  # Shape: (8.2M, 384)
    >>> lfp_dec = decimate_like_mtracer_fast(lfp, r=50)  # Shape: (164k, 384)
    >>> # Original: 2500 Hz, Decimated: 50 Hz
    
    See Also
    --------
    plot_lfp_heatmap : Main function for LFP visualization
    
    References
    ----------
    Based on MTracer's MMath.Decimate:
    https://github.com/yaxigeigei/MTracer
    """
    if n is None:
        n = r // 2
    
    # Design 10th-order elliptic lowpass filter
    # Wn=1/r gives normalized cutoff at Nyquist frequency of output rate
    sos = signal.ellip(
        N=10,           # Filter order
        rp=0.01,        # Passband ripple (dB)
        rs=80,          # Stopband attenuation (dB)
        Wn=1/r,         # Normalized cutoff frequency
        btype='low',
        output='sos'    # Second-order sections for numerical stability
    )
    
    # Apply zero-phase filtering (forward-backward pass)
    filtered = signal.sosfiltfilt(sos, data, axis=0)
    
    # Downsample by taking every r-th sample starting at offset n
    decimated = filtered[n::r, :]
    
    return decimated


def sort_channels_by_depth(data, rec):
    """
    Sort LFP data by channel depth for proper visualization.
    
    Channels must be sorted by y-coordinate (depth) before displaying
    in a heatmap to ensure proper spatial alignment.
    
    Parameters
    ----------
    data : np.ndarray
        LFP data, shape (time, channels)
    rec : RecordingExtractor
        Recording object with channel location information
    
    Returns
    -------
    sorted_data : np.ndarray
        Data with channels sorted by depth
    sorted_depths : np.ndarray
        Depth values in μm, sorted
    
    Examples
    --------
    >>> lfp_sorted, depths = sort_channels_by_depth(lfp_data, recording)
    """
    channel_locs = rec.get_channel_locations()
    depths = channel_locs[:, 1]  # Y-coordinate (depth in μm)
    sort_idx = np.argsort(depths)
    return data[:, sort_idx], depths[sort_idx]

def plot_lfp_heatmap_plotly(
    lfp,
    fs,
    time_bin_s=0.10,
    chan_bin=1,
    depths_um=None,
    clip_pct=99.5,
    title=None
):
    """
    Create interactive plotly heatmap of LFP data with optional binning.
    
    Generates a plotly figure showing LFP voltage across channels (y-axis)
    and time (x-axis). Supports temporal and spatial binning to reduce
    data size for faster rendering of large datasets.
    
    Parameters
    ----------
    lfp : np.ndarray
        LFP data in microvolts, shape (time, channels)
    fs : float
        Sampling frequency of LFP data (Hz)
    time_bin_s : float, optional
        Time bin width in seconds for averaging (default: 0.1)
        Set to 0 or very small value to skip temporal binning
    chan_bin : int, optional
        Number of adjacent channels to average together (default: 1)
        Set to 1 to use all channels
    depths_um : np.ndarray, optional
        Channel depths in micrometers, shape (channels,)
        If None, uses channel indices
    clip_pct : float, optional
        Percentile for robust color scaling (default: 99.5)
        Uses symmetric limits around 0
    title : str, optional
        Figure title
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure
    
    Notes
    -----
    - Temporal binning uses numpy's reduceat for efficient computation
    - Spatial binning averages adjacent channels
    - Color scale is centered at 0 with RdBu colormap
    - For large datasets (>100k timepoints), consider binning
    
    Examples
    --------
    >>> # Basic usage
    >>> fig = plot_lfp_heatmap_plotly(lfp, fs=50, depths_um=depths)
    >>> fig.show()
    
    >>> # High resolution (no binning)
    >>> fig = plot_lfp_heatmap_plotly(lfp, fs=50, time_bin_s=0, 
    ...                               chan_bin=1, depths_um=depths)
    
    >>> # Aggressive binning for fast preview
    >>> fig = plot_lfp_heatmap_plotly(lfp, fs=50, time_bin_s=1.0,
    ...                               chan_bin=4, depths_um=depths)
    """
    T, C = lfp.shape
    
    # ----- Temporal binning (mean over time bins) -----
    step = max(1, int(round(time_bin_s * fs)))
    edges = np.arange(0, T + step, step)
    edges[-1] = T
    
    # Use reduceat for efficient binning (handles ragged last bin)
    sums = np.add.reduceat(lfp.astype(np.float32), edges[:-1], axis=0)
    counts = np.diff(edges)[:, None].astype(np.float32)
    lfp_tb = sums / counts  # Shape: (n_time_bins, channels)
    
    # Bin centers in seconds
    t_bins = (edges[:-1] + np.diff(edges)/2) / fs

    # ----- Spatial binning (optional) -----
    if chan_bin > 1:
        keep = (C // chan_bin) * chan_bin
        lfp_tb = lfp_tb[:, :keep].reshape(lfp_tb.shape[0], -1, chan_bin).mean(axis=2)
        if depths_um is not None:
            d = np.asarray(depths_um)[:keep].reshape(-1, chan_bin).mean(axis=1)
        else:
            d = np.arange(keep).reshape(-1, chan_bin).mean(axis=1)
    else:
        d = np.asarray(depths_um) if depths_um is not None else np.arange(C)

    # Transpose for plotting: channels on Y-axis, time on X-axis
    Z = lfp_tb.T  # Shape: (n_channels_binned, n_time_bins)

    # Robust color limits centered at 0 (for RdBu diverging colormap)
    v = np.nanpercentile(Z, [100-clip_pct, clip_pct])
    vmax = float(max(abs(v[0]), abs(v[1])))

    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=t_bins,
        y=d,
        colorscale='RdBu',
        zmid=0.0,
        zmin=-vmax, zmax=vmax,
        colorbar=dict(title="Voltage (µV)"),
        showscale=True
    ))
    fig.update_layout(
        title=title or f"LFP Heatmap (bin={time_bin_s*1000:.0f} ms, chan_bin={chan_bin})",
        xaxis_title="Time (s)",
        yaxis_title="Distance from probe tip (µm)" if depths_um is not None else "Channel",
        height=700
    )
    return fig


def plot_lfp_heatmap(
    lfp_path: Union[str, Path], 
    title: str = None, 
    splice_to_ends: bool = False, 
    verbose: bool = False
):
    """
    Load, process, and visualize LFP data for pial surface identification.
    
    Main entry point for LFP visualization. Loads LFP binary, applies
    MTracer-style filtering and decimation, sorts channels by depth, and
    displays as an interactive plotly heatmap.
    
    The pial surface typically appears as a transition in LFP amplitude and
    patterns, with clearer signals below the surface and more noise above.
    
    Parameters
    ----------
    lfp_path : str or Path
        Path to LFP binary file (.lf.bin)
        Can be full path or relative path that works with core utilities
    title : str, optional
        Custom title for the plot
        If None, uses default title with recording info
    splice_to_ends : bool, optional
        If True, only processes start/end segments of recording (default: False)
        Useful for quick preview of full recording duration
        Uses stable range from config/sheets to determine splice points
    verbose : bool, optional
        Print progress messages (default: False)
    
    Returns
    -------
    None
        Displays interactive plotly figure in browser or notebook
    
    Notes
    -----
    Processing Pipeline:
    1. Load LFP recording via SpikeInterface
    2. Optional: Splice to start/end segments
    3. Decimate 50x: 2500 Hz → 50 Hz (MTracer-compatible)
    4. Sort channels by depth (y-coordinate)
    5. Bin temporally/spatially for faster rendering
    6. Display as interactive heatmap
    
    The decimation uses a 10th-order elliptic filter to prevent aliasing,
    exactly matching MTracer's approach.
    
    Examples
    --------
    >>> # Basic usage - full recording
    >>> from np_utils.spikeinterface import plot_lfp_heatmap
    >>> plot_lfp_heatmap("path/to/recording.lf.bin")
    
    >>> # Quick preview with splicing
    >>> plot_lfp_heatmap("path/to/recording.lf.bin", 
    ...                  splice_to_ends=True,
    ...                  title="Quick preview")
    
    >>> # From core utilities
    >>> import np_utils as nu
    >>> lf_dict = nu.find_all_neural_binaries("NP156_B1", band='lf')
    >>> nu.spikeinterface.plot_lfp_heatmap(lf_dict['imec0'])
    
    See Also
    --------
    decimate_like_mtracer_fast : Filtering and decimation details
    plot_lfp_heatmap_plotly : Lower-level plotting function
    splice_recording_to_ends : Segment extraction
    
    References
    ----------
    Based on MTracer LFP visualization:
    https://github.com/yaxigeigei/MTracer
    """
    if verbose:
        print("Loading LFP data and decimating...")
    raw_rec = si.read_spikeglx(
        folder_path= get_neural_binary_folder(lfp_path),
        stream_id= parse_binary_path(lfp_path)['stream_id']
    )
    raw_rec.shift_times(-raw_rec.get_start_time())

    if splice_to_ends:
        REC_ID = extract_rec_id_from_path(lfp_path)
        try:
            t0, t1 = read_stable_range(REC_ID)
        except Exception as e:
            print(f"Warning: Could not read stable range for {REC_ID}: {e}")
            t0, t1 = 0, raw_rec.get_total_duration()
        raw_rec = splice_recording_to_ends(raw_rec, t0, t1, epsilon=50)
    decimation_factor = 50
    lfp = raw_rec.get_traces(return_in_uV=True)
    lfp_decimated = decimate_like_mtracer_fast(lfp, r=decimation_factor)

    fs_new = raw_rec.get_sampling_frequency() / decimation_factor
    time = np.arange(lfp_decimated.shape[0]) / fs_new
    sorted_lfp, sorted_depths = sort_channels_by_depth(lfp_decimated, raw_rec)  # shape (T, C)

    fig = plot_lfp_heatmap_plotly(sorted_lfp, fs_new, title=title, time_bin_s=0.10, chan_bin=1, depths_um=sorted_depths)
    fig.show()


