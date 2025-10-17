"""
SpikeInterface preprocessing utilities.

Provides reusable functions for preprocessing neural recordings,
including filtering, referencing, and bad channel detection.
"""
import pandas as pd
from pathlib import Path
import spikeinterface.preprocessing as spre


def preprocess_recordings(rec_ap, rec_lfp):
    """
    Apply standard preprocessing to AP and LFP recordings.
    
    Parameters
    ----------
    rec_ap : RecordingExtractor
        High-frequency AP band recording
    rec_lfp : RecordingExtractor
        Low-frequency LFP band recording
        
    Returns
    -------
    rec_ap_ref : RecordingExtractor
        Preprocessed AP recording (centered, common-reference)
    rec_lfp_f : RecordingExtractor
        Preprocessed LFP recording (common-reference, bandpass filtered)
    """
    rec_ap_c = spre.center(rec_ap, mode='median')
    rec_ap_ref = spre.common_reference(rec_ap_c, reference="global", operator="median")
    rec_lfp_ref = spre.common_reference(rec_lfp, reference="global", operator="median")
    rec_lfp_f = spre.bandpass_filter(rec_lfp_ref, freq_min=0.001, freq_max=300)
    return rec_ap_ref, rec_lfp_f


def slice_to_stable_interval(rec_ap, rec_lfp, start_time, end_time):
    """
    Slice recordings to stable interval (e.g., Kilosort sort times).
    
    Parameters
    ----------
    rec_ap : RecordingExtractor
        AP recording
    rec_lfp : RecordingExtractor
        LFP recording
    start_time : float
        Start time in seconds
    end_time : float
        End time in seconds
        
    Returns
    -------
    rec_ap_stable : RecordingExtractor
        Sliced AP recording
    rec_lfp_stable : RecordingExtractor
        Sliced LFP recording
    """
    lfp_sf = rec_lfp.get_sampling_frequency()
    ap_sf = rec_ap.get_sampling_frequency()
    rec_lfp_stable = rec_lfp.frame_slice(int(start_time * lfp_sf), int(end_time * lfp_sf))
    rec_ap_stable = rec_ap.frame_slice(int(start_time * ap_sf), int(end_time * ap_sf))
    return rec_ap_stable, rec_lfp_stable


def detect_and_save_bad_channels(recording, out_path, method="coherence+psd"):
    """
    Detect bad channels and save results to CSV.
    
    Parameters
    ----------
    recording : RecordingExtractor
        Recording to analyze (typically LFP)
    out_path : Path
        Output directory path
    method : str, optional
        Bad channel detection method (default: "coherence+psd")
        
    Returns
    -------
    bad_ids : list
        List of bad channel IDs
    labels : array
        Channel labels ('good' or 'bad')
    csv_path : Path
        Path to saved CSV file
    """
    bad_ids, labels = spre.detect_bad_channels(recording=recording, method=method)
    
    df = pd.DataFrame({
        "channel_id": recording.get_channel_ids(),
        "label": labels,
        "position_x": recording.get_channel_locations()[:, 0],
        "position_y": recording.get_channel_locations()[:, 1],
    })
    
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "channel_labels.csv"
    df.to_csv(csv_path, index=False)
    
    return bad_ids, labels, csv_path