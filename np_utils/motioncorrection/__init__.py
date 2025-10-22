"""
Motion correction tools for Neuropixels recordings.

Supports multiple motion correction algorithms and multi-probe recordings.
"""

from .motion_correction import MotionCorrection
from .motion_utils import (
    get_peaks_medicine,
    compute_medicine_external,
    load_motion_from_folder,
    save_peak_map,
    save_motion_traces,
    plot_drift_maps_before_after,
    reorganize_motion_traces,
)
from .submit_mc import submit_motion_jobs, build_probe_job_list

__all__ = [
    # Main class
    'MotionCorrection',
    # Batch submission
    'submit_motion_jobs',
    'build_probe_job_list',
    # Utilities
    'get_peaks_medicine',
    'compute_medicine_external',
    'load_motion_from_folder',
    'save_peak_map',
    'save_motion_traces',
    'plot_drift_maps_before_after',
    'reorganize_motion_traces',
]

