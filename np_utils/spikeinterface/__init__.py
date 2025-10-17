"""
SpikeInterface processing utilities for Neuropixels data.

This module provides tools for running SpikeInterface analysis pipelines
on Kilosort outputs, including waveform extraction, quality metrics,
and merge suggestions.

Main Functions
--------------
preprocess_recordings : function
    Apply standard preprocessing to recordings
build_sorting_analyzer : function
    Build complete sorting analyzer with extensions
"""

from .preprocessing import (
    preprocess_recordings,
    slice_to_stable_interval,
    detect_and_save_bad_channels,
)
from .analyzers import (
    build_sorting_analyzer,
    compute_and_save_merge_groups,
)
from .io_utils import (
    get_matching_recording,
    resolve_output_path,
)
from .find_pial_surface import (
    splice_recording_to_ends, 
    plot_lfp_heatmap,
    decimate_like_mtracer_fast,
    sort_channels_by_depth,
    plot_lfp_heatmap_plotly,
)

# Remove this line - main() is a CLI entry point, not a library function
# from .run_si_proc import main as run_si_proc

__all__ = [
    # Preprocessing
    'preprocess_recordings',
    'slice_to_stable_interval',
    'detect_and_save_bad_channels',
    # Analyzers
    'build_sorting_analyzer',
    'compute_and_save_merge_groups',
    # I/O
    'get_matching_recording',
    'resolve_output_path',
    'splice_recording_to_ends',
    'plot_lfp_heatmap',
    'decimate_like_mtracer_fast',
    'sort_channels_by_depth',
    'plot_lfp_heatmap_plotly',
]