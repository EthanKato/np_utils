# np_utils/__init__.py
"""
np_utils: A collection of reusable neurophysiology utility functions.

This package provides tools for:
- Job submission and queue management (job_utils)
- Google Sheets oversight and metadata tracking (oversight_utils)  
- SpikeInterface processing pipelines (spikeinterface)
- NWB batch creation and conversion (nwbmaker)
- Motion correction analysis and batch processing (motioncorrection)
- Common parsing and formatting utilities (core)
"""

__version__ = "0.1.0"

# Import commonly used core functions directly
from .core import (
    parse_rec_id, 
    parse_sheet_trange,
    find_neural_binaries,
    find_all_neural_binaries,
    read_stable_range,
    get_stream_id,
    extract_probe_from_path,
)

# Import oversight functions
from .oversight_utils import (
    get_need_nwb, 
    get_rec_ids,
    get_has_nwb,
    validate_sort_times,
    load_sorting_config
)

# Import job submission functions
from .job_utils import (
    submit_job, 
    batch_submit, 
    get_running_jobs, 
    submit_queue_throttled, 
    submit_rec_queue
)

# Import submodules
from . import core
from . import oversight_utils
from . import job_utils
from . import spikeinterface
from . import nwbmaker
from . import motioncorrection

__all__ = [
    # Version
    '__version__',
    # Core parsing functions
    'parse_rec_id',
    'parse_sheet_trange',
    'find_neural_binaries',
    'find_all_neural_binaries',
    'read_stable_range',
    'get_stream_id',
    'extract_probe_from_path',
    # Oversight functions
    'get_need_nwb',
    'get_rec_ids',
    'get_has_nwb',
    'validate_sort_times',
    'load_sorting_config',
    # Job submission functions
    'submit_job',
    'batch_submit',
    'get_running_jobs',
    'submit_queue_throttled',
    'submit_rec_queue',
    # Submodules
    'core',
    'oversight_utils',
    'job_utils',
    'spikeinterface',
    'nwbmaker',
    'motioncorrection',
]