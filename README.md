# np_utils

A comprehensive collection of reusable utilities for neurophysiology data processing, analysis, and oversight.

> **Purpose**: Stop rewriting the same code over and over. Centralize common functionality for Neuropixels processing, NWB creation, SpikeInterface analysis, and lab workflow management.

## Features

- 🔄 **Job Submission**: Flexible SGE/SLURM queue management with throttling
- 📊 **Oversight Tools**: Google Sheets integration for experiment tracking
- 🧠 **SpikeInterface**: Complete preprocessing and analysis pipelines
- 📈 **LFP Visualization**: Interactive pial surface identification with MTracer-compatible processing
- 📦 **NWB Creation**: Batch NWB file generation with interactive filtering
- 🎯 **Motion Correction**: Multi-algorithm, multi-probe motion correction with GPU support
- 🛠️ **Core Utilities**: Common parsing, path resolution, and data management functions

## 📚 Table of Contents
- [Installation](#installation)
  - [Basic](#basic-installation-core-utilities-only)
  - [Module-Specific](#module-specific-installation)
  - [Full Installation](#full-installation-recommended-for-main-analysis-environment)
- [Quick Start](#quick-start)
- [Package Structure](#-package-structure)
- [API Reference](#-api-reference)
  - [Core Utilities](#core-utilities-np_utilscore)
  - [Oversight Utils](#oversight-utils-np_utilsoversight_utils)
  - [Job Utils](#job-utils-np_utilsjob_utils)
  - [LFP Visualization](#lfp-visualization--pial-surface-identification-np_utilsspikeinterfacefind_pial_surface)
  - [Motion Correction](#motion-correction-module-np_utilsmotioncorrection)
  - [NWB Creation](#nwb-creation-module-np_utilsnwbmaker)
  - [SpikeInterface](#spikeinterface-module-np_utilsspikeinterface)
- [Examples](#examples)
- [Contributing](#contributing)

## Installation

### Basic Installation (core utilities only)
```bash
cd /path/to/np_utils
pip install -e .
```

This installs: `core`, `oversight_utils`, `job_utils` (numpy, pandas, np-sheets)

### Module-Specific Installation

Install only what you need:

```bash
# For NWB creation only
pip install -e ".[nwbmaker]"

# For SpikeInterface processing only
pip install -e ".[spikeinterface]"

# For motion correction only
pip install -e ".[motioncorrection]"

# Combine multiple modules
pip install -e ".[nwbmaker,spikeinterface]"
```

### Full Installation (recommended for main analysis environment)
```bash
pip install -e ".[all]"
```

This installs all module dependencies.

### Development Installation
```bash
pip install -e ".[all,dev]"
```

Adds testing and formatting tools (pytest, black, flake8, jupyter).

## Quick Start

```python
import np_utils as nu

# Get recordings that need NWB files
need_nwb = nu.get_need_nwb()

# Submit jobs with automatic throttling
nu.submit_rec_queue(
    rec_ids=["NP147_B1", "NP147_B2"],
    script="-m np_utils.nwbmaker.np_make_nwb",
    python_executable="/path/to/python",
    queue="mind-batch",
    cores=8,
    memory_gb=32,
    log_dir="/path/to/logs",
    job_prefix="nwb",
    max_concurrent=2
)
```

---

## 📦 Package Structure

```
np_utils/
├── core.py                    # Common utilities (parsing, path resolution, stable ranges)
├── oversight_utils.py         # Google Sheets integration
├── job_utils.py              # SGE/SLURM job submission
├── spikeinterface/           # SpikeInterface processing
│   ├── run_si_proc.py       #   Main CLI entry point
│   ├── preprocessing.py     #   Recording preprocessing
│   ├── analyzers.py         #   Analyzer building & QC
│   ├── io_utils.py          #   Path resolution
│   ├── find_pial_surface.py #   LFP visualization & pial surface ID
│   └── submit_si_jobs.py    #   Batch submission
├── nwbmaker/                 # NWB creation tools
│   ├── np_make_nwb.py       #   Single NWB creation
│   └── np_batch_create.py   #   Batch with filtering
└── motioncorrection/         # Motion correction analysis
    ├── motion_correction.py  #   Main MotionCorrection class
    ├── motion_utils.py       #   Plotting and analysis utilities
    ├── run_mc.py            #   CLI entry point
    └── submit_mc.py         #   Batch submission
```

---

## 📖 API Reference

### Core Utilities (`np_utils.core`)

#### `parse_rec_id(rec_id: str) -> Tuple[str, str]`
Parse recording ID into subject and block.

```python
subject, block = nu.parse_rec_id("NP147_B2")
# Returns: ('NP147', 'B2')
```

#### `parse_sheet_trange(value: str, inf_as_string: bool = True) -> Optional[list]`
Parse time range from Google Sheets string.

```python
nu.parse_sheet_trange('[41, 931]')        # [41, 931]
nu.parse_sheet_trange('[0, Inf]')         # [0, 'inf']
nu.parse_sheet_trange('[0, Inf]', False)  # [0, inf] (float)
```

**Args:**
- `value` (str): String from sheet (e.g., `'[41, 931]'` or `'[0, inf]'`)
- `inf_as_string` (bool): If True, convert infinity to string `'inf'`. If False, use `float('inf')`.

#### `find_neural_binaries(rec_id, source='catgt', probe_id=None, band='ap') -> Optional[str]`
Find preprocessed neural binary files (CatGT or motion-corrected).

```python
# Find CatGT AP file for imec0
ap_path = nu.find_neural_binaries("NP147_B1", source='catgt', probe_id='imec0')

# Find motion-corrected AP file
mc_path = nu.find_neural_binaries("NP147_B1", source='mc', probe_id='imec0')

# Find LF band
lf_path = nu.find_neural_binaries("NP147_B1", source='catgt', probe_id='imec0', band='lf')
```

**Args:**
- `rec_id` (str): Recording ID (e.g., 'NP147_B1')
- `source` (str): 'catgt' or 'mc' (motion-corrected)
- `probe_id` (str, optional): Specific probe (e.g., 'imec0'). Returns first if None.
- `band` (str): 'ap' or 'lf'

**Returns:** Path to binary file or None if not found

#### `find_all_neural_binaries(rec_id, source='catgt', band='ap') -> Dict[str, str]`
Find ALL neural binaries for a recording (multi-probe support).

```python
# Find all probes
probes = nu.find_all_neural_binaries("NP147_B1", source='catgt')
# Returns: {'imec0': '/path/to/imec0.ap.bin', 'imec1': '/path/to/imec1.ap.bin'}

# Process each probe
for probe_id, ap_path in probes.items():
    print(f"Processing {probe_id}: {ap_path}")
```

**Returns:** Dictionary mapping probe_id → file_path

#### `read_stable_range(rec_id, config_path=..., fallback_to_sheet=True) -> Tuple[float, float]`
Read stable recording range (sort times) with smart fallback.

```python
# Reads from config, falls back to sheet, then default
t0, t1 = nu.read_stable_range("NP147_B1")
# Returns: (100.0, 1250.0)
```

**Priority:**
1. `sorting_config.json` (if exists)
2. Google Sheets 'Sort time' column (if fallback enabled)
3. Default: `(0.0, inf)`

**Args:**
- `rec_id` (str): Recording ID
- `config_path` (str): Path to sorting_config.json
- `fallback_to_sheet` (bool): Try sheet if config fails

**Returns:** Tuple of (start_time, end_time) in seconds

#### `get_stream_id(probe_id, band='ap') -> str`
Construct SpikeGLX stream ID from probe and band.

```python
nu.get_stream_id('imec0', 'ap')  # 'imec0.ap'
nu.get_stream_id('imec1', 'lf')  # 'imec1.lf'
```

#### `extract_probe_from_path(path) -> Optional[str]`
Extract probe ID (imec0, imec1, etc.) from a file path.

```python
path = "/path/to/NP147_B1_g0_tcat.imec0.ap.bin"
probe = nu.extract_probe_from_path(path)  # 'imec0'
```

---

### Oversight Utils (`np_utils.oversight_utils`)

Functions for tracking experiment metadata via Google Sheets.

#### `get_rec_ids(column_name, condition, recordings_df=None)`
Get recording IDs matching a condition.

```python
# Simple: recordings where NWB column is empty
need_nwb = nu.get_rec_ids('NWB', '')

# Complex: recordings with sort time > 100 seconds
long_sorts = nu.get_rec_ids('Sort time', lambda col: col > 100)
```

**Args:**
- `column_name` (str): Column to check
- `condition`: Value to match OR callable returning boolean mask
- `recordings_df` (pd.DataFrame, optional): Dataframe (loads from sheet if None)

**Returns:** `np.ndarray` of recording IDs (`'Subject_Block'` format)

#### `get_need_nwb(recordings_df=None)`
Shortcut for recordings that need NWB files (where NWB column is empty).

#### `get_has_nwb(recordings_df=None)`
Shortcut for recordings that have NWB files (where NWB column is not empty).

#### `validate_sort_times(config_path=None, recordings_df=None)`
Validate sorting time ranges between JSON config and Google Sheets.

```python
results = nu.validate_sort_times()
print(f"Matches: {results['matches']}/{results['total']}")
for mismatch in results['mismatches']:
    print(f"{mismatch['rec_id']}: config={mismatch['config']}, sheet={mismatch['sheet']}")
```

**Returns:** Dict with keys `'total'`, `'matches'`, `'mismatches'`, `'missing_in_sheet'`, `'errors'`

#### `load_sorting_config(config_path=None)`
Load sorting configuration from JSON file.

---

### Job Utils (`np_utils.job_utils`)

Flexible job submission for SGE/SLURM clusters.

#### `submit_job(script, rec_ids=None, ...)`
Submit a job (or batch of jobs) to queue.

```python
# Single job
nu.submit_job(
    script="process.py",
    queue="mind-batch",
    cores=8,
    memory_gb=32
)

# Batch with recording IDs
nu.submit_job(
    script="-m np_utils.spikeinterface.run_si_proc",  # Run as module
    rec_ids=["NP147_B1", "NP147_B2"],
    python_executable="/path/to/python",
    queue="mind-batch",
    cores=9,
    memory_gb=256,
    job_prefix="si"
)
```

**Key Args:**
- `script` (str | Path): Script path OR `"-m module.name"` for module execution
- `rec_ids` (List[str], optional): If provided, submits one job per rec_id with `--rec-id` flag
- `python_executable` (str): Python path (or use `sys.executable`)
- `queue` (str): Queue name (`'mind-batch'`, `'skull-gpu'`, etc.)
- `cores` (int): CPU cores to request
- `memory_gb` (int): Memory in GB
- `log_dir` (str | Path): Directory for job logs
- `job_prefix` (str): Job name prefix
- `use_time` (bool): Wrap with `/usr/bin/time -v` for profiling
- `gpus` (int, optional): Number of GPUs (for GPU queues)
- `dry_run` (bool): Print command without executing

#### `submit_rec_queue(rec_ids, script, ..., max_concurrent=2)`
Submit recording queue with automatic throttling.

```python
nu.submit_rec_queue(
    rec_ids=["NP147_B1", "NP147_B2", "NP149_B1"],
    script="-m np_utils.spikeinterface.run_si_proc",
    python_executable="/path/to/python",
    queue="mind-batch",
    cores=9,
    memory_gb=256,
    log_dir="/logs",
    job_prefix="SI",
    use_time=True,
    max_concurrent=2,  # Only 2 jobs at a time
    check_interval=30  # Check every 30 seconds
)
```

Monitors running jobs and submits new ones as capacity becomes available.

#### `get_running_jobs()`
Get currently running job count and names.

```python
count, names = nu.get_running_jobs()
print(f"{count} jobs running: {names}")
```

#### `submit_queue_throttled(items, submit_func, max_concurrent=2, ...)`
General-purpose throttled submission for any items.

---

### SpikeInterface Module (`np_utils.spikeinterface`)

Complete SpikeInterface processing pipeline for Neuropixels data.

#### Command-Line Usage

```bash
# Process a single recording
python -m np_utils.spikeinterface.run_si_proc --rec-id NP147_B1

# Or submit batch jobs
python -m np_utils.spikeinterface.submit_si_jobs
```

#### Library Usage

```python
from np_utils.spikeinterface import (
    preprocess_recordings,
    build_sorting_analyzer,
    detect_and_save_bad_channels
)

# Preprocess recordings
rec_ap_ref, rec_lfp_f = preprocess_recordings(rec_ap, rec_lfp)

# Build analyzer with all extensions
analyzer = build_sorting_analyzer(
    sorting=sorting,
    recording=rec_ap_ref,
    out_path="/output/path",
    sparse=True,
    n_jobs=8
)

# Detect bad channels
bad_ids, labels, csv_path = detect_and_save_bad_channels(
    recording=rec_lfp_f,
    out_path="/output/path"
)
```

**What It Does:**
1. Loads NWB file and resolves paths
2. Preprocesses AP/LFP recordings (filtering, referencing)
3. Slices to stable interval (Kilosort sort times)
4. Detects bad channels
5. Builds sorting analyzer with extensions (waveforms, quality metrics, ACGs)
6. Computes merge suggestions
7. Updates Google Sheets with completion status

---

### NWB Maker Module (`np_utils.nwbmaker`)

Tools for batch NWB file creation.

#### Command-Line Usage

```bash
# Create single NWB
python -m np_utils.nwbmaker.np_make_nwb \
    --file-path /output/dir \
    --rec-id NP147_B1 \
    --include-ap

# Interactive batch creation
python -m np_utils.nwbmaker.np_batch_create
```

#### Batch Script Customization

Edit `np_utils/nwbmaker/np_batch_create.py` to set:
- `REC_IDS`: List of recordings to process
- `QUEUE`: SGE queue to use
- `CORES`, `MEM_GB`: Resource allocation
- `INCLUDE_AP`: Include AP band data

The script will:
1. Check if NWB files already exist
2. Prompt for confirmation on each recording
3. Submit approved jobs with throttling

---

### LFP Visualization & Pial Surface Identification (`np_utils.spikeinterface.find_pial_surface`)

Interactive LFP visualization tools for identifying the pial surface in Neuropixels recordings. Emulates MTracer's approach with Python-based filtering and plotly visualization.

#### Features
- ✅ MTracer-compatible decimation (10th-order elliptic filter + 50x downsampling)
- ✅ Interactive plotly heatmaps with zoom/pan
- ✅ Channel sorting by depth for proper spatial visualization
- ✅ Optional splicing to recording endpoints for fast preview
- ✅ Temporal/spatial binning for large datasets
- ✅ **Peak detection and overlay** for enhanced pial surface identification

#### Quick Start

```python
import np_utils as nu

# Find LFP and AP binaries
lf_dict = nu.find_all_neural_binaries("NP156_B1", source='catgt', band='lf')
ap_dict = nu.find_all_neural_binaries("NP156_B1", source='catgt', band='ap')

# Visualize LFP for pial surface identification
nu.spikeinterface.plot_lfp_heatmap(
    lfp_path=lf_dict['imec0'],
    ap_path=ap_dict['imec0'],     # Optional: for peak detection
    title="LFP - NP156_B1 imec0",
    splice_to_ends=True,           # Fast preview
    detect_peaks=True,             # Show spike peaks overlay
    verbose=True
)
```

#### Main Functions

**`plot_lfp_heatmap(lfp_path, ap_path=None, title=None, splice_to_ends=False, detect_peaks=False, peak_threshold=5.0, verbose=False)`**

Main entry point for LFP visualization. Loads LFP binary, applies MTracer-style filtering and decimation, and displays as interactive heatmap. Optionally detects and overlays spike peaks from AP band to help identify the pial surface.

```python
from np_utils.spikeinterface import plot_lfp_heatmap

# Full recording, no peaks
plot_lfp_heatmap("path/to/recording.lf.bin")

# Quick preview with peak detection
plot_lfp_heatmap(
    lfp_path="path/to/recording.lf.bin",
    ap_path="path/to/recording.ap.bin",  # Required for peak detection
    splice_to_ends=True,                 # Only processes first/last 50s
    detect_peaks=True,                   # Show spike peaks overlay
    peak_threshold=5.0,                  # Detection threshold (MAD units)
    title="LFP with Peaks"
)
```

**Args:**
- `lfp_path` (str | Path): Path to LFP binary (.lf.bin)
- `ap_path` (str | Path, optional): Path to AP binary (.ap.bin) for peak detection
- `title` (str, optional): Custom plot title
- `splice_to_ends` (bool): If True, only processes start/end segments (default: False)
- `detect_peaks` (bool): If True, detect and overlay spike peaks (default: False, requires ap_path)
- `peak_threshold` (float): Detection threshold in MAD units (default: 5.0)
- `verbose` (bool): Print progress messages
- `max_peaks` (int): Maximum number of peaks to detect and display (randomly subsampled) (default: 50000)
- `buffer_seconds` (int): Buffer time in seconds around stable range splicing (default: 50)
- `peak_alpha` (float): Transparency of peak markers (default: 0.4)
- `peak_size` (float): Size of peak markers in pixels (default: 3.0)


**`decimate_like_mtracer_fast(data, r=50, n=None)`**

Apply MTracer-style filtering and decimation to LFP data. Uses 10th-order elliptic IIR lowpass filter with bidirectional filtering (zero-phase).

```python
from np_utils.spikeinterface import decimate_like_mtracer_fast

# Load raw LFP
lfp = recording.get_traces()  # Shape: (8.2M, 384) at 2500 Hz

# Decimate 50x: 2500 Hz → 50 Hz
lfp_decimated = decimate_like_mtracer_fast(lfp, r=50)
# Shape: (164k, 384) at 50 Hz
```

**Filter Specifications:**
- Order: 10th-order elliptic IIR
- Passband ripple: 0.01 dB
- Stopband attenuation: 80 dB
- Cutoff: 25 Hz (for 2500 Hz input, r=50)
- Direction: Forward-backward (zero-phase)

**`plot_lfp_heatmap_plotly(lfp, fs, time_bin_s=0.1, chan_bin=1, depths_um=None, clip_pct=99.5, title=None, peak_times=None, peak_depths=None, peak_alpha=0.5, peak_size=12.0)`**

Lower-level function for creating plotly heatmaps with custom binning and optional peak overlay.

```python
from np_utils.spikeinterface import plot_lfp_heatmap_plotly, detect_peaks_for_visualization

# High-resolution plot without peaks
fig = plot_lfp_heatmap_plotly(
    lfp=lfp_data,
    fs=50,
    time_bin_s=0,  # No temporal binning
    chan_bin=1,    # All channels
    depths_um=depths
)
fig.show()

# With peak overlay
peak_times, _, peak_depths, _ = detect_peaks_for_visualization(recording)
fig = plot_lfp_heatmap_plotly(
    lfp=lfp_data,
    fs=50,
    depths_um=depths,
    peak_times=peak_times,     # Overlay detected peaks
    peak_depths=peak_depths,
    peak_size=12.0,            # Marker size in pixels
    peak_alpha=0.5             # Transparency
)
fig.show()

# Fast preview with aggressive binning
fig = plot_lfp_heatmap_plotly(
    lfp=lfp_data,
    fs=50,
    time_bin_s=1.0,  # 1-second bins
    chan_bin=4,      # Average 4 adjacent channels
    depths_um=depths
)
```

**`splice_recording_to_ends(rec, t0, t1, epsilon=50)`**

Extract and concatenate start/end segments for quick preview.

```python
from np_utils.spikeinterface import splice_recording_to_ends
import spikeinterface.full as si

rec = si.read_spikeglx(folder, stream_id='imec0.lf')
rec_splice = splice_recording_to_ends(rec, t0=100, t1=1200, epsilon=50)
# Contains: [0, 150s] + [1150s, end]
```

**`detect_peaks_for_visualization(rec, detect_threshold=5.5, max_peaks=500000, localize=True, job_kwargs=None, preset='rigid_fast')`**

Detect and localize spike peaks from recording for visualization overlay. Uses SpikeInterface's motion-corrected peak detection pipeline with bandpass filtering (300-6000 Hz).

```python
from np_utils.spikeinterface import detect_peaks_for_visualization
import spikeinterface.full as si

# Load AP band recording
rec = si.read_spikeglx(folder, stream_id='imec0.ap')

# Detect peaks with accurate localization
peak_times, peak_channels, peak_depths, peak_locations = detect_peaks_for_visualization(
    rec,
    detect_threshold=5.0,    # MAD units (lower = more peaks)
    max_peaks=50000,         # Subsample for visualization
    localize=True,           # Accurate depth localization
    preset='rigid_fast'      # Motion detection preset
)

print(f"Detected {len(peak_times)} peaks")
print(f"Time range: {peak_times.min():.1f} - {peak_times.max():.1f} s")
print(f"Depth range: {peak_depths.min():.1f} - {peak_depths.max():.1f} μm")
```

**Args:**
- `rec` (RecordingExtractor): Recording to detect peaks from (typically AP band)
- `detect_threshold` (float): Detection threshold in MAD units (default: 5.5)
- `max_peaks` (int): Maximum peaks to return for visualization (default: 500000)
- `localize` (bool): Use accurate localization vs channel position (default: True)
- `job_kwargs` (dict): Job parameters for parallel processing
- `preset` (str): Motion detection preset (default: 'rigid_fast')

**Returns:**
- `peak_times` (np.ndarray): Peak times in seconds
- `peak_channels` (np.ndarray): Channel indices
- `peak_depths` (np.ndarray): Depth in μm for each peak
- `peak_locations` (np.ndarray): Full localization array with x, y coordinates

#### Example Jupyter Notebook

See `np_utils/examples/view_lfp_example.ipynb` for a complete example.

```python
import np_utils as nu
import np_utils.spikeinterface as nusi

REC_ID = "NP156_B1"

# Find LFP binaries
lf_dict = nu.find_all_neural_binaries(REC_ID, source='catgt', band='lf')
print(lf_dict)  # {'imec0': '/path/to/imec0.lf.bin', ...}

# Plot with splicing for fast preview
nusi.plot_lfp_heatmap(
    lfp_path=lf_dict['imec0'],
    title=f"LFP heatmap for {REC_ID}, imec0",
    splice_to_ends=True
)
```

#### Identifying the Pial Surface

The pial surface typically appears as:
- **Transition in LFP amplitude patterns**: Sharp change in voltage amplitude
- **Clearer signals below surface**: Organized activity in brain tissue
- **More noise/artifacts above surface**: CSF, dura, or outside brain
- **Changes in temporal dynamics**: Different frequency content and synchrony
- **Peak detection pattern**: Organized spiking below pial surface, sparse/noisy above

**Using Peak Overlay:**
When `detect_peaks=True`, spike peaks are overlaid as scatter points on the heatmap. Below the pial surface, peaks form organized patterns following the LFP waves. Above the surface, peaks become sparse, noisy, or absent. This contrast helps identify the exact transition depth.

Use the interactive plotly plot to zoom and pan, toggling the peak overlay (click legend) to identify the pial surface across the recording duration.

#### Technical Notes

**Processing Pipeline:**
1. Load LFP recording via SpikeInterface
2. Optional: Splice to start/end segments (fast preview)
3. Optional: Load AP band and detect peaks (if `detect_peaks=True`)
   - Bandpass filter 300-6000 Hz
   - Detect peaks using locally_exclusive method
   - Localize peaks for accurate depth estimation
   - Subsample to max_peaks for visualization
4. Decimate LFP 50x: 2500 Hz → 50 Hz (MTracer-compatible)
5. Sort channels by depth (y-coordinate)
6. Bin temporally/spatially for rendering
7. Display as interactive heatmap with optional peak overlay

**Performance:**
- Full recording: ~1-3 minutes for 1-hour session
- Spliced (start+end): ~10-30 seconds
- Peak detection: +30-120 seconds (depending on recording length and threshold)
- Memory usage: ~120 MB for raw LFP, ~5 MB after decimation, +50 MB for AP band if detecting peaks

**Dependencies:**
- spikeinterface
- scipy (filter design)
- plotly (interactive plotting)

**Peak Visualization Notes:**
- Peak markers are rendered in screen pixels, not data coordinates
- They maintain constant pixel size regardless of zoom level
- Toggle peaks on/off via legend (click "Detected Peaks")
- Recommended marker size: 8-12 pixels for visibility at all zoom levels
- For dense peak patterns, consider increasing `detect_threshold` or reducing `max_peaks`

---

### Motion Correction Module (`np_utils.motioncorrection`)

Comprehensive motion correction analysis for Neuropixels recordings with multi-probe support.

#### Features
- ✅ Multiple algorithms (Dredge, Kilosort-like, MEDiCINE)
- ✅ Multi-probe support (automatic detection)
- ✅ GPU-accelerated batch processing
- ✅ Automatic visualization generation
- ✅ Smart path resolution (CatGT/MC/NWB)

#### Command-Line Usage

```bash
# Single probe
python -m np_utils.motioncorrection.run_mc \
    --rec-id NP147_B1 \
    --probe-id imec0 \
    --presets dredge medicine_ndb4 \
    --source catgt

# Batch submission (auto-detects all probes)
python -m np_utils.motioncorrection.submit_mc
```

#### Library Usage

```python
from np_utils.motioncorrection import MotionCorrection, submit_motion_jobs

# Single probe analysis
mc = MotionCorrection(
    rec_id="NP147_B1",
    probe_id="imec0",
    out_base="/data_store2/neuropixels/preproc/NP147_B1/motion_traces/imec0"
)
mc.resolve_ap_path(source='catgt')
mc.load_and_preprocess()
mc.run_all(["dredge", "medicine_ndb4"])

# Batch processing (multi-probe, GPU)
submit_motion_jobs(
    rec_ids=["NP147_B1", "NP149_B1"],
    presets=["dredge", "medicine_ndb4"],
    source='catgt',
    queue='mind-gpu',
    gpu_count=1,
    max_concurrent=2
)
```

#### Supported Presets
- `dredge`: Standard Dredge (fast, good quality)
- `dredge_th6`: Dredge with threshold=6 (better for noisy data)
- `dredge_fast`: Fast Dredge variant
- `kilosort_like`: Kilosort-style correction
- `medicine_SI`: MEDiCINE via SpikeInterface
- `medicine_ndb2`, `medicine_ndb4`: External MEDiCINE (2 or 4 depth bins, best quality)

#### Output Structure
```
{out_base}/{rec_id}/motion_traces/{probe_id}/
├── _peaks_cache/              # Cached peak detections
├── dredge/                    # Each preset gets own folder
│   ├── motion/                # Motion data
│   ├── peaks.npy
│   ├── peak_locations.npy
│   ├── dredge_spike_map.png
│   ├── dredge_motion_traces.png
│   └── dredge_corrected_drift_maps.png
└── medicine_ndb4/
    └── ...
```

---

## 📝 Examples

### Example 1: Find and Process Recordings Needing SI

```python
import np_utils as nu

# Find recordings that have NWB but need SI processing
have_nwb = nu.get_rec_ids("NWB", lambda col: col != "")
need_si = nu.get_rec_ids("SI", lambda col: col == "")
run_queue = list(set(have_nwb) & set(need_si))

print(f"Found {len(run_queue)} recordings to process")

# Submit with throttling
nu.submit_rec_queue(
    rec_ids=run_queue,
    script="-m np_utils.spikeinterface.run_si_proc",
    python_executable="/path/to/python",
    queue="mind-batch",
    cores=9,
    memory_gb=256,
    log_dir="/logs/SI",
    job_prefix="SI",
    max_concurrent=3
)
```

### Example 2: Validate Sorting Configuration

```python
import np_utils as nu

# Compare JSON config with Google Sheets
results = nu.validate_sort_times()

print(f"✓ Matches: {results['matches']}/{results['total']}")
print(f"✗ Mismatches: {len(results['mismatches'])}")
print(f"⚠ Missing in sheet: {len(results['missing_in_sheet'])}")

# Show mismatches
for m in results['mismatches']:
    print(f"{m['rec_id']}: config={m['config']}, sheet={m['sheet']}")
```

### Example 3: Motion Correction Batch Processing

```python
import np_utils as nu
from np_utils.motioncorrection import submit_motion_jobs

# Find all probes for recordings
probes = nu.find_all_neural_binaries("NP147_B1", source='catgt')
print(f"Found {len(probes)} probes: {list(probes.keys())}")

# Submit motion correction jobs (auto-detects all probes)
submit_motion_jobs(
    rec_ids=["NP147_B1", "NP149_B1", "NP150_B1"],
    presets=["dredge", "medicine_ndb4"],
    source='catgt',
    queue='mind-gpu',
    gpu_count=1,
    max_concurrent=2  # GPU throttling
)

# Or use in Jupyter for interactive control
from np_utils.motioncorrection import MotionCorrection

mc = MotionCorrection(rec_id="NP147_B1", probe_id="imec0")
mc.resolve_ap_path(source='catgt')
mc.load_and_preprocess()
results = mc.run_all(["dredge", "medicine_ndb4"])
```

### Example 4: Custom Job Submission

```python
import np_utils as nu

def my_submit_func(**kwargs):
    nu.submit_job(
        script="/path/to/my_script.py",
        python_executable="/path/to/python",
        queue="mind-batch",
        cores=4,
        memory_gb=16,
        **kwargs
    )

# Submit custom items with throttling
items = ["item1", "item2", "item3"]
nu.submit_queue_throttled(
    items=items,
    submit_func=my_submit_func,
    max_concurrent=2,
    item_to_args=lambda item: {
        'rec_ids': None,
        'extra_args': ['--input', item],
        'job_prefix': f'job_{item}'
    }
)
```

---

## 🔧 Configuration

### Google Sheets Setup

Requires `np-sheets` package configured with credentials. See [np_sheet_utils](https://github.com/EthanKato/np_sheet_utils) for setup.

### Environment Variables

```bash
# Optional: default Python environment
export NP_PYTHON_ENV="/path/to/conda/envs/myenv/bin/python"

# Optional: default queue
export NP_DEFAULT_QUEUE="mind-batch"
```

---

## 🐛 Troubleshooting

### Job Submission Issues

**Problem**: Jobs not appearing in queue
```bash
# Check queue status
qstat -f

# Verify submit_job command exists
which submit_job
```

**Problem**: Module import errors when running as `-m`
- Ensure package is installed: `pip install -e .`
- Check Python path: `python -c "import np_utils; print(np_utils.__file__)"`

### Import Errors

**Problem**: `ImportError: No module named np_utils`
```bash
# Install in editable mode
cd /path/to/np_utils
pip install -e .

# Verify installation
python -c "import np_utils"
```

### SpikeInterface Issues

**Problem**: Memory errors during processing
- Increase `memory_gb` in job submission
- Reduce `n_jobs` parameter
- Use `sparse=True` for analyzer

---

## 🤝 Contributing

This is a personal utility package, but if you find it useful:
1. Fork and adapt for your needs
2. Report issues or suggestions
3. Maintain your own fork

---

## 📄 License

Internal lab use. See repository for details.

---

## 👤 Contact

Ethan Kato - [GitHub](https://github.com/EthanKato)

---

## 🙏 Acknowledgments

Built on top of:
- [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) - Neural data processing and motion correction
- [MEDiCINE](https://github.com/int-brain-lab/ibllib) - Advanced motion correction algorithm
- [NWBMaker](https://github.com/EthanKato/np_nwbmaker) - NWB file creation
- [np-sheets](https://github.com/EthanKato/np_sheet_utils) - Google Sheets integration
- [PyNapple](https://github.com/pynapple-org/pynapple) - Neural data analysis
- [NeuroConv](https://github.com/catalystneuro/neuroconv) - Neurophysiology data conversion
