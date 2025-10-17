# np_utils

A comprehensive collection of reusable utilities for neurophysiology data processing, analysis, and oversight.

> **Purpose**: Stop rewriting the same code over and over. Centralize common functionality for Neuropixels processing, NWB creation, SpikeInterface analysis, and lab workflow management.

## Features

- 🔄 **Job Submission**: Flexible SGE/SLURM queue management with throttling
- 📊 **Oversight Tools**: Google Sheets integration for experiment tracking
- 🧠 **SpikeInterface**: Complete preprocessing and analysis pipelines
- 📦 **NWB Creation**: Batch NWB file generation with interactive filtering
- 🛠️ **Core Utilities**: Common parsing and formatting functions

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

# For motion correction only (future)
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
├── core.py                    # Common utilities (parsing, formatting)
├── oversight_utils.py         # Google Sheets integration
├── job_utils.py              # SGE/SLURM job submission
├── spikeinterface/           # SpikeInterface processing
│   ├── run_si_proc.py       #   Main CLI entry point
│   ├── preprocessing.py     #   Recording preprocessing
│   ├── analyzers.py         #   Analyzer building & QC
│   ├── io_utils.py          #   Path resolution
│   └── submit_si_jobs.py    #   Batch submission
├── nwbmaker/                 # NWB creation tools
│   ├── np_make_nwb.py       #   Single NWB creation
│   └── np_batch_create.py   #   Batch with filtering
└── motioncorrection/         # (Future: motion correction)
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

### Example 3: Custom Job Submission

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
- [SpikeInterface](https://github.com/SpikeInterface/spikeinterface)
- [NWBMaker](https://github.com/EthanKato/np_nwbmaker)
- [np-sheets](https://github.com/EthanKato/np_sheet_utils)
- [PyNapple](https://github.com/pynapple-org/pynapple)
- [NeuroConv](https://github.com/catalystneuro/neuroconv)
