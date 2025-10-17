"""
Submit motion correction jobs for multiple recordings and probes.

This script automatically detects all probes for each recording and submits
a separate GPU job for each rec_id + probe_id combination.

Usage:
    python -m np_utils.motioncorrection.submit_mc
"""

import sys
from pathlib import Path

# Import from parent package
from ..job_utils import submit_job, submit_queue_throttled
from ..oversight_utils import get_rec_ids
from ..core import find_all_neural_binaries


# ============================================================================
# CONFIGURATION
# ============================================================================

# Recording IDs to process (or use get_rec_ids to pull from sheets)
# Example: REC_IDS = get_rec_ids("NP_motion_correction_queue")
REC_IDS = [
    "NP156_B1"
]

# Motion correction presets to run
# Options: dredge, dredge_fast, dredge_th6, kilosort_like, medicine_SI, medicine_ndbN
PRESETS = ["dredge", "medicine_ndb4", "medicine_nbd2"]

# Preprocessing source ('catgt' or 'mc')
# This should never be 'mc' for motion correction jobs
SOURCE = 'catgt'

# Output base directory (run_mc.py will append /{rec_id}/motion_traces/{probe_id})
OUT_BASE = "/data_store2/neuropixels/preproc"

# Job submission parameters
ENV_PATH = "/userdata/ekato/miniforge3/envs/motioncorrection/bin/python"
QUEUE = "mind-gpu"  # GPU queue
CORES = 8
MEM_GB = 128
GPU_COUNT = 1  # Number of GPUs per job

# Logging
LOGDIR = "/data_store2/neuropixels/nwb/temp/MC_logs"

# Concurrency
MAX_CONCURRENT = 2  # Limit concurrent GPU jobs
CHECK_INTERVAL = 30  # Seconds between job count checks

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def build_probe_job_list(rec_ids, source='catgt'):
    """
    Build a list of (rec_id, probe_id) tuples for job submission.
    
    Args:
        rec_ids: List of recording IDs
        source: Preprocessing source ('catgt' or 'mc')
    
    Returns:
        List of (rec_id, probe_id) tuples
    """
    job_list = []
    
    for rec_id in rec_ids:
        # Find all probes for this recording
        probes = find_all_neural_binaries(rec_id, source=source, band='ap')
        
        if not probes:
            print(f"[WARN] No probes found for {rec_id} (source={source}), skipping...")
            continue
        
        print(f"[INFO] {rec_id}: Found {len(probes)} probe(s): {list(probes.keys())}")
        
        # Add each probe to job list
        for probe_id in probes.keys():
            job_list.append((rec_id, probe_id))
    
    return job_list


def submit_motion_jobs(
    rec_ids,
    presets,
    source='catgt',
    out_base='/data_store2/neuropixels/preproc',
    env_path='/userdata/ekato/miniforge3/envs/se2nwb/bin/python',
    queue='mind-gpu',
    cores=8,
    mem_gb=128,
    gpu_count=1,
    logdir='/data_store2/neuropixels/nwb/temp/MC_logs',
    max_concurrent=2,
    check_interval=30,
):
    """
    Submit motion correction jobs for multiple recordings and probes.
    
    Automatically detects all probes for each recording and submits a separate
    GPU job for each rec_id + probe_id combination.
    
    Args:
        rec_ids: List of recording IDs to process
        presets: List of motion correction presets to run
        source: Preprocessing source ('catgt' or 'mc')
        out_base: Output base directory (will be appended with /{rec_id}/motion_traces/{probe_id})
        env_path: Python executable path (with required packages)
        queue: SGE queue name (should be a GPU queue)
        cores: Number of CPU cores per job
        mem_gb: Memory in GB per job
        gpu_count: Number of GPUs per job
        logdir: Directory for job logs
        max_concurrent: Maximum number of concurrent jobs
        check_interval: Seconds between job status checks
    """
    print("\n" + "="*70)
    print("MOTION CORRECTION JOB SUBMISSION")
    print("="*70)
    print(f"Recordings: {len(rec_ids)}")
    print(f"Presets: {presets}")
    print(f"Source: {source}")
    print(f"Queue: {queue}")
    print(f"GPUs per job: {gpu_count}")
    print(f"Max concurrent: {max_concurrent}")
    print("="*70 + "\n")
    
    # Build list of (rec_id, probe_id) jobs
    job_list = build_probe_job_list(rec_ids, source=source)
    
    if not job_list:
        print("[ERROR] No jobs to submit (no probes found)")
        return
    
    print(f"\n[INFO] Total jobs to submit: {len(job_list)}")
    print("-"*70 + "\n")
    
    # Create log directory
    Path(logdir).mkdir(parents=True, exist_ok=True)
    
    # Define how to convert (rec_id, probe_id) tuple to submit_job kwargs
    def item_to_args(item):
        rec_id, probe_id = item
        job_name = f"MC_{rec_id}_{probe_id}"
        
        script_args = [
            "--rec-id", rec_id,
            "--probe-id", probe_id,
            "--out-base", out_base,
            "--source", source,
            "--presets", *presets,
        ]
        
        return {
            'script': "-m np_utils.motioncorrection.run_mc",
            'python_executable': env_path,
            'queue': queue,
            'cores': cores,
            'memory_gb': mem_gb,
            'job_prefix': job_name,      # Changed from 'job_name'
            'log_dir': logdir,            # Changed from 'log_path' and removed Path/filename
            'use_time': True,
            'gpus': gpu_count,
            'extra_args': script_args,    # Changed from 'script_args'
        }
    
    # Submit jobs with throttling
    submit_queue_throttled(
        items=job_list,
        submit_func=lambda **kwargs: submit_job(**kwargs),
        max_concurrent=max_concurrent + 10,
        check_interval=check_interval,
        item_to_args=item_to_args,
        verbose=True,
    )
    
    print("\n" + "="*70)
    print("All jobs submitted!")
    print("="*70)
    print(f"\nMonitor with: qstat -f")
    print(f"View logs in: {logdir}")
    print(f"\nJob naming: MC_{{rec_id}}_{{probe_id}}")
    print(f"Example: MC_{job_list[0][0]}_{job_list[0][1]}")
    print("\nOutput structure: {out_base}/{rec_id}/motion_traces/{probe_id}/")
    print(f"Example: {out_base}/{job_list[0][0]}/motion_traces/{job_list[0][1]}/")


def main():
    """Main entry point for script."""
    # Validate configuration
    if not REC_IDS:
        print("[ERROR] No recording IDs specified. Update REC_IDS in the configuration section.")
        sys.exit(1)
    
    if not PRESETS:
        print("[ERROR] No presets specified. Update PRESETS in the configuration section.")
        sys.exit(1)
    
    # Submit jobs
    submit_motion_jobs(
        rec_ids=REC_IDS,
        presets=PRESETS,
        source=SOURCE,
        out_base=OUT_BASE,
        env_path=ENV_PATH,
        queue=QUEUE,
        cores=CORES,
        mem_gb=MEM_GB,
        gpu_count=GPU_COUNT,
        logdir=LOGDIR,
        max_concurrent=MAX_CONCURRENT,
        check_interval=CHECK_INTERVAL,
    )


if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
Example 1: Pull recordings from Google Sheets
----------------------------------------------
from ..oversight_utils import get_rec_ids

REC_IDS = get_rec_ids("motion_correction_queue")
PRESETS = ["dredge", "medicine_ndb4"]
SOURCE = 'catgt'


Example 2: Process specific recordings
--------------------------------------
REC_IDS = ["NP147_B1", "NP149_B1", "NP150_B1"]
PRESETS = ["kilosort_like", "dredge_th6"]
SOURCE = 'mc'  # Use motion-corrected binaries


Example 3: Full MEDiCINE comparison
-----------------------------------
REC_IDS = ["NP147_B1"]
PRESETS = [
    "dredge",
    "dredge_th6",
    "kilosort_like",
    "medicine_SI",
    "medicine_ndb2",
    "medicine_ndb4",
]
SOURCE = 'catgt'


Example 4: Import and use programmatically
------------------------------------------
from np_utils.motioncorrection.submit_mc import submit_motion_jobs

submit_motion_jobs(
    rec_ids=["NP147_B1", "NP149_B1"],
    presets=["dredge", "medicine_ndb4"],
    source='catgt',
    out_base='/data_store2/neuropixels/preproc',
    queue='mind-gpu',
    gpu_count=1,
    max_concurrent=2,
)


Example 5: High throughput (more concurrent jobs)
-------------------------------------------------
REC_IDS = ["NP147_B1", "NP149_B1", "NP150_B1", "NP151_B1"]
PRESETS = ["dredge"]
MAX_CONCURRENT = 4  # Run up to 4 GPUs simultaneously
"""