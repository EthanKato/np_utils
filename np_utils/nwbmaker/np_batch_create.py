"""
Batch NWB creation with interactive filtering and job submission.
"""
import sys
from pathlib import Path
from NWBMaker import NPNWBMaker

# Import from parent package
from ..job_utils import submit_rec_queue

# RECORDING IDS TO PROCESS
REC_IDS = ['NP136_B1', 'NP139_B2', 'NP140_B1', 'NP143_B2', 'NP145_B1', 'NP147_B1', 'NP147_B2']

# QUEUE TO USE (mind-batch best)
QUEUE = "mind-batch"

CORES = 8
MEM_GB = 32
PYTHON = sys.executable # "/userdata/ekato/miniforge3/envs/se2nwb/bin/python"
FILE_PATH = "/data_store2/neuropixels/nwb/temp"
LOG_DIR = "/data_store2/neuropixels/nwb/temp/logs"
USE_TIME = True
INCLUDE_AP = False


def ask_keep(ids):
    """
    Interactively filter recordings before submission.
    
    Skips recordings that already exist and prompts for confirmation
    on each recording that can be resolved.
    
    Args:
        ids: List of recording IDs to filter
        
    Returns:
        List of recording IDs approved for processing
    """
    kept = []
    for rec_id in ids:
        # Check temp location
        if Path(f"/data_store2/neuropixels/nwb/temp/{rec_id}/{rec_id}.nwb").exists():
            print(f"[SKIP] {rec_id} already exists in temp")
            continue
            
        # Check permanent location
        if Path(f"/data_store2/neuropixels/nwb/{rec_id}/{rec_id}.nwb").exists():
            ans = input(f"{rec_id} already exists. Proceed with {rec_id}? (y/n): ").strip().lower()
            if ans != "y":
                print(f"[SKIP] {rec_id}")
                continue
        
        # Try to resolve paths
        try:
            nwb = NPNWBMaker(
                file_path="/data_store2/neuropixels/nwb/temp/",
                rec_id=rec_id,
                make_log_file=False
            )
            nwb.resolve_paths(auto_resolve=True, ks_select_all=True)
            
            ans = input(f"Proceed with {rec_id}? (y/n): ").strip().lower()
            if ans == "y":
                kept.append(rec_id)
                print(f"[KEEP] {rec_id}")
            else:
                print(f"[SKIP] {rec_id}")
                
        except Exception as e:
            print(f"[ERROR resolving {rec_id}]: {e}")
            
    return kept


def main():
    """Submit NWB creation jobs for approved recordings."""
    # Filter recordings interactively
    kept = ask_keep(REC_IDS)
    
    if not kept:
        print("No recordings selected. Exiting.")
        return
    
    print(f"\n{len(kept)} recordings approved for submission")
    
    # Build extra args if needed
    extra_args = ["--include-ap"] if INCLUDE_AP else []
    
    # Submit using job_utils (imported from parent package)
    submit_rec_queue(
        rec_ids=kept,
        script="-m np_utils.nwbmaker.np_make_nwb",  # Run as module
        python_executable=PYTHON,
        queue=QUEUE,
        cores=CORES,
        memory_gb=MEM_GB,
        log_dir=LOG_DIR,
        job_prefix="nwb",
        use_time=USE_TIME,
        max_concurrent=len(kept) + 10,  # Submit all at once (or set lower for throttling) + 10 for already running jobs
        extra_args_per_rec=lambda rec_id: ["--file-path", FILE_PATH] + extra_args,
    )
    
    print("\nAll jobs submitted!")
    print("Monitor with: qstat -f")


if __name__ == "__main__":
    main()