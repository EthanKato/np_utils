"""
Batch submission script for SpikeInterface processing.

Submits SI processing jobs for all recordings that have NWB files
but haven't been processed yet.
"""
import numpy as np
from ..job_utils import submit_rec_queue
from ..oversight_utils import get_rec_ids

ENV_PATH = "/userdata/ekato/miniforge3/envs/se2nwb/bin/python"

def main():
    """Submit SI processing jobs for pending recordings."""
    # Find recordings that have NWB but need SI
    have_nwb = get_rec_ids("NWB", lambda col: col != "")
    need_si = get_rec_ids("SI", lambda col: col == "")
    run_queue = np.intersect1d(have_nwb, need_si).tolist()
    
    print(f"Found {len(run_queue)} recordings to process: {run_queue}")
    
    submit_rec_queue(
        rec_ids=run_queue,
        script="-m np_utils.spikeinterface.run_si_proc",
        python_executable=ENV_PATH,
        queue="mind-batch",
        cores=9,
        memory_gb=256,
        log_dir="/data_store2/neuropixels/nwb/temp/SI_logs",
        job_prefix="SI",
        use_time=True,
        max_concurrent=3,
        check_interval=30
    )


if __name__ == "__main__":
    main()