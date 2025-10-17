"""
SpikeInterface processing pipeline CLI.

Main entry point for running SpikeInterface analysis on NWB files.
"""
import os, sys
import glob
import shutil
import psutil
import argparse
from pathlib import Path
import numpy as np
import pynapple as nap

from NWBMaker import NPNWBMaker
from neuroconv.datainterfaces import SpikeGLXRecordingInterface, KiloSortSortingInterface
import np_sheets as sheet_utils

from .preprocessing import preprocess_recordings, slice_to_stable_interval, detect_and_save_bad_channels
from .analyzers import build_sorting_analyzer, compute_and_save_merge_groups, ts
from .io_utils import get_matching_recording, resolve_output_path


from .preprocessing import preprocess_recordings, slice_to_stable_interval, detect_and_save_bad_channels
from .analyzers import build_sorting_analyzer, compute_and_save_merge_groups, ts
from .io_utils import get_matching_recording, resolve_output_path


def main():
    """Run SpikeInterface processing pipeline."""
    parser = argparse.ArgumentParser(description="Run SpikeInterface processing on Neuropixels data")
    parser.add_argument("--rec-id", required=True, help="Recording ID (e.g., NP153_B1)")
    parser.add_argument("--file-path", required=False, help="Path to NWB file (optional)")
    parser.add_argument("--sparse", action="store_true", default=True, help="Use sparse analyzer")
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of parallel jobs")
    args = parser.parse_args()
    
    ts("Starting SI processing")
    
    # Resolve NWB path
    if args.file_path:
        nwb_path = Path(args.file_path)
    else:
        nwb_path = Path(f"/data_store2/neuropixels/nwb/{args.rec_id}/{args.rec_id}.nwb")
    
    # Initialize NWB maker
    nwb = NPNWBMaker(file_path=nwb_path, rec_id=args.rec_id, make_log_file=False, silent=True)
    nwb.initialize_nwbfile()
    nwb.resolve_paths(auto_resolve=True, ks_select_all=True)
    ts("NWB resolved")
    
    # Load pynapple data once
    data = nap.load_file(nwb_path)
    
    # Caches to avoid redundant computation
    preproc_cache = {}  # imec -> (rec_ap_pre, rec_lfp_pre, rec_ap_stable, rec_lfp_stable)
    bad_cache = {}  # imec -> (bad_ids, labels, csv_path)
    
    ks_paths = nwb.paths['kilosort']
    out_root = nwb_path.parent / "SI"
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Process each Kilosort run
    for ks in ks_paths:
        ts(f"--- KS run: {ks.name} ---")
        
        # Resolve paths
        ap_name, ap_path, ap_stream_id, imec = get_matching_recording(nwb, ks.name, "AP")
        lfp_name, lfp_path, lfp_stream_id, _ = get_matching_recording(nwb, ks.name, "LFP")
        
        if imec is None or ap_path is None or lfp_path is None:
            ts(f"SKIP {ks}: could not resolve AP/LFP paths")
            continue
        
        # Load and preprocess recordings (cached per imec)
        if imec not in preproc_cache:
            ts(f"[{imec}] loading SpikeGLX interfaces")
            rec_ap = SpikeGLXRecordingInterface(
                folder_path=ap_path.parent, stream_id=ap_stream_id
            ).recording_extractor
            rec_lfp = SpikeGLXRecordingInterface(
                folder_path=lfp_path.parent, stream_id=lfp_stream_id
            ).recording_extractor
            
            ts(f"[{imec}] preprocessing")
            rec_ap_pre, rec_lfp_pre = preprocess_recordings(rec_ap, rec_lfp)
            
            ts(f"[{imec}] slicing to stable interval")
            t0 = data['KilosortSortTimes'].start[0]
            t1 = data['KilosortSortTimes'].end[0]
            rec_ap_stable, rec_lfp_stable = slice_to_stable_interval(
                rec_ap_pre, rec_lfp_pre, t0, t1
            )
            
            preproc_cache[imec] = (rec_ap_pre, rec_lfp_pre, rec_ap_stable, rec_lfp_stable)
        
        rec_ap_pre, rec_lfp_pre, rec_ap_stable, rec_lfp_stable = preproc_cache[imec]
        
        # Detect bad channels (cached per imec)
        if imec not in bad_cache:
            ts(f"[{imec}] bad-channel detection")
            per_imec_dir = out_root / f"{imec}_shared"
            bad_ids, labels, csv_path = detect_and_save_bad_channels(
                rec_lfp_stable, per_imec_dir
            )
            bad_cache[imec] = (bad_ids, labels, csv_path)
        else:
            _, labels, csv_path = bad_cache[imec]
        
        # Load sorting for this Kilosort run
        ts(f"[{imec}] loading sorting for {ks.name}")
        sorting_interface = KiloSortSortingInterface(folder_path=ks)
        sorting = sorting_interface.sorting_extractor
        sorting._sampling_frequency = rec_ap_pre.get_sampling_frequency()
        
        # Build analyzer
        out_path = resolve_output_path(nwb_path, ks.name, sparse=args.sparse)
        ts(f"[{imec}] build analyzer")
        analyzer = build_sorting_analyzer(
            sorting=sorting,
            recording=rec_ap_pre,
            out_path=out_path,
            sparse=args.sparse,
            n_jobs=args.n_jobs
        )
        
        # Save merge groups and channel labels
        ts(f"[{imec}] save merge groups + channel labels")
        compute_and_save_merge_groups(analyzer, out_path)
        (out_path / "channel_labels.csv").write_text(csv_path.read_text())
        
        ts(f"[{imec}] DONE {ks.name}")
        
        # Diagnostics
        print(f"[diag] RSS: {psutil.Process().memory_info().rss/1e9:.2f} GB")
        for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
            if os.path.exists(p):
                v = open(p).read().strip()
                print("[diag] cgroup limit:", "unlimited" if not v.isdigit() else f"{int(v)/1e9:.1f} GB")
                break
        print("[diag] shm entries:", [os.path.basename(x) for x in glob.glob("/dev/shm/psm_*")][:8], "...")
        du = shutil.disk_usage("/dev/shm")
        print(f"[diag] /dev/shm used/free: {du.used/1e9:.1f}G/{du.free/1e9:.1f}G")
    
    ts("All KS runs completed")
    
    # Update spreadsheet
    subject = args.rec_id.split('_')[0]
    block = args.rec_id.split('_')[1]
    sheet_utils.write_to_recordings(subject, block, 'SI', 'y')


if __name__ == "__main__":
    main()