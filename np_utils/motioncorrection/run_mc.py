
"""
Command-line entry point for motion correction.

Usage:
    python -m np_utils.motioncorrection.run_mc --rec-id NP147_B1 --probe-id imec0 --presets dredge medicine_ndb4
"""
import argparse
from pathlib import Path
from .motion_correction import MotionCorrection

def main():
    parser = argparse.ArgumentParser(description="Run motion correction")
    parser.add_argument("--rec-id", required=True)
    parser.add_argument("--probe-id", default="imec0")
    parser.add_argument("--presets", nargs="+", required=True)
    parser.add_argument("--source", default='catgt', choices=['catgt', 'mc'])
    parser.add_argument("--out-base", default="/data_store2/neuropixels/preproc")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    
    # Construct full output path
    outdir = Path(args.out_base) / args.rec_id / "motion_traces" / args.probe_id
    
    mc = MotionCorrection(rec_id=args.rec_id, probe_id=args.probe_id, out_base=outdir)
    mc.resolve_ap_path(source=args.source)
    mc.load_and_preprocess()
    mc.run_all(args.presets, replace=args.replace)
    
    print(f"[MC] Complete!")

if __name__ == "__main__":
    main()