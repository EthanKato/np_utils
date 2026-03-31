"""Command-line Neuroopixels NWB creation entry point."""
import argparse
from NWBMaker import NPNWBMaker


def main():
    parser = argparse.ArgumentParser(description="Make NWB for a single Neuroopixels recording")
    parser.add_argument("--output-path", required=True, help="Output directory")
    parser.add_argument("--rec-id", required=True, help="Recording ID")
    parser.add_argument("--include-ap", action="store_true", help="Include AP data")
    args = parser.parse_args()

    nwb = NPNWBMaker(
        output_path=args.output_path,
        rec_id=args.rec_id,
        silent=False,
        include_ap=args.include_ap,
    )
    nwb.run_all()


if __name__ == "__main__":
    main()