from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Literal
import shutil
import numpy as np
import matplotlib.pyplot as plt

import spikeinterface.full as si
from spikeinterface.preprocessing import get_motion_parameters_preset
from spikeinterface.core.motion import Motion

from NWBMaker import NPNWBMaker

# Import from our core utilities
from ..core import (
    parse_rec_id,
    find_neural_binaries,
    read_stable_range,
    get_stream_id,
)

# Import from motion_utils
from .motion_utils import (
    get_peaks_medicine,
    compute_medicine_external,
    load_motion_from_folder,
    save_peak_map,
    save_motion_traces,
    plot_drift_maps_before_after,
)


class MotionCorrection:
    """
    Motion correction pipeline for single-probe Neuropixels recordings.
    
    Supports multiple motion correction algorithms:
    - Kilosort-like (SpikeInterface)
    - Dredge (SpikeInterface)
    - MEDiCINE (external package)
    
    Example:
        >>> mc = MotionCorrection("NP147_B1", probe_id="imec0", out_base="/path/to/output")
        >>> mc.resolve_ap_path(source='catgt')
        >>> mc.load_and_preprocess()
        >>> mc.run_all(["dredge", "medicine_ndb4"])
    """
    
    def __init__(
        self,
        rec_id: str,
        out_base: Optional[Path] = None,
        probe_id: str = "imec0",
        n_jobs: int = 16,
        chunk_duration: str = "1s",
        progress_bar: bool = True,
    ):
        """
        Initialize motion correction pipeline for a single probe.
        
        Args:
            rec_id: Recording ID (e.g., 'NP147_B1')
            out_base: Base output directory for motion correction results
            probe_id: Probe identifier (e.g., 'imec0', 'imec1')
            n_jobs: Number of parallel jobs for processing
            chunk_duration: Chunk size for processing
            progress_bar: Show progress bars during processing
        """
        self.rec_id = rec_id
        self.probe_id = probe_id
        self.subject, self.block = parse_rec_id(rec_id)
        self.out_base = Path(out_base) if out_base is not None else Path(f"/data_store2/neuropixels/preproc/{rec_id}/motion_traces/{probe_id}")
        self.job_kwargs = dict(
            n_jobs=n_jobs, 
            chunk_duration=chunk_duration, 
            progress_bar=progress_bar
        )

        # Will be populated by resolve_ap_path and load_and_preprocess
        self.ap_path: Optional[Path] = None
        self.raw_rec = None
        self.rec = None
        self.stable_range: Tuple[float, float] = (0.0, 0.0)

        # Peaks cache shared across MEDiCINE runs
        self.peaks_cache_dir = self.out_base / "_peaks_cache"
        self.peaks_cache_dir.mkdir(parents=True, exist_ok=True)

        # Preset tweaks
        self.dredge_th6 = dict(get_motion_parameters_preset("dredge"))
        self.dredge_th6["detect_kwargs"] = dict(self.dredge_th6["detect_kwargs"])
        self.dredge_th6["detect_kwargs"]["detect_threshold"] = 6.0

    def resolve_ap_path(
        self, 
        ap_path: Optional[str] = None,
        source: Literal['catgt', 'mc'] = 'catgt',
        use_nwb: bool = True
    ) -> Path:
        """
        Resolve AP recording path from multiple sources.
        
        Priority:
        1. Explicit ap_path argument
        2. Preprocessed file (CatGT or MC, based on source parameter)
        3. NWB file paths (if use_nwb=True)
        
        Args:
            ap_path: Explicit path to AP recording
            source: Preprocessing source - 'catgt' or 'mc'
            use_nwb: Try resolving from NWB if other methods fail
        
        Returns:
            Path to AP recording
        """
        # 1) Explicit path
        if ap_path:
            self.ap_path = Path(ap_path)
            print(f"[MC] Using explicit AP path: {self.ap_path}")
            return self.ap_path
        
        # 2) Try finding preprocessed file
        preproc_path = find_neural_binaries(
            rec_id=self.rec_id,
            source=source,
            probe_id=self.probe_id,
            band='ap'
        )
        
        if preproc_path:
            self.ap_path = Path(preproc_path)
            print(f"[MC] Found {source.upper()} AP: {self.ap_path}")
            return self.ap_path
        
        # 3) Try NWB as fallback
        if use_nwb:
            try:
                nwb = NPNWBMaker(rec_id=self.rec_id, silent=True, make_log_file=False)
                nwb.resolve_paths(auto_resolve=True, ks_select_all=True)
                
                # Find matching probe in NWB paths
                for ap in nwb.paths.get("AP", []):
                    if self.probe_id in str(ap):
                        self.ap_path = Path(ap)
                        print(f"[MC] Using NWB AP path: {self.ap_path}")
                        return self.ap_path
                
                # If no match, use first available
                if nwb.paths.get("AP"):
                    self.ap_path = Path(nwb.paths["AP"][0])
                    print(f"[MC] Using first NWB AP path (no probe match): {self.ap_path}")
                    return self.ap_path
            except Exception as e:
                print(f"[MC] Could not resolve from NWB: {e}")
        
        raise FileNotFoundError(
            f"Could not resolve AP path for {self.rec_id} (probe={self.probe_id}, source={source})"
        )

    def load_and_preprocess(
        self,
        t0: Optional[float] = None,
        t1: Optional[float] = None,
        config_path: str = "/userdata/ekato/git_repos/np_preproc/neural/sorting_config.json",
    ):
        """
        Load raw recording and preprocess for motion correction.
        
        Steps:
        1. Load raw AP recording using SpikeGLX interface
        2. Determine stable time range (from config, sheet, or manual override)
        3. Time-slice to stable range
        4. Apply preprocessing: bandpass filter + common reference
        
        Args:
            t0: Manual override for start time (seconds)
            t1: Manual override for end time (seconds)
            config_path: Path to sorting_config.json for stable ranges
        """
        if self.ap_path is None:
            raise ValueError("Must call resolve_ap_path() before load_and_preprocess()")
        
        # Load raw recording
        stream_id = get_stream_id(self.probe_id, band='ap')
        self.raw_rec = si.read_spikeglx(
            folder_path=self.ap_path.parent,
            stream_id=stream_id
        )
        self.raw_rec.shift_times(-self.raw_rec.get_start_time())
        
        # Determine stable range
        if t0 is not None and t1 is not None:
            self.stable_range = (float(t0), float(t1))
            print(f"[MC] Using manual stable range: {self.stable_range}")
        else:
            self.stable_range = read_stable_range(
                rec_id=self.rec_id,
                config_path=config_path,
                fallback_to_sheet=True,
            )
            print(f"[MC] Stable range: {self.stable_range}")
        
        # Time-slice to stable range
        t0, t1 = self.stable_range
        duration = self.raw_rec.get_total_duration()
        
        if not np.isfinite(t1) or t1 > duration:
            rec = self.raw_rec.time_slice(start_time=t0, end_time=None)
            t1 = round(duration, 3)
        else:
            t0 = max(0.0, t0)
            t1 = min(float(t1), float(duration))
            rec = self.raw_rec.time_slice(start_time=t0, end_time=t1)
        
        # Preprocess
        rec = rec.astype("float32")
        rec = si.bandpass_filter(rec, freq_min=300.0, freq_max=6000.0)
        rec = si.common_reference(rec, reference="global", operator="median")
        
        self.rec = rec
        self.stable_range = (t0, t1)
        print(f"[MC] Preprocessing complete. Duration: {t1 - t0:.2f}s")

    def run_preset(
        self, 
        preset: str, 
        replace: bool = False
    ) -> Tuple[Optional[Motion], Optional[Dict[str, Any]], Optional[Path]]:
        """
        Run a single motion correction preset.
        
        Supported presets:
        - 'dredge': Standard dredge algorithm
        - 'dredge_th6': Dredge with detection threshold = 6
        - 'dredge_fast': Fast dredge variant
        - 'kilosort_like': Kilosort-style motion correction
        - 'medicine_SI': MEDiCINE through SpikeInterface
        - 'medicine_ndbN': External MEDiCINE with N depth bins (e.g., medicine_ndb4)
        
        Args:
            preset: Preset name
            replace: Overwrite existing results if folder exists
        
        Returns:
            tuple: (Motion object, info dict, output folder path) or (None, None, None) if skipped
        """
        if self.rec is None:
            raise ValueError("Must call load_and_preprocess() before run_preset()")
        
        folder = self.out_base / preset
        
        if folder.exists() and not replace:
            print(f"[MC] Preset folder exists, skipping: {folder}")
            return None, None, None
        
        if folder.exists() and replace:
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        
        # Run motion correction based on preset
        if preset == "dredge_th6":
            motion, info = si.compute_motion(
                self.rec,
                preset="dredge",
                detect_kwargs=self.dredge_th6["detect_kwargs"],
                folder=folder,
                output_motion_info=True,
                **self.job_kwargs,
            )
        elif preset in {"dredge", "dredge_fast", "kilosort_like", "medicine_SI"}:
            p = "medicine" if preset == "medicine_SI" else preset
            motion, info = si.compute_motion(
                self.rec, 
                preset=p, 
                folder=folder, 
                output_motion_info=True, 
                **self.job_kwargs
            )
        elif preset.startswith("medicine_ndb"):
            # External MEDiCINE
            ndb = int(preset.split("ndb")[-1])
            peaks, locs = get_peaks_medicine(
                recording=self.rec,
                cache_dir=self.peaks_cache_dir,
                detect_threshold=5.0,
                job_kwargs=self.job_kwargs,
            )
            motion, info = compute_medicine_external(
                recording=self.rec,
                output_folder=folder,
                num_depth_bins=ndb,
                peaks=peaks,
                peak_locations=locs,
            )
            # Save peaks for later use
            np.save(folder / 'peaks.npy', peaks)
            np.save(folder / 'peak_locations.npy', locs)
        else:
            raise ValueError(f"Unknown preset: {preset}")
        
        # Generate all visualizations
        self._generate_visualizations(preset, folder)
        
        return motion, info, folder

    def _generate_visualizations(self, preset: str, folder: Path):
        """Generate all standard visualizations for a preset."""
        # Load peaks and motion
        peaks = np.load(folder / 'peaks.npy')
        locs = np.load(folder / 'peak_locations.npy')
        motion = load_motion_from_folder(folder)
        
        # Peak map
        save_peak_map(
            recording=self.rec,
            peaks=peaks,
            peak_locations=locs,
            stable_range=self.stable_range,
            output_path=folder / f"{preset}_spike_map.png",
            rec_id=self.rec_id,
            probe_id=self.probe_id,
            preset_name=preset,
        )
        
        # Motion traces
        save_motion_traces(
            recording=self.rec,
            motion=motion,
            peaks=peaks,
            peak_locations=locs,
            stable_range=self.stable_range,
            output_path=folder / f"{preset}_motion_traces.png",
            rec_id=self.rec_id,
            probe_id=self.probe_id,
            preset_name=preset,
        )
        
        # Before/after drift maps
        plot_drift_maps_before_after(
            recording=self.rec,
            motion=motion,
            peaks=peaks,
            peak_locations=locs,
            stable_range=self.stable_range,
            output_path=folder / f"{preset}_corrected_drift_maps.png",
            rec_id=self.rec_id,
            probe_id=self.probe_id,
            preset_name=preset,
        )

    def run_all(self, presets: List[str], replace: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple motion correction presets in sequence.
        
        Args:
            presets: List of preset names to run
            replace: Overwrite existing results
        
        Returns:
            dict: Results keyed by preset name
        """
        results = {}
        for p in presets:
            print(f"[MC] Running preset: {p}")
            motion, info, folder = self.run_preset(p, replace=replace)
            results[p] = dict(motion=motion, info=info, folder=folder)
            if folder:
                print(f"[MC] Complete: {p} → {folder}")
        return results


# ---------- example usage ----------
if __name__ == "__main__":
    import os
    
    # Example: single probe motion correction
    mc = MotionCorrection(
        rec_id="NP149_B1",
        probe_id="imec0",
        out_base=Path("/data_store2/neuropixels/preproc/NP149_B1/motion_traces/imec0")
    )
    mc.resolve_ap_path(source='catgt')
    mc.load_and_preprocess()

    presets = ["kilosort_like", "dredge_th6", "dredge", "medicine_SI", "medicine_ndb2", "medicine_ndb4"]
    results = mc.run_all(presets)

    # Quick overlay of temporal medians
    plt.figure(figsize=(9, 4))
    for name, pack in results.items():
        if pack["motion"] is None:
            continue
        m = pack["motion"]
        t = np.squeeze(m.temporal_bins_s)
        if t.size == m.displacement.shape[0] + 1:
            t = 0.5 * (t[:-1] + t[1:])
        disp = np.nanmedian(m.displacement, axis=1)
        plt.plot(t, disp, label=name)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (µm)")
    plt.title(f"Motion traces (temporal median) — {mc.rec_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(mc.out_base / f"{mc.rec_id}_motion_overlay.png", dpi=150)
    print(f"[MC] Saved motion overlay")
