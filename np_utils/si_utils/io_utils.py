"""
I/O utilities for SpikeInterface workflows.

Functions for resolving paths, matching recordings to Kilosort runs, etc.
"""
import re
from typing import Literal, Tuple, Optional
from pathlib import Path


def get_matching_recording(nwbfile, kilosort_run, recording: Literal["LFP", "AP"] = "AP"):
    """
    Find the recording path that matches a Kilosort run.
    
    Parameters
    ----------
    nwbfile : NPNWBMaker
        NWB maker object with resolved paths
    kilosort_run : str
        Kilosort run name (e.g., "KS4_imec0")
    recording : {"LFP", "AP"}, optional
        Recording type (default: "AP")
        
    Returns
    -------
    stream : str
        Stream name for NWB
    matched_path : Path or None
        Path to matching recording
    stream_id : str
        SpikeGLX stream ID
    imec : str or None
        Imec probe identifier (e.g., "imec0")
    """
    m = re.search(r'imec\d+', str(kilosort_run))
    imec = m.group(0) if m else None
    matched = None
    
    if nwbfile.paths[recording] and imec:
        for p in nwbfile.paths[recording]:
            if imec in str(p):
                matched = p
                break
    
    stream = "ElectricalSeries" + recording[:2] + (imec or "imec0").capitalize()
    stream_id = (imec or "imec0") + "." + recording[:2].lower()
    
    return stream, matched, stream_id, imec


def resolve_output_path(nwb_path, kilosort_name, sparse=True):
    """
    Generate output path for SI analysis.
    
    Parameters
    ----------
    nwb_path : Path
        Path to NWB file
    kilosort_name : str
        Kilosort run name
    sparse : bool, optional
        Whether using sparse analyzer
        
    Returns
    -------
    out_path : Path
        Output directory path
    """
    label = "sparse" if sparse else "dense"
    out_root = Path(nwb_path).parent / "SI"
    out_path = out_root / f"{kilosort_name}_{label}"
    return out_path