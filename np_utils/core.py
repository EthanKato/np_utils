# Super commonly used functions

from typing import Tuple, Optional, Literal, List, Dict
import json
import glob
import os
import math
from pathlib import Path
import np_sheets as sheet_utils
import re

def parse_rec_id(rec_id: str) -> Tuple[str, str]:
    """
    Parse recording ID into subject and block.
    
    Args:
        rec_id (str): Recording ID in format 'Subject_Block' (e.g., 'NP04_B2').
    
    Returns:
        tuple: (subject, block) as strings.
    
    Examples:
        >>> parse_rec_id("NP04_B2")
        ('NP04', 'B2')
        >>> parse_rec_id("NP154_B1")
        ('NP154', 'B1')
    """
    parts = rec_id.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid rec_id format: {rec_id}. Expected 'Subject_Block'")
    return parts[0], parts[1]

def parse_sheet_trange(value: str, inf_as_string: bool = True) -> Optional[list]:
    """
    Parse time range from Google Sheets string value.
    
    Args:
        value (str): String from sheet (e.g., '[41, 931]' or '[0, inf]').
        inf_as_string (bool): If True, convert infinity to string 'inf'.
            If False, convert to float('inf'). Default: True.
    
    Returns:
        list or None: Parsed time range, or None if empty/invalid.
    
    Examples:
        >>> parse_sheet_trange('[41, 931]')
        [41, 931]
        >>> parse_sheet_trange('[0, Inf]')
        [0, 'inf']
        >>> parse_sheet_trange('[0, Inf]', inf_as_string=False)
        [0, inf]
        >>> parse_sheet_trange('[0,inf]')
        [0, 'inf']
        >>> parse_sheet_trange('')
        None
        >>> parse_sheet_trange('?')
        None
    """
    import math
    import re
    
    if not value or value.strip() == "" or value.strip() == "?":
        return None
    
    try:
        # Handle any capitalization of 'inf' by temporarily replacing with quoted version for JSON parsing
        processed_value = re.sub(r'\binf\b', '"inf"', value, flags=re.IGNORECASE)
        result = json.loads(processed_value)
        
        # Normalize infinity values based on flag
        if isinstance(result, list):
            if inf_as_string:
                # Convert float inf to string 'inf'
                result = [
                    'inf' if (isinstance(x, float) and math.isinf(x)) 
                    else ('inf' if (isinstance(x, str) and x.lower() == 'inf') else x)
                    for x in result
                ]
            else:
                # Convert string 'inf' to float inf
                result = [
                    float('inf') if (isinstance(x, str) and x.lower() == 'inf')
                    else x
                    for x in result
                ]
        
        return result
    except:
        # Fallback: try eval for simple cases
        try:
            result = eval(value)
            
            # Normalize infinity values based on flag
            if isinstance(result, list):
                if inf_as_string:
                    # Convert float inf to string 'inf'
                    result = [
                        'inf' if (isinstance(x, float) and math.isinf(x)) else x
                        for x in result
                    ]
                else:
                    # Convert string 'inf' to float inf (though eval already does this)
                    result = [
                        float('inf') if (isinstance(x, str) and x.lower() == 'inf')
                        else x
                        for x in result
                    ]
            
            return result
        except:
            return None

def read_stable_range(
    rec_id: str,
    config_path: str = "/userdata/ekato/git_repos/np_preproc/neural/sorting_config.json",
    fallback_to_sheet: bool = True,
    default: Tuple[float, float] = (0.0, math.inf)
) -> Tuple[float, float]:
    """
    Read stable recording range (sort times) for a recording.
    
    Priority order:
    1. sorting_config.json (if exists and has rec_id)
    2. Google Sheets 'Sort time' column (if fallback enabled)
    3. Default range (0, inf)
    
    Args:
        rec_id (str): Recording ID
        config_path (str): Path to sorting_config.json
        fallback_to_sheet (bool): Try reading from sheet if config fails
        default (tuple): Default range if all methods fail
    
    Returns:
        tuple: (t0, t1) in seconds
    
    Examples:
        >>> read_stable_range("NP147_B1")
        (100.0, 1250.0)
        >>> read_stable_range("NP999_B1")  # Not in config/sheet
        (0.0, inf)
    """
    subject, block = parse_rec_id(rec_id)
    
    # 1) Try config file first
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        if rec_id in config:
            trange = config[rec_id].get('trange')
            if trange and len(trange) == 2:
                t0, t1 = trange
                # Convert 'inf' string to float if needed
                if isinstance(t0, str) and t0.lower() == 'inf':
                    t0 = math.inf
                if isinstance(t1, str) and t1.lower() == 'inf':
                    t1 = math.inf
                
                # Validate
                if math.isfinite(t0) and (math.isinf(t1) or t1 > t0):
                    return (float(t0), float(t1))
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        pass  # Fall through to sheet
    
    # 2) Try sheet as fallback
    if fallback_to_sheet:
        try:
            sr = sheet_utils.read_from_recordings(subject, block, "Sort time")
            parsed = parse_sheet_trange(sr, inf_as_string=False)
            if parsed and len(parsed) == 2:
                t0, t1 = parsed
                if math.isfinite(t0) and (math.isinf(t1) or t1 > t0):
                    return (float(t0), float(t1))
        except Exception as e:
            print(f"[WARN] Could not read sort time from sheet for {rec_id}: {e}")
    
    # 3) Use default
    print(f"[WARN] Using default stable range {default} for {rec_id}")
    return default


def extract_probe_from_path(path: str) -> Optional[str]:
    """
    Extract probe ID (imec0, imec1, etc.) from a file path.
    
    Args:
        path (str): Path to AP or LF binary file
    
    Returns:
        str or None: Probe ID (e.g., 'imec0') or None if not found
    
    Examples:
        >>> extract_probe_from_path("/path/to/NP150_B1_g0_imec0/file.ap.bin")
        'imec0'
        >>> extract_probe_from_path("/path/NP150_B1_g0_tcat.imec1.lf.bin")
        'imec1'
    """
    # Try pattern: imecN.ap.bin or imecN.lf.bin
    match = re.search(r'(imec\d+)\.(ap|lf)\.bin', path)
    if match:
        return match.group(1)
    
    # Try pattern: folder name contains imecN
    match = re.search(r'/(imec\d+)/', path)
    if match:
        return match.group(1)
    
    return None

def get_stream_id(probe_id: str, band: str = 'ap') -> str:
    """
    Construct SpikeGLX stream ID from probe ID and band.
    
    Args:
        probe_id (str): Probe identifier (e.g., 'imec0', 'imec1')
        band (str): Recording band - 'ap' or 'lf' (default: 'ap')
    
    Returns:
        str: Stream ID for SpikeGLX (e.g., 'imec0.ap')
    
    Examples:
        >>> get_stream_id('imec0', 'ap')
        'imec0.ap'
        >>> get_stream_id('imec1', 'lf')
        'imec1.lf'
    """
    band = band.lower()
    if band not in ['ap', 'lf']:
        raise ValueError(f"Band must be 'ap' or 'lf', got '{band}'")
    return f"{probe_id}.{band}"

def get_corresponding_path(path: str, target_band: str) -> Optional[Path]:
    """
    Get the corresponding AP or LF path from the other band.
    
    Args:
        path (str): Path to AP or LF file
        target_band (str): Target band - 'ap' or 'lf'
    
    Returns:
        Path or None: Path to corresponding file if it exists, None otherwise
    
    Examples:
        >>> get_corresponding_path("/path/file.imec0.ap.bin", "lf")
        Path('/path/file.imec0.lf.bin')  # if file exists
        >>> get_corresponding_path("/path/file.imec1.lf.bin", "ap")
        Path('/path/file.imec1.ap.bin')  # if file exists
    """
    target_band = target_band.lower()
    if target_band not in ['ap', 'lf']:
        raise ValueError(f"target_band must be 'ap' or 'lf', got '{target_band}'")
    
    # Replace .ap.bin with .lf.bin or vice versa
    if target_band == 'lf':
        corresponding_path = re.sub(r'\.ap\.bin$', '.lf.bin', path)
    else:
        corresponding_path = re.sub(r'\.lf\.bin$', '.ap.bin', path)
    
    # Convert to Path and check if it exists
    path_obj = Path(corresponding_path)
    return path_obj if path_obj.exists() else None

def find_all_probes_in_path(directory: str, band: str = 'ap') -> Dict[str, str]:
    """
    Find all probe binary files in a directory.
    
    Args:
        directory (str): Directory to search
        band (str): Band to search for - 'ap' or 'lf' (default: 'ap')
    
    Returns:
        dict: Mapping of probe_id -> file_path
    
    Examples:
        >>> find_all_probes_in_path("/path/to/catgt_dir", "ap")
        {'imec0': '/path/to/.../file.imec0.ap.bin', 
         'imec1': '/path/to/.../file.imec1.ap.bin'}
    """
    import glob
    
    band = band.lower()
    directory = str(directory)
    
    # Search for all files matching pattern
    pattern = os.path.join(directory, "**", f"*.{band}.bin")
    files = glob.glob(pattern, recursive=True)
    
    # Extract probe IDs and build dict
    probe_dict = {}
    for file in files:
        probe_id = extract_probe_from_path(file)
        if probe_id:
            probe_dict[probe_id] = file
    
    return probe_dict

def parse_binary_path(path: str) -> Dict[str, Optional[str]]:
    """
    Parse a SpikeGLX binary path to extract all components.
    
    Args:
        path (str): Path to binary file
    
    Returns:
        dict: Dictionary with keys 'probe_id', 'band', 'stream_id', 'directory'
    
    Examples:
        >>> parse_binary_path("/path/NP150_B1_g0_tcat.imec0.ap.bin")
        {'probe_id': 'imec0', 'band': 'ap', 'stream_id': 'imec0.ap', 
         'directory': '/path'}
    """
    path_obj = Path(path)
    probe_id = extract_probe_from_path(path)
    
    # Extract band
    band = None
    if '.ap.bin' in path:
        band = 'ap'
    elif '.lf.bin' in path:
        band = 'lf'
    
    # Construct stream_id
    stream_id = get_stream_id(probe_id, band) if (probe_id and band) else None
    
    return {
        'probe_id': probe_id,
        'band': band,
        'stream_id': stream_id,
        'directory': str(path_obj.parent)
    }

def find_neural_binaries(
    rec_id: str,
    source: Literal['catgt', 'mc'] = 'catgt',
    root_path: str = '/data_store2/neuropixels/preproc',
    probe_id: Optional[str] = None,
    band: str = 'ap'
) -> Optional[str]:
    """
    Find preprocessed binary file(s) for a recording.
    
    Args:
        rec_id (str): Recording ID (e.g., 'NP79_B1')
        source (str): Source type - 'catgt' or 'mc' (motion-corrected)
        root_path (str): Root directory for raw or preprocessed data
        probe_id (str, optional): Specific probe to find (e.g., 'imec0'). 
            If None, returns first found.
        band (str): Band to find - 'ap' or 'lf' (default: 'ap')
    
    Returns:
        str or None: Path to binary file, or None if not found
    
    Examples:
        >>> find_neural_binaries("NP79_B1", source='catgt', probe_id='imec0')
        '/data_store2/.../NP79_B1_g0_tcat.imec0.ap.bin'
        >>> find_neural_binaries("NP79_B1", source='mc', band='lf')
        '/data_store2/.../NP79_B1_g0_tcat.imec0.lf.bin'
    """
    import glob
    import os
    
    band = band.lower()
    if band not in ['ap', 'lf']:
        raise ValueError(f"Band must be 'ap' or 'lf', got '{band}'")
    
    if source == 'catgt':
        # CatGT pattern: catgt_*/*/file.imecN.{band}.bin
        pattern = os.path.join(
            str(root_path), rec_id, "sglx", "catgt_*", "*", 
            f"*.{band}.bin"
        )
    elif source == 'mc':
        # MC pattern: mc_{rec_id}_g0/{rec_id}_g0_imecN/{rec_id}_g0_tcat.imecN.{band}.bin
        if probe_id:
            pattern = os.path.join(
                str(root_path), rec_id, "sglx", 
                f"mc_{rec_id}_g0", f"{rec_id}_g0_{probe_id}",
                f"{rec_id}_g0_tcat.{probe_id}.{band}.bin"
            )
        else:
            pattern = os.path.join(
                str(root_path), rec_id, "sglx", 
                f"mc_{rec_id}_g0", f"{rec_id}_g0_imec*",
                f"{rec_id}_g0_tcat.imec*.{band}.bin"
            )
    else:
        raise ValueError(f"Unknown source type: {source}. Use 'catgt' or 'mc'")
    
    files = glob.glob(pattern)
    
    # Filter by probe_id if specified
    if probe_id and files:
        files = [f for f in files if probe_id in f]
    
    if len(files) == 0:
        return None
    elif len(files) == 1:
        return files[0]
    else:
        print(f"[WARN] Multiple files found for {rec_id} ({source}, {band}), returning first: {files[0]}")
        return files[0]


def find_all_neural_binaries(
    rec_id: str,
    source: Literal['catgt', 'mc'] = 'catgt',
    root_path: str = '/data_store2/neuropixels/preproc',
    band: str = 'ap'
) -> Dict[str, str]:
    """
    Find ALL preprocessed binary files for a recording (multi-probe support).
    
    Args:
        rec_id (str): Recording ID
        source (str): Source type - 'catgt' or 'mc'
        root_path (str): Root directory for raw or preprocessed data
        band (str): Band to find - 'ap' or 'lf' (default: 'ap')
    
    Returns:
        dict: Mapping of probe_id -> file_path
    
    Examples:
        >>> find_all_neural_binaries("NP79_B1", source='mc', band='ap')
        {'imec0': '/path/to/imec0.ap.bin', 'imec1': '/path/to/imec1.ap.bin'}
    """
    import glob
    import os
    
    band = band.lower()
    if band not in ['ap', 'lf']:
        raise ValueError(f"Band must be 'ap' or 'lf', got '{band}'")
    
    if source == 'catgt':
        pattern = os.path.join(
            str(root_path), rec_id, "sglx", "catgt_*", "*", 
            f"*.{band}.bin"
        )
    elif source == 'mc':
        pattern = os.path.join(
            str(root_path), rec_id, "sglx", 
            f"mc_{rec_id}_g0", f"{rec_id}_g0_imec*",    
            f"{rec_id}_g0_tcat.imec*.{band}.bin"
        )
    else:
        raise ValueError(f"Unknown source type: {source}. Use 'catgt' or 'mc'")
    
    files = glob.glob(pattern)
    
    # Build probe_id -> path mapping
    probe_dict = {}
    for file in sorted(files):
        probe_id = extract_probe_from_path(file)
        if probe_id:
            probe_dict[probe_id] = file
    
    return probe_dict