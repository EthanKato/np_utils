import np_sheets as sheet_utils
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .core import parse_rec_id, parse_sheet_trange


def get_rec_ids(column_name, condition, recordings_df=None):
    """
    Get recording IDs that match a specific condition on a given column.
    
    Filters recordings dataframe to find entries where the specified column matches
    the condition, then returns a numpy array of formatted recording IDs in the format 'Subject_Block'.
    
    Args:
        column_name (str): Name of the column to check the condition against.
        condition: Either a value to match (for equality check) or a callable that takes 
            the column and returns a boolean mask for complex conditions.
            Examples:
                - Simple: condition='' checks for empty strings
                - Complex: condition=lambda col: col < some_date
                - Complex: condition=lambda col: (col > 10) & (col < 20)
        recordings_df (pd.DataFrame, optional): Recordings dataframe. 
            If None, loads from 'recordings' sheet.
    
    Returns:
        np.ndarray: Array of recording IDs that match the condition.
    """
    if recordings_df is None:
        recordings_df = sheet_utils.sheet_to_df('recordings')
    
    # Check if condition is callable (function/lambda) for complex conditions
    if callable(condition):
        mask = condition(recordings_df[column_name])
    else:
        # Simple equality check
        mask = recordings_df[column_name] == condition
    
    rec_ids = (recordings_df.loc[mask, 'Subject'].astype(str) + '_' + 
               recordings_df.loc[mask, 'Block'].astype(str))
    
    return rec_ids.values

    
def get_need_nwb(recordings_df=None):
    """
    Get recording IDs that need NWB files generated.
    
    Filters recordings dataframe to find entries where NWB column is empty,
    then returns a numpy array of formatted recording IDs in the format 'Subject_Block'.
    
    Args:
        recordings_df (pd.DataFrame, optional): Recordings dataframe. 
            If None, loads from 'recordings' sheet.
    
    Returns:
        np.ndarray: Array of recording IDs that need NWB files.
    """
    return get_rec_ids('NWB', '', recordings_df)

def get_has_nwb(recordings_df=None):
    """
    Get recording IDs that have NWB files generated.
    
    Filters recordings dataframe to find entries where NWB column is not empty,
    then returns a numpy array of formatted recording IDs in the format 'Subject_Block'.
    
    Args:
        recordings_df (pd.DataFrame, optional): Recordings dataframe. 
            If None, loads from 'recordings' sheet.
    
    Returns:
        np.ndarray: Array of recording IDs that have NWB files.
    """
    return get_rec_ids('NWB', lambda col: col != '', recordings_df)



def format_trange(trange: list) -> str:
    """
    Format time range list for Google Sheets storage.
    
    Args:
        trange (list): Time range as [start, end], where end can be "inf".
    
    Returns:
        str: Formatted string representation.
    
    Examples:
        >>> format_trange([41, 931])
        '[41, 931]'
        >>> format_trange([0, "inf"])
        '[0, inf]'
    """
    return str(trange)

def load_sorting_config(config_path: str = "/userdata/ekato/git_repos/np_preproc/neural/sorting_config.json") -> dict:
    """
    Load sorting configuration from JSON file.
    
    Args:
        config_path (str): Path to sorting_config.json file.
    
    Returns:
        dict: Sorting configuration with rec_id as keys.
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def validate_sort_times(config_path: str = None, recordings_df=None) -> Dict[str, any]:
    """
    Validate sort times between JSON config and Google Sheets.
    
    Compares the 'Sort time' column in the Recordings sheet with trange values
    from sorting_config.json. Returns detailed results about matches, mismatches,
    and missing entries.
    
    Args:
        config_path (str, optional): Path to sorting_config.json. 
            Defaults to standard location.
        recordings_df (pd.DataFrame, optional): Recordings dataframe.
            If None, loads from 'Recordings' sheet.
    
    Returns:
        dict: Results with keys:
            - 'total': Total recordings in config
            - 'matches': Number of matching entries
            - 'mismatches': List of mismatched entries with details
            - 'missing_in_sheet': List of rec_ids not found in sheet
            - 'errors': List of errors encountered
    
    Example:
        >>> results = validate_sort_times()
        >>> print(f"Matches: {results['matches']}/{results['total']}")
        >>> for m in results['mismatches']:
        ...     print(f"{m['rec_id']}: config={m['config']}, sheet={m['sheet']}")
    """
    if config_path is None:
        config_path = "/userdata/ekato/git_repos/np_preproc/neural/sorting_config.json"
    
    # Load config
    config = load_sorting_config(config_path)
    
    # Load sheet if not provided (single API call)
    if recordings_df is None:
        recordings_df = sheet_utils.sheet_to_df('Recordings')
    
    # Create lookup dictionary from dataframe for fast access
    # Key: (subject, block), Value: sort_time value
    sheet_lookup = {}
    for idx, row in recordings_df.iterrows():
        subject = str(row.get('Subject', '')).strip()
        block = str(row.get('Block', '')).strip()
        sort_time = str(row.get('Sort time', '')).strip()
        if subject and block:
            sheet_lookup[(subject, block)] = sort_time
    
    # Results
    results = {
        'total': len(config),
        'matches': 0,
        'mismatches': [],
        'missing_in_sheet': [],
        'errors': []
    }
    
    # Process each recording in config
    for rec_id, rec_config in sorted(config.items()):
        try:
            # Parse rec_id
            subject, block = parse_rec_id(rec_id)
            
            # Get trange from config
            config_trange = rec_config.get('trange')
            if config_trange is None:
                results['errors'].append({
                    'rec_id': rec_id,
                    'error': 'No trange in config'
                })
                continue
            
            # Look up in dataframe (no API call!)
            sheet_value = sheet_lookup.get((subject, block))
            
            if sheet_value is None:
                # Recording not in sheet
                results['missing_in_sheet'].append(rec_id)
            else:
                # Parse and compare
                sheet_trange = parse_sheet_trange(sheet_value, inf_as_string=True)
                
                if sheet_trange == config_trange:
                    results['matches'] += 1
                else:
                    results['mismatches'].append({
                        'rec_id': rec_id,
                        'subject': subject,
                        'block': block,
                        'config': config_trange,
                        'sheet': sheet_trange,
                        'sheet_raw': sheet_value
                    })
                    
        except Exception as e:
            results['errors'].append({
                'rec_id': rec_id,
                'error': f"Unexpected error: {str(e)}"
            })
    
    return results