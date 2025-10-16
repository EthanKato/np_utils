import np_sheets as sheet_utils
import numpy as np



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