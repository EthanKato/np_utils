# Super commonly used functions

from typing import Tuple, Optional
import json


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