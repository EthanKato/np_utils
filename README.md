# np_utils
So I don't rewrite the same code over and over and over

## Installation
```bash
pip install -e .
```

## API Reference

### Oversight Utils (`np_utils.oversight_utils`)

#### `get_rec_ids(column_name, condition, recordings_df=None)`
Get recording IDs that match a specific condition on a given column.

**Args:**
- `column_name` (str): Name of the column to check the condition against.
- `condition`: Either a value to match (for equality check) or a callable that takes the column and returns a boolean mask for complex conditions.
  - Simple: `condition=''` checks for empty strings
  - Complex: `condition=lambda col: col < some_date`
  - Complex: `condition=lambda col: (col > 10) & (col < 20)`
- `recordings_df` (pd.DataFrame, optional): Recordings dataframe. If None, loads from 'recordings' sheet.

**Returns:**
- `np.ndarray`: Array of recording IDs that match the condition.

#### `get_need_nwb(recordings_df=None)`
Get recording IDs that need NWB files generated.

**Args:**
- `recordings_df` (pd.DataFrame, optional): Recordings dataframe. If None, loads from 'recordings' sheet.

**Returns:**
- `np.ndarray`: Array of recording IDs that need NWB files.

#### `get_has_nwb(recordings_df=None)`
Get recording IDs that have NWB files generated.

**Args:**
- `recordings_df` (pd.DataFrame, optional): Recordings dataframe. If None, loads from 'recordings' sheet.

**Returns:**
- `np.ndarray`: Array of recording IDs that have NWB files.



### NWB Utils (`np_utils.nwb_utils`)
...