"""
SpikeInterface analyzer utilities.

Tools for building sorting analyzers and computing quality metrics.
"""
import time
from pathlib import Path
from spikeinterface import create_sorting_analyzer
from spikeinterface.curation import compute_merge_unit_groups
from spikeinterface.qualitymetrics import compute_quality_metrics


def ts(msg):
    """Timestamp print helper."""
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def build_sorting_analyzer(sorting, recording, out_path, sparse=True, n_jobs=8):
    """
    Build a complete sorting analyzer with all extensions.
    
    Parameters
    ----------
    sorting : SortingExtractor
        Spike sorting output
    recording : RecordingExtractor
        Preprocessed recording
    out_path : Path
        Output directory for analyzer
    sparse : bool, optional
        Whether to use sparse representation (default: True)
    n_jobs : int, optional
        Number of parallel jobs (default: 8)
        
    Returns
    -------
    analyzer : SortingAnalyzer
        Complete analyzer with all computed extensions
    """
    analyzer = create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        sparse=sparse,
        format="memory",
        return_in_uV=True
    )
    
    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
    
    # Phase 1: waveforms + templates
    ext_params_1 = {
        'random_spikes': {'method': 'uniform', 'max_spikes_per_unit': 10000},
        'waveforms': {'ms_before': 1.0, 'ms_after': 2.0},
        'templates': {'operators': ["average", "median", "std"]},
    }
    analyzer.compute(['random_spikes', 'waveforms', 'templates'],
                     extension_params=ext_params_1, **job_kwargs)
    ts("waveforms + templates done")
    analyzer.delete_extension('waveforms')  # free memory
    
    # Phase 2: metrics
    ext_params_2 = {
        'spike_amplitudes': {'peak_sign': 'neg'},
        'unit_locations': {'method': 'monopolar_triangulation'},
    }
    analyzer.compute(
        ['template_similarity', 'correlograms', 'spike_amplitudes', 'unit_locations',
         'template_metrics', 'isi_histograms', 'noise_levels'],
        extension_params=ext_params_2, **job_kwargs)
    ts("metrics done")
    
    # Phase 3: ACGs (memory-intensive, single-threaded)
    analyzer.compute(['acgs_3d'], n_jobs=1, chunk_duration="1s", progress_bar=True)
    ts("acgs done")
    
    # Compute quality metrics table
    _ = compute_quality_metrics(sorting_analyzer=analyzer)
    
    # Save to disk
    analyzer.save_as(format='binary_folder', folder=out_path)
    return analyzer


def compute_and_save_merge_groups(analyzer, out_path, preset="similarity_correlograms"):
    """
    Compute merge unit groups and save to file.
    
    Parameters
    ----------
    analyzer : SortingAnalyzer
        Sorting analyzer with computed extensions
    out_path : Path
        Output directory
    preset : str, optional
        Merge preset (default: "similarity_correlograms")
        
    Returns
    -------
    groups : list
        List of merge unit groups
    """
    groups = compute_merge_unit_groups(analyzer, preset=preset, resolve_graph=True)
    out_path = Path(out_path)
    (out_path / "merge_unit_groups.txt").write_text(str(groups))
    return groups