# np_utils/job_utils.py
import subprocess
import shlex
from pathlib import Path
import time
from datetime import datetime
from typing import List, Optional, Union, Callable, Any

def submit_job(
    script: Union[str, Path],
    rec_ids: Optional[List[str]] = None,
    python_executable: str = "/userdata/ekato/miniforge3/envs/se2nwb/bin/python",
    queue: str = "mind-batch",
    cores: int = 8,
    memory_gb: int = 16,
    log_dir: Union[str, Path] = "/tmp/job_logs",
    job_prefix: str = "job",
    use_time: bool = False,
    extra_args: Optional[List[str]] = None,
    executable: Optional[str] = None,
    gpus: Optional[int] = None,
    dry_run: bool = False
):
    """
    Submit a job (or batch of jobs) to SGE queue using submit_job command.
    
    Args:
        script (str | Path): Path to the script to run.
        rec_ids (List[str], optional): List of recording IDs to process. 
            If provided, submits one job per rec_id, passing --rec-id to script.
        python_executable (str): Path to Python executable. 
            Default: conda se2nwb environment.
        queue (str): Queue name (e.g., 'mind-batch', 'skull-gpu', 'pia-batch.q').
        cores (int): Number of CPU cores to request (ignored for GPU queues).
        memory_gb (int): Total memory in GB to allocate.
        log_dir (str | Path): Directory to store job output logs.
        job_prefix (str): Prefix for job names and log files.
        use_time (bool): If True, wraps command with /usr/bin/time -v for profiling.
        extra_args (List[str], optional): Additional arguments to pass to the script.
        executable (str, optional): Override Python with custom executable 
            (e.g., 'matlab', 'pythonconda3'). If provided, python_executable is ignored.
        gpus (int, optional): Number of GPUs to request (use with GPU queues only).
        dry_run (bool): If True, prints command without executing.
    
    Returns:
        None
    
    Examples:
        # Simple single job
        submit_job(
            script="process.py",
            memory_gb=32,
            cores=4
        )
        
        # Batch jobs with recording IDs
        submit_job(
            script="run_si_proc.py",
            rec_ids=["NP139_B2", "NP140_B1"],
            queue="mind-batch",
            cores=9,
            memory_gb=256,
            job_prefix="si"
        )
        
        # GPU job
        submit_job(
            script="train_model.py",
            queue="skull-gpu",
            gpus=2,
            memory_gb=80
        )
        
        # MATLAB job
        submit_job(
            script="/home/user/my_script.m",
            executable="matlab",
            queue="pia-batch.q"
        )
    """
    # Ensure paths
    script = Path(script)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine if batch mode (multiple rec_ids) or single job
    if rec_ids:
        for rec_id in rec_ids:
            _submit_single_job(
                script=script,
                rec_id=rec_id,
                python_executable=python_executable,
                queue=queue,
                cores=cores,
                memory_gb=memory_gb,
                log_dir=log_dir,
                job_prefix=job_prefix,
                use_time=use_time,
                extra_args=extra_args,
                executable=executable,
                gpus=gpus,
                dry_run=dry_run
            )
    else:
        _submit_single_job(
            script=script,
            rec_id=None,
            python_executable=python_executable,
            queue=queue,
            cores=cores,
            memory_gb=memory_gb,
            log_dir=log_dir,
            job_prefix=job_prefix,
            use_time=use_time,
            extra_args=extra_args,
            executable=executable,
            gpus=gpus,
            dry_run=dry_run
        )


def _submit_single_job(
    script: Path,
    rec_id: Optional[str],
    python_executable: str,
    queue: str,
    cores: int,
    memory_gb: int,
    log_dir: Path,
    job_prefix: str,
    use_time: bool,
    extra_args: Optional[List[str]],
    executable: Optional[str],
    gpus: Optional[int],
    dry_run: bool
):
    """Internal function to submit a single job."""
    # Build job name
    jobname = f"{job_prefix}_{rec_id}" if rec_id else job_prefix
    logfile = log_dir / f"{jobname}.log"
    
    # Build the payload (what actually runs)
    payload = []
    
    # Add time wrapper if requested
    if use_time:
        payload.extend(["/usr/bin/time", "-v"])
    
    # Add executable and script
    if executable:
        # Custom executable (e.g., matlab)
        payload.extend([executable, str(script)])
    else:
        # Python
        payload.extend([python_executable, str(script)])
    
    # Add rec_id if provided
    if rec_id:
        payload.extend(["--rec-id", rec_id])
    
    # Add extra arguments
    if extra_args:
        payload.extend(extra_args)
    
    # Build submit_job command
    cmd = ["submit_job", "-q", queue, "-m", str(memory_gb), 
           "-o", str(logfile), "-n", jobname]
    
    # Add cores or GPUs
    if gpus is not None:
        cmd.extend(["-g", str(gpus)])
    else:
        cmd.extend(["-c", str(cores)])
    
    # Add execution payload
    cmd.extend(["-x", *payload])
    
    # Print and optionally execute
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    print(f"{'[DRY RUN] ' if dry_run else ''}Submitting: {cmd_str}")
    
    if not dry_run:
        try:
            result = subprocess.run(cmd, text=True, capture_output=True)
            output = result.stdout.strip() or result.stderr.strip() or "(no scheduler output)"
            print(output)
        except Exception as e:
            print(f"[ERROR] Failed to submit job {jobname}: {e}")


def batch_submit(rec_ids: List[str], **kwargs):
    """
    Convenience function for batch submissions.
    
    Args:
        rec_ids: List of recording IDs to process
        **kwargs: All other arguments to pass to submit_job()
    
    Example:
        batch_submit(
            rec_ids=["NP139_B2", "NP140_B1"],
            script="run_si_proc.py",
            cores=9,
            memory_gb=256
        )
    """
    submit_job(rec_ids=rec_ids, **kwargs)



def get_running_jobs():
    """
    Returns the number of currently running jobs and their names.
    
    Returns:
        tuple: (job_count: int, job_names: list[str])
    """
    import subprocess
    
    try:
        # Try qstat first (SGE/UGE/PBS)
        result = subprocess.run(
            ['qstat', '-u', subprocess.run(['whoami'], capture_output=True, text=True).stdout.strip()],
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            # Skip header lines (usually first 2 lines)
            job_lines = [line for line in lines[2:] if line.strip()]
            
            job_names = []
            for line in job_lines:
                # qstat output format: job-ID prior name user state submit/start at queue
                parts = line.split()
                if len(parts) >= 3:
                    job_names.append(parts[2])  # job name is typically 3rd column
            
            return len(job_names), job_names
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    try:
        # Try squeue for SLURM
        result = subprocess.run(
            ['squeue', '-u', subprocess.run(['whoami'], capture_output=True, text=True).stdout.strip(), 
             '-h', '-o', '%j'],  # -h: no header, -o %j: only job name
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            job_names = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return len(job_names), job_names
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return 0, []


def submit_queue_throttled(
    items: List[Any],
    submit_func: Callable,
    max_concurrent: int = 2,
    check_interval: int = 10,
    item_to_args: Optional[Callable] = None,
    verbose: bool = True
):
    """
    Submit jobs from a queue while maintaining a maximum number of concurrent jobs.
    
    Monitors running jobs and submits new ones from the queue when capacity is available.
    Continues until all items in the queue have been submitted.
    
    Args:
        items (List[Any]): Queue of items to process (e.g., rec_ids, file paths).
        submit_func (Callable): Function to call for each item. Should accept keyword args.
        max_concurrent (int): Maximum number of concurrent jobs allowed.
        check_interval (int): Seconds to wait between job count checks.
        item_to_args (Callable, optional): Function that converts an item to kwargs dict
            for submit_func. If None, assumes items are rec_ids and creates 
            {'rec_ids': None, 'extra_args': ['--rec-id', item], 'job_prefix': f'job_{item}'}.
        verbose (bool): Print status messages.
    
    Returns:
        None
    
    Examples:
        # Simple: just rec_ids with default conversion
        rec_ids = ["NP139_B2", "NP140_B1", "NP141_B3"]
        submit_queue_throttled(
            items=rec_ids,
            submit_func=lambda **kwargs: nu.submit_job(
                script="/path/to/script.py",
                python_executable="/path/to/python",
                queue="mind-batch",
                cores=9,
                memory_gb=256,
                **kwargs
            ),
            max_concurrent=2
        )
        
        # Advanced: custom item to args conversion
        def rec_to_args(rec_id):
            return {
                'rec_ids': None,
                'extra_args': ['--rec-id', rec_id, '--verbose'],
                'job_prefix': f'SI_{rec_id}',
                'log_dir': f'/logs/{rec_id}'
            }
        
        submit_queue_throttled(
            items=["NP139_B2", "NP140_B1"],
            submit_func=lambda **kwargs: nu.submit_job(
                script="/userdata/ekato/git_repos/np_se2nwb/SI/run_si_proc.py",
                python_executable="/userdata/ekato/miniforge3/envs/se2nwb/bin/python",
                queue="mind-batch",
                cores=9,
                memory_gb=256,
                use_time=True,
                **kwargs
            ),
            max_concurrent=2,
            item_to_args=rec_to_args
        )
    """
    # Make a copy to avoid modifying original
    queue = items.copy()
    
    # Default item converter for rec_ids
    if item_to_args is None:
        def item_to_args(item):
            return {
                'rec_ids': None,
                'extra_args': ['--rec-id', str(item)],
                'job_prefix': f'job_{item}'
            }
    
    submitted_count = 0
    total_items = len(queue)
    
    if verbose:
        print(f"Starting throttled submission: {total_items} items, max {max_concurrent} concurrent jobs")
    
    while len(queue) > 0:
        job_count, job_names = get_running_jobs()
        
        if job_count < max_concurrent:
            # Submit next job
            item = queue.pop(0)
            kwargs = item_to_args(item)
            
            if verbose:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Submitting job for: {item} ({submitted_count + 1}/{total_items})")
            
            submit_func(**kwargs)
            submitted_count += 1
            
            # Brief pause after submission
            time.sleep(1)
        else:
            if verbose:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Waiting for jobs to finish. {job_count} jobs running. "
                      f"Queue: {len(queue)} remaining.")
            time.sleep(check_interval)
    
    if verbose:
        print(f"All {total_items} jobs submitted successfully!")


def submit_rec_queue(
    rec_ids: List[str],
    script: str,
    python_executable: str,
    queue: str,
    cores: int,
    memory_gb: int,
    log_dir: str,
    job_prefix: str = "job",
    use_time: bool = False,
    max_concurrent: int = 2,
    check_interval: int = 10,
    extra_args_per_rec: Optional[Callable[[str], List[str]]] = None,
    **submit_kwargs
):
    """
    Convenience wrapper for submitting recording IDs with throttling.
    
    Args:
        rec_ids: List of recording IDs to process
        script: Path to script
        python_executable: Python executable path
        queue: Queue name
        cores: CPU cores
        memory_gb: Memory in GB
        log_dir: Log directory
        job_prefix: Base prefix for job names (rec_id will be appended)
        use_time: Use /usr/bin/time wrapper
        max_concurrent: Max concurrent jobs
        check_interval: Check interval in seconds
        extra_args_per_rec: Optional function that takes rec_id and returns extra args list
        **submit_kwargs: Additional kwargs to pass to submit_job
    
    Example:
        submit_rec_queue(
            rec_ids=["NP139_B2", "NP140_B1", "NP141_B3"],
            script="/userdata/ekato/git_repos/np_se2nwb/SI/run_si_proc.py",
            python_executable="/userdata/ekato/miniforge3/envs/se2nwb/bin/python",
            queue="mind-batch",
            cores=9,
            memory_gb=256,
            log_dir="/data_store2/neuropixels/nwb/temp/SI_logs",
            job_prefix="SI",
            use_time=True,
            max_concurrent=2
        )
    """
    def rec_to_args(rec_id):
        extra_args = ['--rec-id', rec_id]
        if extra_args_per_rec:
            extra_args.extend(extra_args_per_rec(rec_id))
        
        return {
            'rec_ids': None,
            'extra_args': extra_args,
            'job_prefix': f'{job_prefix}_{rec_id}',
        }
    
    submit_queue_throttled(
        items=rec_ids,
        submit_func=lambda **kwargs: submit_job(
            script=script,
            python_executable=python_executable,
            queue=queue,
            cores=cores,
            memory_gb=memory_gb,
            log_dir=log_dir,
            use_time=use_time,
            **submit_kwargs,
            **kwargs
        ),
        max_concurrent=max_concurrent,
        check_interval=check_interval,
        item_to_args=rec_to_args
    )
