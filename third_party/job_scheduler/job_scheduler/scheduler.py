import subprocess
import time
import signal
from threading import Thread, Lock
from collections import defaultdict

class GracefulKiller:
    """Gracefully handles termination signals."""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("\nTermination signal received. Exiting gracefully...")
        self.kill_now = True

def get_gpu_utilization(visible_gpus):
    """
    Get the utilization of GPUs using nvidia-smi.
    Filters to only consider the specified visible GPUs.
    Returns a dictionary where keys are GPU IDs and values are utilization percentages.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        gpu_utilizations = list(map(int, result.stdout.strip().split("\n")))
        visible_gpu_utilizations = {gpu_id: gpu_utilizations[gpu_id] for gpu_id in visible_gpus}
        return visible_gpu_utilizations
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found. Ensure NVIDIA drivers are installed and accessible.")

def get_gpu_memory_usage(visible_gpus):
    """
    Get the memory usage of the visible GPUs.
    Returns a dictionary with GPU IDs as keys and memory usage percentage as values.
    """
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    gpu_info = result.stdout.decode('utf-8').strip().split('\n')

    memory_usage = {}
    for info in gpu_info:
        gpu_id, used_memory, total_memory = map(int, info.split(','))
        if gpu_id in visible_gpus:
            memory_usage[gpu_id] = used_memory / total_memory * 100

    return memory_usage

def find_free_gpu(visible_gpus, assigned_gpus, gpu_job_counts, max_jobs_per_gpu, lock):
    """
    Find the first GPU with low utilization and below the max job limit among the visible GPUs.
    Returns the GPU ID if available, or None if all are busy.
    """
    with lock:
        memory_usage = get_gpu_memory_usage(visible_gpus)
        min_jobs = float('inf')
        selected_gpu = None
        for gpu_id, usage in memory_usage.items():
            if gpu_id in visible_gpus and (gpu_id not in assigned_gpus or gpu_job_counts[gpu_id] < max_jobs_per_gpu):
                if gpu_job_counts[gpu_id] < min_jobs:
                    min_jobs = gpu_job_counts[gpu_id]
                    selected_gpu = gpu_id
        return selected_gpu
    return None

def run_job_on_gpu(command, gpu_id, gpu_job_counts, lock):
    """
    Run a command on a specific GPU.
    Args:
        command (str): The command to execute.
        gpu_id (int): The GPU ID to use.
        gpu_job_counts (dict): A dictionary tracking the number of jobs per GPU.
        lock (Lock): A threading lock to synchronize access.
    """
    env = {**subprocess.os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    print(f"Running on GPU {gpu_id}: {command}")
    subprocess.run(command, shell=True, env=env)

    # Allow GPU job count to decrease after the job completes
    with lock:
        gpu_job_counts[gpu_id] -= 1

def schedule_jobs(commands, visible_gpus, max_jobs_per_gpu):
    """
    Schedule and run commands across available GPUs.
    Args:
        commands (list): A list of commands to execute.
        visible_gpus (list): A list of GPU IDs to consider for scheduling.
        max_jobs_per_gpu (int): The maximum number of jobs allowed per GPU.
    """
    threads = []
    gpu_job_counts = defaultdict(int)
    lock = Lock()
    killer = GracefulKiller()

    for command in commands:
        while not killer.kill_now:
            gpu_id = find_free_gpu(visible_gpus, set(gpu_job_counts.keys()), gpu_job_counts, max_jobs_per_gpu, lock)
            if gpu_id is not None:
                with lock:
                    gpu_job_counts[gpu_id] += 1
                thread = Thread(target=run_job_on_gpu, args=(command, gpu_id, gpu_job_counts, lock))
                threads.append(thread)
                thread.start()
                break
            else:
                print("All GPUs are busy or at max capacity. Waiting...")
                time.sleep(5)  # Wait before checking again
        if killer.kill_now:
            break

    # Wait for threads to finish
    for thread in threads:
        thread.join() 