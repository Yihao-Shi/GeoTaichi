# Job Scheduler

A Python library for scheduling and running jobs across multiple GPUs with automatic load balancing and resource management.

> **⚠️ Warning**: This code was generated and written by Cursor AI. Please review and test thoroughly before using in production environments.

## Features

- **GPU Load Balancing**: Automatically distributes jobs across available GPUs based on memory usage
- **Resource Management**: Limits the number of concurrent jobs per GPU
- **Graceful Shutdown**: Handles termination signals gracefully
- **Real-time Monitoring**: Monitors GPU utilization and memory usage
- **Thread-safe**: Uses locks to ensure thread safety when accessing shared resources

## Installation

### Install from source

```bash
pip install -e .
```

### For development

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from job_scheduler import schedule_jobs

# Example commands to run
commands = [
    "python train_model.py --config config1.yaml",
    "python train_model.py --config config2.yaml", 
    "python train_model.py --config config3.yaml",
]

# Specify the GPUs to use
visible_gpus = [0, 1]  # Use GPU 0 and 1

# Set maximum jobs per GPU
max_jobs_per_gpu = 2

# Schedule and run the jobs
schedule_jobs(commands, visible_gpus, max_jobs_per_gpu)
```

## API Reference

### Functions

#### `schedule_jobs(commands, visible_gpus, max_jobs_per_gpu)`

Schedule and run commands across available GPUs.

**Parameters:**
- `commands` (list): A list of shell commands to execute
- `visible_gpus` (list): A list of GPU IDs to consider for scheduling
- `max_jobs_per_gpu` (int): The maximum number of jobs allowed per GPU

#### `get_gpu_utilization(visible_gpus)`

Get the utilization of specified GPUs.

**Parameters:**
- `visible_gpus` (list): A list of GPU IDs to query

**Returns:**
- dict: Dictionary with GPU IDs as keys and utilization percentages as values

#### `get_gpu_memory_usage(visible_gpus)`

Get the memory usage of specified GPUs.

**Parameters:**
- `visible_gpus` (list): A list of GPU IDs to query

**Returns:**
- dict: Dictionary with GPU IDs as keys and memory usage percentages as values

### Classes

#### `GracefulKiller`

Handles termination signals gracefully, allowing for clean shutdown of running jobs.

## Requirements

- Python 3.7+
- NVIDIA drivers with nvidia-smi command available
- CUDA-capable GPUs

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 