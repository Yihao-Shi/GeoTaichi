"""
Job Scheduler - A GPU job scheduling library for distributed computing.

This package provides utilities for scheduling and running jobs across multiple GPUs
with automatic load balancing and resource management.
"""

from .scheduler import schedule_jobs, GracefulKiller, get_gpu_utilization, get_gpu_memory_usage

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "schedule_jobs",
    "GracefulKiller", 
    "get_gpu_utilization",
    "get_gpu_memory_usage"
] 