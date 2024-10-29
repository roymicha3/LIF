import gc
import torch
import time
import logging

class GpuMemoryTracker:
    def __init__(self, log_file=None, warning_threshold=0.8, verbosity=1):
        self.log_file = log_file
        self.warning_threshold = warning_threshold
        self.verbosity = verbosity
        self.tensors = []  # Track tensors for detecting large allocations

        # Configure logging if a log file is provided
        if log_file:
            logging.basicConfig(filename=log_file, level=logging.INFO, 
                                format='%(asctime)s - %(message)s')

    def __enter__(self):
        self.start_memory = torch.cuda.memory_allocated()
        self.start_time = time.time()
        if self.verbosity > 1:
            self._log_message(f"Starting memory: {self.start_memory / (1024 ** 2):.2f} MB", level="info")
        self._check_memory()  # Check initial memory
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_memory = torch.cuda.memory_allocated()
        self.end_time = time.time()
        self._report_memory_usage()
        self._detect_large_tensors()
        
        gc.collect()
        torch.cuda.empty_cache()  # Clear unused memory

        # Log any exceptions that occurred
        if exc_type is not None:
            self._log_message(f"Exception occurred: {exc_val}", level="error")

    def track_tensor(self, tensor):
        """Adds a tensor to tracking for large variable detection."""
        self.tensors.append(tensor)

    def _check_memory(self):
        reserved_memory = torch.cuda.memory_reserved()
        allocated_memory = torch.cuda.memory_allocated()
        cached_memory = reserved_memory - allocated_memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        usable_memory = total_memory - reserved_memory
        used_memory_percentage = allocated_memory / usable_memory if usable_memory > 0 else 0

        if used_memory_percentage > self.warning_threshold:
            warning_msg = (
                f"Warning: GPU memory usage is at {used_memory_percentage * 100:.2f}% "
                f"({allocated_memory / (1024 ** 2):.2f} MB allocated / {usable_memory / (1024 ** 2):.2f} MB usable). "
                "Consider reducing model size."
            )
            self._log_message(warning_msg, level="warning")

        
        cached_msg = f"Cached memory: {cached_memory / (1024 ** 2):.2f} MB"
        self._log_message(cached_msg, level="warning")

    def _report_memory_usage(self):
        memory_used = (self.end_memory - self.start_memory) / (1024 ** 2)
        elapsed_time = self.end_time - self.start_time

        report = (
            f"Ending memory: {self.end_memory / (1024 ** 2):.2f} MB\n"
            f"Memory used: {memory_used:.2f} MB\n"
            f"Elapsed time: {elapsed_time:.2f} seconds\n"
        )
        self._log_message(report, level="info")

    def _detect_large_tensors(self):
        """Detects and reports large tensors that are occupying significant memory."""
        largest_tensor = None
        largest_size = 0

        for tensor in self.tensors:
            size = tensor.element_size() * tensor.numel() / (1024 ** 2)  # Size in MB
            if size > largest_size:
                largest_size = size
                largest_tensor = tensor

        if largest_tensor is not None and largest_size > 0:
            large_tensor_msg = (
                f"Largest tensor: {largest_size:.2f} MB. "
                "Consider releasing or reducing this tensor if memory is tight."
            )
            self._log_message(large_tensor_msg, level="warning")

    def _log_message(self, message, level="info"):
        """Logs or prints messages based on verbosity and log level."""
        if level == "error" or (level == "warning" and self.verbosity > 0) or (level == "info" and self.verbosity > 1):
            if self.log_file:
                if level == "info":
                    logging.info(message)
                elif level == "warning":
                    logging.warning(message)
                elif level == "error":
                    logging.error(message)
            else:
                print(message)