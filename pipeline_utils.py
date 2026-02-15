import logging
import time
import json
import sys
from pathlib import Path
from functools import wraps
from typing import Callable, Any, Dict, List, Optional
import asyncio

import tenacity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

# ──────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────
LOG_FILE = Path("output/pipeline.log")

class CustomFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    def format(self, record):
        timestamp = self.formatTime(record, self.datefmt)
        return f"{timestamp} | {record.levelname:<7} | {record.name:<15} | {record.getMessage()}"

def setup_logging():
    """Configure logging to both console and file."""
    # Ensure output directory exists (handled in main.py usually, but safe to check)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("VideoPipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers

    # Console Handler (INFO+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    # File Handler (DEBUG+)
    file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(CustomFormatter())
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()

# ──────────────────────────────────────────────
# Error Tracking
# ──────────────────────────────────────────────
class ErrorTracker:
    """Track successes and failures for report generation."""
    def __init__(self):
        self.errors = []
        self.provider_stats = {}  # { "Pexels": {"success": 0, "fail": 0}, ... }

    def record_attempt(self, provider: str, success: bool, error_msg: str = None, latency: float = 0.0):
        if provider not in self.provider_stats:
            self.provider_stats[provider] = {"success": 0, "fail": 0, "total_latency": 0.0}
        
        stats = self.provider_stats[provider]
        if success:
            stats["success"] += 1
        else:
            stats["fail"] += 1
            self.errors.append({
                "provider": provider,
                "error": str(error_msg),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        stats["total_latency"] += latency

    def generate_report(self, output_path: Path):
        """Generate structured JSON error report."""
        report = {
            "summary": "Pipeline Execution Report",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider_stats": self.provider_stats,
            "errors": self.errors
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Error report saved to {output_path}")

# Global tracker instance
tracker = ErrorTracker()

# ──────────────────────────────────────────────
# Retry Logic
# ──────────────────────────────────────────────

# Retry configuration
# Exponential backoff: 1s, 2s, 4s... max 3 attempts
RETRY_CONFIG = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=1, min=1, max=10),
    "retry": retry_if_exception_type(Exception),
    "before_sleep": before_sleep_log(logger, logging.WARNING),
    "reraise": True
}

def log_attempt(provider_name: str):
    """Decorator to log API call attempts and track latency."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                if result is not None: # success
                    logger.info(f"[{provider_name}] Success ({latency:.2f}s)")
                    tracker.record_attempt(provider_name, True, latency=latency)
                else: # explicit None return (often used for failures in this codebase)
                    logger.warning(f"[{provider_name}] Returned None ({latency:.2f}s)")
                    tracker.record_attempt(provider_name, False, error_msg="Returned None", latency=latency)
                return result
            except Exception as e:
                latency = time.time() - start_time
                logger.error(f"[{provider_name}] Failed: {e} ({latency:.2f}s)")
                tracker.record_attempt(provider_name, False, error_msg=str(e), latency=latency)
                raise e
        return wrapper
    return decorator

def retry_api_call(provider_name: str):
    """
    Decorator that adds retry logic AND logging/tracking.
    Usage: @retry_api_call("Gemini")
    """
    def decorator(func):
        # We combine tenacity retry with our logging wrapper
        # Order matters: Retry(Log(Func)) -> Retry catches exceptions from Log wrapper
        
        # Apply logging/tracking first (inner)
        logged_func = log_attempt(provider_name)(func)
        
        # Apply retry (outer)
        retried_func = tenacity.retry(**RETRY_CONFIG)(logged_func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retried_func(*args, **kwargs)
        return wrapper
    return decorator

# Async version for Edge TTS
def retry_async_api_call(provider_name: str):
    """Async version of retry decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
             async for attempt in tenacity.AsyncRetrying(**RETRY_CONFIG):
                with attempt:
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        latency = time.time() - start_time
                        logger.info(f"[{provider_name}] Success ({latency:.2f}s)")
                        tracker.record_attempt(provider_name, True, latency=latency)
                        return result
                    except Exception as e:
                        latency = time.time() - start_time
                        logger.error(f"[{provider_name}] Failed: {e} ({latency:.2f}s)")
                        tracker.record_attempt(provider_name, False, error_msg=str(e), latency=latency)
                        raise e
        return wrapper
    return decorator
