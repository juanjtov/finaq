"""Shared logging configuration and the tenacity retry decorator used by every external-call site."""

from __future__ import annotations

import logging
import os

import httpx
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("finaq")

RETRYABLE_EXCEPTIONS = (
    httpx.HTTPError,
    httpx.TimeoutException,
    ConnectionError,
    OSError,
    TimeoutError,
)

tenacity_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    reraise=True,
)
