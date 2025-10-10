"""Logger configuration for the SysCap-CLI application."""

import logging
import os
from datetime import datetime

os.makedirs('logs', exist_ok=True)

LOG_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FILE = f"logs/vansys-cli_{datetime.now().strftime('%Y-%m-%d')}.log"

logger = logging.getLogger('vansys-cli')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

logger.addHandler(console_handler)
logger.addHandler(file_handler)
