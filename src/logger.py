import logging
import os
from datetime import datetime

# Correct timestamp format for the file name
LOG_FILE_NAME = datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + ".log"
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)  # Ensure the logs directory exists

# Correct path joining for the log file
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE_NAME)

# Correct logging setup
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(levelname)s - %(message)s",
    level=logging.INFO  # Correct logging level
)

if __name__ == "__main__":
    logging.info("Logging has started")

