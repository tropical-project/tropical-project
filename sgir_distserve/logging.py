import logging
import os

home_dir = os.path.join(os.environ["HOME"], "")
SCALING_LOGGING_PATH = os.path.join(home_dir, ".sgir_distserve/scaling_log.txt")
SLO_LOGGING_PATH = os.path.join(home_dir, ".sgir_distserve/slo_log.txt")

scaling_file_handler = logging.FileHandler(SCALING_LOGGING_PATH)
scaling_file_handler.setLevel(logging.INFO)

slo_file_handler = logging.FileHandler(SLO_LOGGING_PATH)
slo_file_handler.setLevel(logging.INFO)

# 创建一个handler，用于输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建formatter并添加到handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
slo_file_handler.setFormatter(formatter)
scaling_file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

scaling_logger = logging.getLogger("scaling_logger")
scaling_logger.setLevel(logging.INFO)
scaling_logger.addHandler(scaling_file_handler)
scaling_logger.addHandler(console_handler)

slo_logger = logging.getLogger("slo_logger")
slo_logger.setLevel(logging.INFO)
slo_logger.addHandler(slo_file_handler)
slo_logger.addHandler(console_handler)
