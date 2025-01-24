import os
import sqlite3

SGIR_DISTSERVE_LOG_PATH = os.getenv("SGIR_DISTSERVE_LOG_PATH", ".logs/")
os.makedirs(SGIR_DISTSERVE_LOG_PATH, exist_ok=True)

SGIR_DISTSERVE_RESULT_PATH = os.getenv("SGIR_DISTSERVE_RESULT_PATH", ".result/")
os.makedirs(SGIR_DISTSERVE_RESULT_PATH, exist_ok=True)

SGIR_DISTSERVE_PROFILER_PATH = os.getenv("SGIR_DISTSERVE_PROFILER_PATH", ".profiler")
os.makedirs(SGIR_DISTSERVE_PROFILER_PATH, exist_ok=True)

SGIR_DISTSERVE_PREFILL_DB_PATH = os.path.join(
    SGIR_DISTSERVE_PROFILER_PATH,
    "prefill.db",
)

SGIR_DISTSERVE_CHUNK_DB_PATH = os.path.join(
    SGIR_DISTSERVE_PROFILER_PATH,
    "chunk.db",
)

SGIR_DISTSERVE_DECODE_DB_PATH = os.path.join(
    SGIR_DISTSERVE_PROFILER_PATH,
    "decode.db",
)


CHUNK_PROFILER_DB_NAME = "chunk_execution_coefficients"
PREFILL_PROFILER_DB_NAME = "prefill_execution_coefficients"
DECODE_PROFILER_DB_NAME = "decode_execution_coefficients"

# create db
conn = sqlite3.connect(SGIR_DISTSERVE_PREFILL_DB_PATH)
cursor = conn.cursor()

cursor.execute(
    f"""
CREATE TABLE IF NOT EXISTS {PREFILL_PROFILER_DB_NAME} (
    model TEXT NOT NULL,
    tp TEXT NOT NULL,
    pp TEXT NOT NULL,
    chunk_size INTEGER,
    coeff_0 REAL NOT NULL,
    coeff_1 REAL NOT NULL,
    coeff_2 REAL NOT NULL,
    PRIMARY KEY (model, tp, pp, chunk_size)
)
"""
)

# create db
conn = sqlite3.connect(SGIR_DISTSERVE_CHUNK_DB_PATH)
cursor = conn.cursor()
# prefill execution time

# chunk execution time
cursor.execute(
    f"""
CREATE TABLE IF NOT EXISTS {CHUNK_PROFILER_DB_NAME} (
    model TEXT NOT NULL,
    tp INTEGER NOT NULL,
    pp INTEGER NOT NULL,
    chunk_size INTEGER NOT NULL,
    coeff_0 REAL NOT NULL,
    coeff_1 REAL NOT NULL,
    PRIMARY KEY (model, tp, pp, chunk_size)
)
"""
)

# create db
conn = sqlite3.connect(SGIR_DISTSERVE_DECODE_DB_PATH)
cursor = conn.cursor()
# prefill execution time

# decode execution time
cursor.execute(
    f"""
CREATE TABLE IF NOT EXISTS {DECODE_PROFILER_DB_NAME} (
    model TEXT NOT NULL,
    tp INTEGER NOT NULL,
    pp INTEGER NOT NULL,
    coeff_0 REAL NOT NULL,
    coeff_1 REAL NOT NULL,
    PRIMARY KEY (model, tp, pp)
)
"""
)
