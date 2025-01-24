import json
import sqlite3

from sgir_distserve.env import (
    CHUNK_PROFILER_DB_NAME,
    DECODE_PROFILER_DB_NAME,
    PREFILL_PROFILER_DB_NAME,
    SGIR_DISTSERVE_CHUNK_DB_PATH,
    SGIR_DISTSERVE_DECODE_DB_PATH,
    SGIR_DISTSERVE_PREFILL_DB_PATH,
)


def insert_chunk_profiler(model, tp, pp, chunk_size, coeff_0, coeff_1):
    conn = sqlite3.connect(SGIR_DISTSERVE_CHUNK_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        f"""
    INSERT OR REPLACE INTO {CHUNK_PROFILER_DB_NAME} (model, tp, pp, chunk_size, coeff_0, coeff_1)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
        (model, tp, pp, chunk_size, coeff_0, coeff_1),
    )
    conn.commit()
    conn.close()


def query_chunk_profiler(model, tp, pp, chunk_size):
    conn = sqlite3.connect(SGIR_DISTSERVE_CHUNK_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        f"""
    SELECT * FROM {CHUNK_PROFILER_DB_NAME} WHERE model = ? AND tp = ? AND pp = ? AND chunk_size = ?
    """,
        (model, tp, pp, chunk_size),
    )

    rows = cursor.fetchall()
    conn.close()
    return rows


def insert_prefill_profiler(model, tp, pp, chunk_size, coeff_0, coeff_1, coeff_2):
    conn = sqlite3.connect(SGIR_DISTSERVE_PREFILL_DB_PATH)
    cursor = conn.cursor()
    print(chunk_size, PREFILL_PROFILER_DB_NAME, SGIR_DISTSERVE_PREFILL_DB_PATH, model)

    cursor.execute(
        f"""
    INSERT OR REPLACE INTO {PREFILL_PROFILER_DB_NAME} (model, tp, pp, chunk_size, coeff_0, coeff_1, coeff_2)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (model, tp, pp, chunk_size, coeff_0, coeff_1, coeff_2),
    )
    conn.commit()
    conn.close()


def query_prefill_profiler(model, tp, pp, chunk_size):
    conn = sqlite3.connect(SGIR_DISTSERVE_PREFILL_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        f"""
    SELECT * FROM {PREFILL_PROFILER_DB_NAME} WHERE model = ? AND tp = ? AND pp = ? AND chunk_size = ?
    """,
        (model, tp, pp, chunk_size),
    )

    rows = cursor.fetchall()
    conn.close()
    return rows


def insert_decode_profiler(model, tp, pp, coeff_0, coeff_1):
    conn = sqlite3.connect(SGIR_DISTSERVE_DECODE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        f"""
    INSERT OR REPLACE INTO {DECODE_PROFILER_DB_NAME} (model, tp, pp, coeff_0, coeff_1)
    VALUES (?, ?, ?, ?, ?)
    """,
        (model, tp, pp, coeff_0, coeff_1),
    )
    conn.commit()
    conn.close()


def query_decode_profiler(model, tp, pp):
    conn = sqlite3.connect(SGIR_DISTSERVE_DECODE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        f"""
    SELECT * FROM {DECODE_PROFILER_DB_NAME} WHERE model = ? AND tp = ? AND pp = ?
    """,
        (model, tp, pp),
    )

    rows = cursor.fetchall()
    conn.close()
    return rows
