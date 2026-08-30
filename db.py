# -*- coding: utf-8 -*-
"""
db.py
Database schema creation, connection helper, and default criteria seeding
for the Agentic RFP Evaluation project.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation.db")

DEFAULT_CRITERIA = [
    # name, description, weight, max_score
    ("Technical Capability", "Architecture, integrations, scalability, technical fit", 30.0, 10),
    ("Implementation Plan", "Timeline, milestones, staffing, risk plan", 20.0, 10),
    ("Commercial Value", "Pricing clarity, total cost, assumptions", 20.0, 10),
    ("Security & Compliance", "Controls, certifications, privacy, auditability", 20.0, 10),
    ("Support & Experience", "Support model, similar projects, references", 10.0, 10),
]


def get_conn() -> sqlite3.Connection:
    """Returns a SQLite connection with foreign key enforcement enabled.
    PRAGMA foreign_keys is per-connection in SQLite, so it must be set every time."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Creates all tables if they do not already exist. Safe to call on every app start."""
    conn = get_conn()
    cursor = conn.cursor()

    # Parent table first: evaluation_criteria has no dependencies
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_criteria (
            criterion_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            weight REAL NOT NULL,
            max_score INTEGER DEFAULT 100,
            is_active INTEGER DEFAULT 1
        )
    ''')

    # rfp_runs has no dependencies either, but must exist before supplier_results (FK target)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rfp_runs (
            rfp_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
            completed_at TEXT,
            status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'failed')),
            criteria_snapshot_json TEXT,
            supplier_count INTEGER DEFAULT 0,
            error_message TEXT
        )
    ''')

    # supplier_results references rfp_runs via FK, with cascade delete
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS supplier_results (
            supplier_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            rfp_run_id INTEGER NOT NULL,
            supplier_name TEXT NOT NULL,
            submission_date TEXT,
            experience_rating REAL,
            source_filename TEXT,
            status TEXT NOT NULL CHECK(status IN ('evaluated', 'failed', 'pending', 'processing')),
            validation_warnings TEXT,
            absolute_score REAL,
            ppi REAL,
            final_rank INTEGER,
            result_json TEXT,
            evaluated_at TEXT DEFAULT (CURRENT_TIMESTAMP),
            FOREIGN KEY (rfp_run_id) REFERENCES rfp_runs (rfp_run_id) ON DELETE CASCADE
        )
    ''')

    conn.commit()

    # Seed default criteria only if the table is currently empty, so re-runs never duplicate rows
    cursor.execute("SELECT COUNT(*) FROM evaluation_criteria")
    count = cursor.fetchone()[0]
    if count == 0:
        cursor.executemany(
            "INSERT INTO evaluation_criteria (name, description, weight, max_score, is_active) "
            "VALUES (?, ?, ?, ?, 1)",
            DEFAULT_CRITERIA,
        )
        conn.commit()

    conn.close()


def get_active_criteria() -> dict:
    """Returns active criteria as {criterion_id: {name, weight, max_score}}."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT criterion_id, name, weight, max_score FROM evaluation_criteria WHERE is_active = 1"
    )
    rows = cursor.fetchall()
    conn.close()
    return {
        row[0]: {"name": row[1], "weight": row[2], "max_score": row[3]}
        for row in rows
    }


def get_all_criteria_rows() -> list:
    """Returns all criteria rows (active and inactive) for display purposes."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT criterion_id, name, description, weight, max_score, is_active FROM evaluation_criteria"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def list_runs() -> list:
    """Returns every batch/run, most recent first."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT rfp_run_id, created_at, completed_at, status, supplier_count "
        "FROM rfp_runs ORDER BY rfp_run_id DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_run(run_id: int):
    """Returns one run's metadata, including its frozen criteria snapshot."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT rfp_run_id, created_at, completed_at, status, criteria_snapshot_json, "
        "supplier_count, error_message FROM rfp_runs WHERE rfp_run_id = ?",
        (run_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return row


def get_supplier_results(run_id: int) -> list:
    """Returns all supplier results for one batch, ranked suppliers first (final_rank asc),
    failed/unranked suppliers last."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT supplier_result_id, supplier_name, submission_date, experience_rating, "
        "source_filename, status, validation_warnings, absolute_score, ppi, final_rank, "
        "result_json, evaluated_at FROM supplier_results WHERE rfp_run_id = ? "
        "ORDER BY (final_rank IS NULL), final_rank ASC",
        (run_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows
