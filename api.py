import os
import sys
import sqlite3
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="SentinelAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT_DIR, "sentinelai.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def ensure_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS email_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT UNIQUE NOT NULL,
            sender TEXT,
            subject TEXT,
            risk_score INTEGER,
            risk_level TEXT,
            urls_found INTEGER,
            urgency_risk INTEGER,
            scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    extra_columns = {
        "reasons": "TEXT",
        "detected_urls": "TEXT",
        "scan_summary": "TEXT",
        "action_status": "TEXT DEFAULT 'pending'",
    }

    cursor.execute("PRAGMA table_info(email_scans)")
    existing_columns = [column[1] for column in cursor.fetchall()]

    for column_name, column_type in extra_columns.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE email_scans ADD COLUMN {column_name} {column_type}"
            )

    cursor.execute("""
        UPDATE email_scans
        SET action_status = 'pending'
        WHERE action_status IS NULL OR action_status = ''
    """)

    conn.commit()
    conn.close()


def calculate_stats():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM email_scans")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM email_scans WHERE risk_level='LOW RISK'")
    low = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM email_scans WHERE risk_level='MEDIUM RISK'")
    medium = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM email_scans WHERE risk_level='HIGH RISK'")
    high = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(risk_score) FROM email_scans")
    avg = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM email_scans WHERE action_status='marked_safe'")
    marked_safe = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM email_scans WHERE action_status='quarantined'")
    quarantined = cursor.fetchone()[0]

    conn.close()

    return {
        "total_emails": total,
        "low_risk": low,
        "medium_risk": medium,
        "high_risk": high,
        "average_risk": round(avg or 0, 2),
        "marked_safe": marked_safe,
        "quarantined": quarantined,
    }


@app.get("/")
def home():
    return {"message": "SentinelAI API is running"}


@app.get("/api/stats")
def get_stats():
    ensure_database()
    return calculate_stats()


@app.get("/api/scans")
def get_scans():
    ensure_database()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, sender, subject, risk_score, risk_level,
               urls_found, urgency_risk, scanned_at, reasons,
               detected_urls, scan_summary, action_status
        FROM email_scans
        ORDER BY scanned_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    scans = []

    for row in rows:
        scans.append({
            "id": row[0],
            "sender": row[1],
            "subject": row[2],
            "risk_score": row[3],
            "risk_level": row[4],
            "urls_found": row[5],
            "urgency_risk": row[6],
            "scanned_at": row[7],
            "reasons": row[8] if row[8] else "No major risk detected",
            "detected_urls": row[9] if row[9] else "No URLs found",
            "scan_summary": row[10] if row[10] else "No summary available.",
            "action_status": row[11] if row[11] else "pending",
        })

    return scans


def update_action_status(scan_id: int, status: str):
    ensure_database()

    allowed_statuses = ["pending", "marked_safe", "quarantined"]

    if status not in allowed_statuses:
        raise HTTPException(status_code=400, detail="Invalid action status")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE email_scans
        SET action_status = ?
        WHERE id = ?
    """, (status, scan_id))

    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Scan not found")

    conn.commit()
    conn.close()

    return {
        "message": "Action status updated",
        "scan_id": scan_id,
        "action_status": status,
    }


@app.post("/api/scans/{scan_id}/mark-safe")
def mark_scan_safe(scan_id: int):
    return update_action_status(scan_id, "marked_safe")


@app.post("/api/scans/{scan_id}/quarantine")
def quarantine_scan(scan_id: int):
    return update_action_status(scan_id, "quarantined")
@app.get("/api/top-risky-senders")
def top_risky_senders():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            sender,
            COUNT(*) as detections,
            MAX(scanned_at) as last_seen,
            MAX(risk_score) as max_score
        FROM email_scans
        GROUP BY sender
        ORDER BY detections DESC
        LIMIT 5
    """).fetchall()

    conn.close()

    data = []

    for row in rows:

        score = row["max_score"]

        if score >= 70:
            level = "high"
        elif score >= 40:
            level = "medium"
        else:
            level = "safe"

        data.append({
            "email": row["sender"],
            "detections": row["detections"],
            "lastSeen": row["last_seen"],
            "level": level
        })

    return data


@app.post("/api/scan-inbox")
def scan_inbox():
    ensure_database()

    before_stats = calculate_stats()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [sys.executable, "sentinel_ai_v2.py"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            env=env,
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Inbox scan timed out. Try again with fewer unread emails.",
        )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=result.stderr or result.stdout or "Scanner failed.",
        )

    after_stats = calculate_stats()

    new_scans = max(
        after_stats["total_emails"] - before_stats["total_emails"],
        0,
    )

    return {
        "message": "Inbox scan completed",
        "scanned": new_scans,
        "new_scans": new_scans,
        "total_emails": after_stats["total_emails"],
        "low_risk": after_stats["low_risk"],
        "medium_risk": after_stats["medium_risk"],
        "high_risk": after_stats["high_risk"],
        "average_risk": after_stats["average_risk"],
        "marked_safe": after_stats["marked_safe"],
        "quarantined": after_stats["quarantined"],
        "highRisk": after_stats["high_risk"],
    }


ensure_database()