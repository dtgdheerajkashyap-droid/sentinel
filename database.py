import sqlite3

conn = sqlite3.connect("sentinelai.db")
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

conn.commit()
conn.close()

print("Database created successfully.")