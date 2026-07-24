import sqlite3

DB_PATH = "sentinelai.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE email_scans ADD COLUMN scan_summary TEXT")
    print("Added column: scan_summary")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e).lower():
        print("Column already exists: scan_summary")
    else:
        print("Database error:", e)

conn.commit()
conn.close()

print("Database summary upgrade completed.")