import sqlite3

DB_PATH = "sentinelai.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE email_scans ADD COLUMN reasons TEXT")
    print("Added column: reasons")
except sqlite3.OperationalError:
    print("Column already exists: reasons")

try:
    cursor.execute("ALTER TABLE email_scans ADD COLUMN detected_urls TEXT")
    print("Added column: detected_urls")
except sqlite3.OperationalError:
    print("Column already exists: detected_urls")

conn.commit()
conn.close()

print("Database upgraded successfully.")