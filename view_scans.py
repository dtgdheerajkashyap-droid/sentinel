import sqlite3

conn = sqlite3.connect("sentinelai.db")
cursor = conn.cursor()

cursor.execute("""
SELECT
    id,
    sender,
    subject,
    risk_score,
    risk_level,
    urls_found,
    urgency_risk,
    scanned_at
FROM email_scans
ORDER BY id DESC
""")

rows = cursor.fetchall()

print(f"\nTotal scans stored: {len(rows)}\n")

for row in rows:
    print("=" * 80)
    print("ID           :", row[0])
    print("Sender       :", row[1])
    print("Subject      :", row[2])
    print("Risk Score   :", row[3])
    print("Risk Level   :", row[4])
    print("URLs Found   :", row[5])
    print("Urgency Risk :", row[6])
    print("Scanned At   :", row[7])

conn.close()