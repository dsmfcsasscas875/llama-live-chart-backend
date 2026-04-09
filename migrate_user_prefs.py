import sqlite3
import os

DB_PATH = None
for root, dirs, files in os.walk(os.path.dirname(__file__)):
    for f in files:
        if f.endswith('.db'):
            DB_PATH = os.path.join(root, f)
            break
    if DB_PATH:
        break

if not DB_PATH:
    exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(user)")
columns = [row[1] for row in cursor.fetchall()]

if 'email_alerts_enabled' not in columns:
    cursor.execute("ALTER TABLE user ADD COLUMN email_alerts_enabled BOOLEAN DEFAULT 0")
if 'alert_time' not in columns:
    cursor.execute("ALTER TABLE user ADD COLUMN alert_time TEXT DEFAULT '18:00'")

conn.commit()
conn.close()
print("Success: Added user prefs columns.")
