"""
Quick migration script to add 'notes' column to watchlistitem table.
Run once: python migrate_add_notes.py
"""
import sqlite3
import os

# Find the DB file
DB_PATH = None
for root, dirs, files in os.walk(os.path.dirname(__file__)):
    for f in files:
        if f.endswith('.db'):
            DB_PATH = os.path.join(root, f)
            break
    if DB_PATH:
        break

if not DB_PATH:
    print("❌ No SQLite .db file found. Check your DATABASE_URI in config.")
    exit(1)

print(f"📂 Found database: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check if column already exists
cursor.execute("PRAGMA table_info(watchlistitem)")
columns = [row[1] for row in cursor.fetchall()]

if 'notes' in columns:
    print("✅ 'notes' column already exists in watchlistitem. Nothing to do.")
else:
    cursor.execute("ALTER TABLE watchlistitem ADD COLUMN notes TEXT DEFAULT ''")
    conn.commit()
    print("✅ Successfully added 'notes' column to watchlistitem table.")

conn.close()
