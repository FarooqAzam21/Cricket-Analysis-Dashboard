import sqlite3

conn = sqlite3.connect('cricket_dashboard.db')
cursor = conn.cursor()

# Get schema for tournaments table
cursor.execute("PRAGMA table_info(tournaments)")
columns = cursor.fetchall()

print("Tournaments table schema:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# Try inserting manually with all fields
print("\nTrying to insert tournament 3...")
try:
    cursor.execute("""
        INSERT INTO tournaments (name, start_date, end_date, status)
        VALUES (?, ?, ?, ?)
    """, ("Test Tournament 3", "2026-02-01", "2026-02-15", "planning"))
    conn.commit()
    print("✅ Inserted successfully")
except Exception as e:
    print(f"❌ Error: {e}")

# Check again
cursor.execute("SELECT * FROM tournaments WHERE name LIKE '%Tournament 3%'")
result = cursor.fetchone()
if result:
    print(f"\nFound: {result}")
else:
    print("\nNot found after insert")

conn.close()
