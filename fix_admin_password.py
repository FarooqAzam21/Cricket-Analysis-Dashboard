import sqlite3
import hashlib

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(str.encode(password)).hexdigest()

conn = sqlite3.connect('cricket_dashboard.db')
c = conn.cursor()

# Update admin password to hashed 'admin'
hashed_admin = hash_password('admin')
c.execute("UPDATE users SET password = ? WHERE username = 'admin'", (hashed_admin,))
conn.commit()

# Verify
c.execute("SELECT username, password FROM users WHERE username = 'admin'")
result = c.fetchone()

if result:
    print("✅ Admin account fixed successfully!")
    print(f"   Username: {result[0]}")
    print(f"   Password (hashed): {result[1][:16]}...")
    print("\n✅ You can now login with:")
    print("   Username: admin")
    print("   Password: admin")
else:
    print("❌ Admin account not found")

conn.close()
