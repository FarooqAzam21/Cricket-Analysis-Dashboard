import sqlite3

conn = sqlite3.connect('cricket_dashboard.db')
conn.row_factory = sqlite3.Row

# Check tournament 3
tournament = conn.execute('SELECT * FROM tournaments WHERE id = 3').fetchone()
if tournament:
    print('Tournament 3 found:')
    print(f'  Name: {tournament["name"]}')
    print(f'  Status: {tournament["status"]}')
    print(f'  Start: {tournament["start_date"]}')
    print(f'  End: {tournament["end_date"]}')
else:
    print('Tournament ID 3 not found')

# Show all tournaments with status
print('\nAll tournaments:')
all_tourn = conn.execute('SELECT id, name, status FROM tournaments').fetchall()
for t in all_tourn:
    print(f'  ID {t[0]}: {t[1]} (Status: {t[2]})')

conn.close()
