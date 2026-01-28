import csv
import os
import re

def is_garbage(val):
    v = val.lower()
    # URL path fragments or strange suffixes
    garbage = ['t_ds_', 'w_', 'q_', 'f_', 'auto', 'lsci', 'db', 'pictures', 'cms', 'upload', 'image', 'hscicdn', 'img1']
    if any(g in v for g in garbage): return True
    if '/' in v and not re.match(r'^\d+/\d+$', v): return True
    if v == '-' or v == '--': return False # Not garbage, just empty stat
    return False

def semnat_type(val):
    val = val.strip().replace('"', '')
    if not val: return None
    v = val.lower()
    
    if is_garbage(val) or 'http' in v or '.jpg' in v or '.png' in v:
        return 'URL'
    
    role_keywords = ['batsman', 'bowler', 'all-rounder', 'wicket-keeper', 'fast', 'spinner', 'medium', 'orthadox', 'wrist-spin']
    if any(r in v for r in role_keywords): return 'ROLE'
    
    known_formats = ['odi', 't20', 't20i', 'test', 'tests']
    if v in known_formats: return 'FMT'
    
    if '/' in val and re.match(r'^\d+/\d+$', val): return 'BBI'
    if '*' in val: return 'HS'
    
    try:
        f = float(val)
        return 'NUM'
    except:
        return 'STR'

def fix_row(parts, header):
    out = [""] * len(header)
    player_parts = []
    team = ""
    fmt = ""
    url_parts = []
    role = ""
    nums = []
    
    known_teams = ['Pakistan', 'India', 'Australia', 'South Africa', 'England', 'Windies', 'Srilanka', 'New Zeland', 'Newzeland', 'NewZeland', 'Afghanistan', 'Bangladesh', 'Ireland', 'Zimbabwe', 'Nepal', 'Canada', 'Oman', 'Namabia', 'Italy', 'Netherlands']

    for p in parts:
        p = p.strip().replace('"', '')
        if not p: continue
        t = semnat_type(p)
        
        if t == 'URL':
            url_parts.append(p)
        elif t == 'ROLE':
            role = p
        elif t == 'FMT':
            fmt = p
        elif t in ['NUM', 'BBI', 'HS']:
            # Handle '-' as a valid numeric placeholder
            nums.append(p)
        elif t == 'STR':
            # Check if it's a known team
            is_team_found = False
            for kt in known_teams:
                if kt.lower() == p.lower():
                    team = kt
                    is_team_found = True
                    break
            if not is_team_found:
                # If it's a known team but shifted/fragmented, catch it
                for kt in known_teams:
                    if kt.lower() in p.lower() and len(p) < 20: # avoid catching long garbage
                        team = kt
                        is_team_found = True
                        break
            
            if not is_team_found:
                # If not team and not garbage, it's player name
                if not is_garbage(p):
                    player_parts.append(p)

    out[0] = " ".join(player_parts)
    out[1] = team
    out[header.index('Format')] = fmt
    out[header.index('image_url')] = ",".join(url_parts)
    out[header.index('role')] = role

    # Distribute Numbers
    num_indices = [j for j in range(len(header)) if not out[j] and j not in [0, 1, header.index('Format'), header.index('image_url'), header.index('role')]]
    
    n_ptr = 0
    for n in nums:
        if n_ptr < len(num_indices):
            out[num_indices[n_ptr]] = n
            n_ptr += 1
            
    # Pad
    for j in range(len(header)):
        if not out[j]: out[j] = "0"
            
    return out

def main():
    files = {
        "odi_batsman.csv": "batsman",
        "odi_bowler.csv": "bowler",
        "odi_all_rounders.csv": "all_rounder"
    }
    schemas = {
        'batsman': ['player', 'Team', 'Format', 'matches', 'Innings', 'NO', 'runs', 'wickets', 'average', 'bowling_average', 'strike_rate', 'HS', '100s', '50s', 'batting_position', 'image_url', 'role', 'economy'],
        'bowler': ['player', 'Team', 'Format', 'matches', 'Innings', 'runs', 'wickets', 'average', 'bowling_average', 'bowling_strike_rate', 'economy', 'batting_position', '5 wkts', 'image_url', 'role', 'strike_rate', '100s', '50s'],
        'all_rounder': ['player', 'Team', 'Format', 'matches', 'Innings', 'NO', 'runs', 'wickets', 'average', 'bowling_average', 'strike_rate', 'HS', '100s', '50s', 'batting_position', 'image_url', 'role']
    }
    
    for fname, stype in files.items():
        if not os.path.exists(fname): continue
        print(f"Final rescue for {fname}...")
        header = schemas[stype]
        new_rows = [header]
        with open(fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[1:]:
            # Split by comma and strip quotes carefully
            raw_line = line.replace('"', '').replace('\t', ',')
            parts = [p.strip() for p in raw_line.split(',')]
            parts = [p for p in parts if p]
            new_rows.append(fix_row(parts, header))
            
        with open(fname, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)
            
    print("Rescue complete.")

if __name__ == "__main__":
    main()
