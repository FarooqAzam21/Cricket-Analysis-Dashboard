import csv
import os
import re

def clean_row(parts, header):
    out = [""] * len(header)
    
    # Target indices
    fmt_idx = header.index('Format')
    url_idx = header.index('image_url')
    role_idx = header.index('role')
    
    # 1. Find Format Anchor
    known_formats = ['odi', 't20', 'test', 't20i', 'tests']
    anchor_idx = -1
    for i, p in enumerate(parts):
        if p.lower().strip() in known_formats:
            anchor_idx = i
            break
    
    if anchor_idx == -1:
        # Fallback if format missing
        return parts + [""] * (len(header) - len(parts))

    # Prefix: Player, Team
    prefix = [p.strip() for p in parts[:anchor_idx]]
    known_teams = ['Pakistan', 'India', 'Australia', 'South Africa', 'England', 'Windies', 'Srilanka', 'New Zeland', 'Newzeland', 'NewZeland', 'Afghanistan', 'Bangladesh', 'Ireland', 'Zimbabwe', 'Nepal', 'Canada', 'Oman', 'Namabia', 'Italy', 'Netherlands']
    
    player = ""
    team = ""
    for p in prefix:
        is_team = False
        for kt in known_teams:
            if kt.lower() in p.lower():
                team = kt
                is_team = True
                break
        if not is_team:
            player = (player + " " + p).strip()
    
    # Suffix: Stats, URL, Role
    suffix = [p.strip() for p in parts[anchor_idx+1:]]
    
    url = ""
    role = ""
    role_keywords = ['batsman', 'bowler', 'all-rounder', 'wicket-keeper', 'fast', 'spinner', 'medium', 'orthadox', 'wrist-spin']
    stats = []
    
    in_url = False
    for p in suffix:
        if 'http' in p.lower():
            url = p
            in_url = True
            continue
        
        if in_url:
            # Check if p is a role or numeric
            is_role = any(r in p.lower() for r in role_keywords)
            is_num = re.match(r'^-?\d+(\.\d+)?\*?$', p) or '/' in p
            if is_role:
                role = p
                in_url = False
            elif is_num:
                stats.append(p)
                in_url = False
            else:
                url += "," + p # greedy merge url
            continue
        
        if any(r in p.lower() for r in role_keywords):
            role = p
            continue
            
        stats.append(p)

    # Reassemble
    out[0] = player
    out[1] = team
    out[fmt_idx] = parts[anchor_idx]
    out[url_idx] = url
    out[role_idx] = role
    
    stat_ptr = 0
    num_indices = [j for j in range(len(header)) if not out[j] and j not in [0, 1, fmt_idx, url_idx, role_idx]]
    for ni in num_indices:
        if stat_ptr < len(stats):
            out[ni] = stats[stat_ptr]
            stat_ptr += 1
        else:
            out[ni] = "0"
            
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
        print(f"Applying real fix to {fname}...")
        header = schemas[stype]
        new_rows = [header]
        with open(fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[1:]:
            line = line.strip().replace('"', '')
            parts = [p.strip() for p in line.split(',')]
            parts = [p for p in parts if p]
            new_rows.append(clean_row(parts, header))
        with open(fname, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)
    print("All CSVs cleaned.")

if __name__ == "__main__":
    main()
