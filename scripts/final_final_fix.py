import csv
import os
import re

def sem_extract(line, header):
    # Normalize
    line = line.replace('"', '').replace('\t', ',')
    parts = [p.strip() for p in line.split(',')]
    full_str = " ".join(parts)
    
    known_teams = ['Pakistan', 'India', 'Australia', 'South Africa', 'England', 'Windies', 'Srilanka', 'New Zeland', 'Newzeland', 'NewZeland', 'Afghanistan', 'Bangladesh', 'Ireland', 'Zimbabwe', 'Nepal', 'Canada', 'Oman', 'Namabia', 'Italy', 'Netherlands']
    known_formats = ['odi', 't20', 'test', 't20i', 'tests']
    role_keywords = ['batsman', 'bowler', 'all-rounder', 'wicket-keeper', 'fast', 'spinner', 'medium', 'orthadox', 'wrist-spin']
    garbage_tokens = ['t_ds_', 'w_', 'q_', 'f_', 'auto', 'lsci', 'db', 'pictures', 'cms', 'upload', 'image', 'hscicdn', 'img1', 'yimg', 'static', 'photo']

    # 1. URL
    url = ""
    url_m = re.search(r'https?://[^\s,]+', full_str)
    if url_m:
        url = url_m.group(0)
        # Greedily join subsequent path parts
        for p in parts:
             if ('PICTURES' in p or 'CMS' in p or '.jpg' in p or 'q_50' in p or 'f_auto' in p) and p not in url:
                 url += "," + p

    # 2. Team
    team = ""
    for kt in known_teams:
        if kt.lower() in full_str.lower():
            team = kt
            break
            
    # 3. Format
    fmt = ""
    for f in known_formats:
        if re.search(r'\b' + f + r'\b', full_str, re.I):
            fmt = f.capitalize() if f.lower() != 't20' else 'T20'
            break
            
    # 4. Role
    role = ""
    for r in role_keywords:
        if r in full_str.lower():
            role = r
            for p in parts:
                if r in p.lower():
                    role = p
                    break
            break

    # 5. Name - Support name particles
    name_parts_regex = r'\b(?:[A-Z][a-z]+|de|van|der|von|di|del|al)\b'
    candidates = re.findall(name_parts_regex, full_str)
    player_parts = []
    for c in candidates:
        if c in known_teams: continue
        if c.lower() in known_formats: continue
        if c.lower() in role_keywords: continue
        if any(g in c.lower() for g in garbage_tokens): continue
        if len(c) < 2: continue # Allow "de"
        player_parts.append(c)
    
    seen = set()
    player_parts = [x for x in player_parts if not (x in seen or seen.add(x))]
    player = " ".join(player_parts)

    # 6. Stats
    stats = []
    for p in parts:
        if re.match(r'^-?\d+(\.\d+)?\*?$', p) or '/' in p or p == '-':
            stats.append(p)
    
    # 7. Map to Header
    out = ["0"] * len(header)
    out[0] = player if player else "Unknown"
    out[1] = team if team else "International"
    
    fmt_idx = header.index('Format')
    url_idx = header.index('image_url')
    role_idx = header.index('role')
    
    out[fmt_idx] = fmt if fmt else "Odi"
    out[url_idx] = url
    out[role_idx] = role if role else "Player"
    
    # Distribute Numbers
    num_idxs = [j for j in range(len(header)) if out[j] == "0" and j not in [0, 1, fmt_idx, url_idx, role_idx]]
    s_ptr = 0
    for ni in num_idxs:
        if s_ptr < len(stats):
            out[ni] = stats[s_ptr]
            s_ptr += 1
            
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
        print(f"Final reconstruction of {fname}...")
        header = schemas[stype]
        new_rows = [header]
        with open(fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[1:]:
            if not line.strip(): continue
            new_rows.append(sem_extract(line.strip(), header))
        with open(fname, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)
    print("Done.")

if __name__ == "__main__":
    main()
