import csv
import os

def backfill():
    files = ["odi_batsman.csv", "odi_bowler.csv", "odi_all_rounders.csv"]
    
    # 1. Build Mapping
    url_to_name = {}
    url_to_team = {}
    
    for fname in files:
        if not os.path.exists(fname): continue
        with open(fname, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            name_idx = 0
            team_idx = 1
            url_idx = header.index('image_url')
            
            for row in reader:
                name = row[name_idx]
                team = row[team_idx]
                url = row[url_idx]
                
                if name and name != "Unknown" and url:
                    # Use unique URL part for mapping if it's long
                    # Actually just use full URL
                    url_to_name[url] = name
                    if team:
                        url_to_team[url] = team

    # 2. Apply Backfill
    for fname in files:
        if not os.path.exists(fname): continue
        print(f"Backfilling names for {fname}...")
        
        with open(fname, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
            url_idx = header.index('image_url')
            
        new_rows = [header]
        for row in rows:
            url = row[url_idx]
            if (row[0] == "Unknown" or not row[0]) and url in url_to_name:
                row[0] = url_to_name[url]
            if (row[1] == "International" or not row[1]) and url in url_to_team:
                row[1] = url_to_team[url]
            new_rows.append(row)
            
        with open(fname, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)
            
    print("Backfill complete.")

if __name__ == "__main__":
    backfill()
