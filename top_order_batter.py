import pandas as pd
import numpy as np
top_order = pd.read_csv(r'C:\Users\Farooq\Desktop\New folder (4)\Cricket_Analysis\odi_batters_toporder.csv')
middle_order = pd.read_csv(r'C:\Users\Farooq\Desktop\New folder (4)\Cricket_Analysis\odi_batter_middleorder.csv')
print("Top 3 batters")
numeric_cols = ['matches' , 'runs' , 'average', 'strike_rate' , 'HS' , '50s' , '100s']
for col in numeric_cols:
    top_order[col] = pd.to_numeric(top_order[col], errors='coerce')
filtered_top_order = top_order[(top_order['matches']>10) & (top_order['average'] > 40) & (top_order['strike_rate'] > 90)].head(3)   
filtered_middle_order = middle_order[(middle_order['matches']>10) & (middle_order['average'] > 35) & (middle_order['strike_rate'] > 90)].head(2)
filtered_order = pd.concat([filtered_top_order , filtered_middle_order]).reset_index(drop=True)
print("🏏 Final Top 5 Batting Lineup:\n")
print(filtered_order[["player", "Team", "matches", "average", "strike_rate"]])