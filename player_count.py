import pandas as pd
import numpy as np
df1 = pd.read_csv(r"C:\Users\Farooq\Desktop\New folder (4)\Cricket_Analysis\odi_batsman.csv")
df2 = pd.read_csv(r'C:\Users\Farooq\Desktop\New folder (4)\Cricket_Analysis\odi_bowler.csv')
df3 = pd.read_csv(r'C:\Users\Farooq\Desktop\New folder (4)\Cricket_Analysis\odi_all_rounders.csv')
df = pd.concat([df1 , df2 , df3])
batsmen_count = df3['player'].nunique()
print(batsmen_count)