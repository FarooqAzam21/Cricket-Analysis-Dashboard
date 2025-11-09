import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\Farooq\Desktop\New folder (4)\Cricket_Analysis\odi_all_rounders.csv")
batsmen_count = df['player'].nunique()
print(batsmen_count)