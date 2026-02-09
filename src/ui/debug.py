import pandas as pd 
odi_bowler = pd.read_csv('odi_bowler.csv')
odi__all_rounders = pd.read_csv('odi_all_rounders.csv')

role = odi_bowler['role'].unique()
print(role) 
