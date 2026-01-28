import pandas as pd
data = pd.read_csv(r'odi_bowler.csv')
data2 = pd.read_csv(r'odi_all_rounders.csv')
data3= pd.read_csv(r'odi_batsman.csv')
all_data = pd.concat([data, data2, data3], ignore_index=True)
team = all_data['Format'].unique()
print(team)
team = all_data[all_data['Format'] == 5]
print(team)