import pandas as pd
import pickle

df_name = 'df.csv'
df = pd.read_csv(df_name)

with open('df_978_l5.p','wb') as f:
	pickle.dump(df,f)