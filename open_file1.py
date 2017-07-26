from cmapPy.pandasGEXpress import parse
import pandas as pd
import numpy as np
import sys


dataset_file = 'dataset_l5.gctx'
if len(sys.args) > 1:
	dataseet_file = sys.args[1]

gen_info = pd.read_csv('GSE92742_Broad_LINCS_gene_info.csv')
lm_gen_info = gen_info.loc[gen_info['pr_is_lm'] == 1]
ids_lm = lm_gen_info['pr_gene_id'].tolist()
l5_lm = parse(dataseet_file,rid=ids_lm)
df = l5_lm.data_df
df.to_csv('df.csv')


