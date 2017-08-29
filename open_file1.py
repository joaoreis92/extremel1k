from cmapPy.pandasGEXpress import parse
import pandas as pd
import numpy as np
import sys


dataset_file = 'dataset_l5.gctx'
if len(sys.args) > 1:
	dataseet_file = sys.args[1]


sig_7 = pd.read_csv('GSE70138_Broad_LINCS_sig_info.csv')
sig_9 = pd.read_csv('GSE92742_Broad_LINCS_sig_info.csv')
ids_sig = sig_9.loc[sig_9['pert_iname'].isin(sig_7['pert_iname'])]
ids_sig = ids_sig['sig_id'].tolist()

gen_info = pd.read_csv('GSE92742_Broad_LINCS_gene_info.csv')
lm_gen_info = gen_info.loc[gen_info['pr_is_lm'] == 1]
ids_lm = lm_gen_info['pr_gene_id'].tolist()
l5_lm = parse(dataseet_file,rid=ids_lm,cid=ids_sig)
df = l5_lm.data_df
df.to_csv('df.csv')


