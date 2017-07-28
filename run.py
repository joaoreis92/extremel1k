import numpy as np
import main_experiments
import sys
choice = sys.argv[1]

if choice == 'lomtree':
	dict_choice = [{'train_cells':['BT20','A549','A375'],'model':['vw'],'passes':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],'learning_rate':[0.01,0.1,0.2,0.3,0.5,1]}]
if choice == 'liblinear':
	dict_choice = [{'train_cells':['BT20','A549','A375'],'model':['liblinear'],'C':[0.01,0.1,1,10]}]
if choice == 'pdsparse':
	dict_choice = [{'train_cells':['BT20','A549','A375'],'model':['pdsparse']}]
if choice == 'test': 
	dict_choice = [{'model':'liblinear','test_cells':['BT20']}]

results = main_experiments.experiments_cv(dict_choice)