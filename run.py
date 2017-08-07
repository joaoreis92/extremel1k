import numpy as np
import main_experiments
import sys
choice = sys.argv[1]
nr_samples=0
nr_samples = sys.argv[2]

if choice == 'lomtree':
	dict_choice = [{'train_cell':['BT20','A549','A375'],'model':['vw'],'passes':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],'learning_rate':[0.01,0.1,0.2,0.3,0.5,1]}]
if choice == 'liblinear':
	dict_choice = [{'train_cell':['BT20','A549','A375'],'model':['liblinear'],'C':[0.01,0.1,1,10]}]
if choice == 'pdsparse':
	dict_choice = [{'train_cell':['BT20','A549','A375'],'model':['pdsparse']}]

if choice == 'lomtree_random':
	dict_choice = [{'train_cell':['BT20','A549','A375'],'model':['vw'],'passes':[1,2,3,4],'learning_rate':np.linspace(0.001,3).tolist(),'loss_function':['squared','hinge','logistic']}]
if choice == 'liblinear_random':
	dict_choice = [{'train_cell':['BT20','A549','A375'],'model':['liblinear'],'C':np.logspace(-3,3).tolist()}]
if choice == 'pdsparse_random':
	dict_choice = [{'train_cell':['BT20','A549','A375'],'model':['pdsparse'],'iter_pd':[int(x) for x in np.linspace(1,100,20).tolist()]},{'train_cell':['BT20','A549','A375'],'model':['pdsparse'],'C':[int(x) for x in np.logspace(-2,2,20).tolist()]},{'train_cell':['BT20','A549','A375'],'model':['pdsparse'],'lambda':[int(x) for x in np.linspace(1,100,20).tolist()]}]


if choice == 'lomtree_random_all':
	dict_choice = [{'train_cell':['All'],'all_cells':[True],'model':['vw'],'passes':[1,2,3,4,5,6,7,8,9,10],'learning_rate':np.linspace(0.001,3).tolist(),'loss_function':['squared','hinge','logistic']}]
if choice == 'liblinear_random_all':
	dict_choice = [{'train_cell':['All'],'all_cells':[True],'model':['liblinear'],'C':np.logspace(-3,3).tolist()}]
if choice == 'pdsparse_random_all':
	dict_choice = [{'train_cell':['All'],'all_cells':[True],'model':['pdsparse'],'iter_pd':[int(x) for x in np.linspace(1,100,20).tolist()]},{'train_cell':['All'],'all_cells':[True],'model':['pdsparse'],'C':[int(x) for x in np.logspace(-2,2,20).tolist()]},{'train_cell':['All'],'all_cells':[True],'model':['pdsparse'],'lambda':[int(x) for x in np.linspace(1,100,20).tolist()]}]

if choice == 'lomtree_random_aloi':
	dict_choice = [{'train_cell':['aloi'],'model':['vw'],'passes':[1,2,3,4],'learning_rate':np.linspace(0.001,3).tolist(),'loss_function':['squared','hinge','logistic']}]
if choice == 'liblinear_random_aloi':
	dict_choice = [{'train_cell':['aloi'],'model':['liblinear'],'C':np.logspace(-3,3).tolist()}]
if choice == 'pdsparse_random_aloi':
	dict_choice = [{'train_cell':['aloi'],'model':['pdsparse'],'iter_pd':[int(x) for x in np.linspace(1,100,20).tolist()]},{'train_cell':['BT20','A549','A375'],'model':['pdsparse'],'C':[int(x) for x in np.logspace(-2,2,20).tolist()]},{'train_cell':['BT20','A549','A375'],'model':['pdsparse'],'lambda':[int(x) for x in np.linspace(1,100,20).tolist()]}]



if choice == 'test': 
	dict_choice = [{'model':['liblinear'],'train_cell':['BT20']}]

results = main_experiments.experiments_cv(dict_choice,int(nr_samples))