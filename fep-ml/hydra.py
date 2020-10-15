#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for Hydra - learning ddGoffset values for free energy perturbations. 
"""

# TF-related imports & some settings to reduce TF verbosity:
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"	# current workstation contains 4 GPUs; exclude 1st
import tensorflow as tf 
from tensorflow import keras

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

# hyperparameter optimisation:
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from tensorflow.python.keras import backend as K
from skopt.utils import use_named_args

# featurisation:
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles


# general imports:
import pandas as pd 
import numpy as np 
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import preprocessing, decomposition
from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm import tqdm
import glob
import pickle

###################################################
###################################################
###################### UTILS ######################
###################################################
###################################################

def retrieveMoleculePDB(ligand_path):
	"""
	Returns RDKit molecule objects for requested path PDB file.

	-- args
	ligand_path (str): path leading to molecule pdb file

	-- returns
	RDKit molecule object
	"""
	mol = rdmolfiles.MolFromPDBFile(
									ligand_path, 
									sanitize=True
									)
	return mol

def readHDF5Iterable(path_to_trainingset, chunksize):
	"""
	Read in a training set using pandas' HDF5 utility

	--args
	path_to_trainingset (str): path to training set (HDF5) to read from 
	chunksize (int): number of items to read in per increment (recommended 5000 for large datasets)

	--returns
	training_set (iterable)

	"""
	training_set = pd.DataFrame()

	# use chunksize to save memory during reading:
	training_set_iterator = pd.read_hdf(path_to_trainingset, chunksize=chunksize)


	return training_set_iterator

###################################################
###################################################
################## FEATURISERS ####################
###################################################
###################################################

###################################################
### Molecular properties:						###
												###
												###

def computeLigMolProps(
					transfrm_path="transformations/", 
					working_dir="features/MOLPROPS/",
					target_columns=None, 
					verbose=False):
	"""
	Compute molecular properties for the molecules in given transfrm_path and write to file.

	--args
	transfrm_path (str): path to directory containing ligand files
	working_dir (str): path to directory to pickle into
	verbose (bool): whether or not to print featurisation info to stdout

	--returns
	molprops_set (pandas dataframe): set of molecules with molecular properties

	"""
	mol_paths = glob.glob(transfrm_path+"*")

	# generate RDKit mol objects from paths:
	mols_rdkit = [ retrieveMoleculePDB(mol) for mol in mol_paths ]

	# generate molecule name from paths for indexing:
	mols_names = [ mol.replace(transfrm_path, "").split(".")[0] for mol in mol_paths ]

	
	# generate all descriptors available in mordred:
	calc = Calculator(descriptors, ignore_3D=False)
	print("Computing molecular properties:")
	molprops_set = calc.pandas(mols_rdkit)

	# remove columns with bools or strings (not fit for subtraction protocol):
	if target_columns is not None:
		# if variable is input the function is handling a testset and must 
		# keep the same columns as train dataset:
		molprops_set = molprops_set[target_columns]
	else:
		# if making a training dataset, decide which columns to retain:
		molprops_set = molprops_set.select_dtypes(include=["float64", "int64"])
	
	molprops_set.index = mols_names

	# pickle dataframe to specified directory:
	molprops_set.to_pickle(working_dir+"molprops.pickle")

	if verbose:
		print(molprops_set)

	return molprops_set

def computePertMolProps(
						perturbation_paths, 
						molprops_set=None,
						free_path="SOLVATED/", 
						working_dir="features/MOLPROPS/"):
	"""
	Read featurised FEP molecules and generate matches based on user input perturbations.
	Writes each perturbation features by appending it to the features.csv file.

	--args
	perturbation_paths (list): nested list of shape [[A~B],[C~D]] with strings describing 
	the perturbations. These combinations will be used to make pairwise extractions 
	from molprops_set.

	molprops_set (pandas dataframe; optional): dataframe object that contains the
	featurised FEP dataset. If None, will attempt to pickle from working_dir

	free_path (str): path to directory containing perturbation directories

	working_dir (str): path to directory to pickle dataset from

	--returns
	None

	"""

	# test if input is there:
	if molprops_set is None:
		try:
			molprops_set = pd.read_pickle(working_dir+"molprops.pickle")
		except FileNotFoundError:
			print("Unable to load pickle file with per-ligand molprop data in absence of molprops_set function input.")
			
	# clean slate featurised perturbations dataset; write column names:
	open(working_dir+"featurised_molprops.h5", "w").close()
	store = pd.HDFStore(working_dir+"featurised_molprops.h5") 

	# write list of column names to file for future testset feature generation:
	pd.DataFrame(molprops_set.columns).transpose().to_csv(working_dir+"featurised_molprops.csv", header=False)

	# iterate over perturbations:
	for perturbation in tqdm(perturbation_paths):

		ligandA = perturbation[0].split("~")[0]
		ligandB = perturbation[0].split("~")[1]

		# extract molprops from per-ligand:
		ligandA_molprops = molprops_set.loc[ligandA]
		ligandB_molprops = molprops_set.loc[ligandB]

		# subtract and transform to dataframe:
		perturbation_molprops = ligandB_molprops.subtract(
			ligandA_molprops).to_frame(name=perturbation[0]).transpose()

		# append to the molprops HDF5 file:
		store.append(
					working_dir+"featurised_molprops.h5", 
					perturbation_molprops,
					format="table",
					index=False,
					min_itemsize=500
					)

	store.close()

###################################################
###################################################
#################### LABELER ######################
###################################################
###################################################

def retrieveMBAROutput(path_to_datafile, verbose=False):
	"""
	Get input data with computed ddG and experimental ddG, compute offsets, write to ./labels/

	"""


	return "poep"


###################################################
###################################################
#################### BUILDERS #####################
###################################################
###################################################

def buildTrainingSet(
				path_to_labels_file, 
				path_to_features_file, 
				path_to_trainingset_file
				):
	"""
	Build a training set by joining the features with labels

	--args
	path_to_labels_file (str): path to file containing labels
	path_to_features_file (str): path to file containing features
	path_to_trainingset_file (str): path to write resulting training set into

	--returns
	None

	"""

	# load featurised dataset into memory per line using the pandas generator:
	featurised_dataset = pd.read_hdf(path_to_features_file, chunksize=1)

	# clean slate the training set file:
	open(path_to_trainingset_file, "w").close()

	# load in the labels as a DF for fast index pairing:
	labels_df = pd.read_csv(path_to_labels_file, index_col=0, names=["ddGoffset", "unc"])
	if "1DCNN" in path_to_features_file:
		num_perturbations = len(labels_df)*3
	else:
		num_perturbations = len(labels_df)
	
	# per featurised datapoint, iterate:
	store = pd.HDFStore(path_to_trainingset_file)
	
	for features_frame in tqdm(featurised_dataset, total=num_perturbations):

		perturbation_name = features_frame.index.values[0]
		fep_info = labels_df.loc[perturbation_name]
		ddGoffset = fep_info["ddGoffset"]
		unc = fep_info["unc"]

		# attach the labels:
		features_frame["ddGoffset"] = round(float(ddGoffset), 8)
		features_frame["unc"] = round(float(unc), 8)

		# TMP FOR DEV:
		features_frame_inf = pd.DataFrame(np.repeat(features_frame.values,300,axis=0))
		features_frame_inf.columns = features_frame.columns
		features_frame = features_frame_inf
		#####################
		# write this perturbation's data to file:
		store.append(
				path_to_trainingset_file, 
				features_frame,
				format="table",
				index=False,
				)
	store.close()


def dropLabels(collection, feature_type):
	"""
	Drop labels and return features + labels separately

	--args
	collection (pandas DF): dataset containing features + labels
	feature_type (str): determines which labels to drop

	--returns
	labels (pandas series): series containing all label names
	collection_features (pandas DF): dataset containing all feature data
	"""

	if feature_type == "1DCNN":
		labels = collection[["frame", "error", "freenrg", "overlap_score"]]
		collection_features = collection.drop(["frame", "error", "freenrg", "overlap_score"], axis=1)
	else:
		labels = collection[["ddGoffset", "unc"]]
		collection_features = collection.drop(["ddGoffset", "unc"], axis=1)

	return labels, collection_features



def normaliseDataset(path_to_raw_trainingset, path_to_save_loc, feature_type, chunksize):
	"""
	PCA + Normalise a provided dataset using pandas and SKLearn normalisation for rapid processing
	that scales linearly with input size.

	Pickles the normalisation object for future external test sets.

	--args
	collection_iterable (iterable): pandas DF object containing an index and column names
	feature_type (str): describes which feature type is being processed

	--returns
	normalised_collection (pandas dataframe): normalised training set containing 
	an index and column names
	labels (list of series): vectors with target labels 
	collection.columns (list of strings): list of column names, including labels 
	collection_features.index (list of strings): list of index names, i.e. perturbations

	"""
	scaler = preprocessing.StandardScaler()

	if feature_type == "1DCNN": # all parameters here were found manually:
		n_components = 200
		print("This function takes ~10s to complete on 15K datapoints.\n")
	elif feature_type == "MOLPROPS":
		n_components = 750
		print("This function takes ~10m to complete on 15K datapoints.\n")
	elif feature_type == "PFP":
		n_components = 200
		print("This function takes ~10s to complete on 15K datapoints.\n")

	pca = decomposition.IncrementalPCA(n_components=n_components)
	

	###########################################################################################
	# we need to perform incremental standardization because of the large dataset:
	print("Making first pass (partial fitting)..")
	collection_iterable = readHDF5Iterable(
										path_to_raw_trainingset, 
										chunksize=chunksize)


	for collection in collection_iterable:

		# omit labels from normalisation:
		labels, collection_features = dropLabels(collection, feature_type)

		# fit the normalisation:
		scaler.partial_fit(collection_features)
		

	###########################################################################################
	# Now with fully updated means + variances, make a second pass through the iterable and transform:
	print("Making second pass (partial transform + partial PCA fit)..")
	collection_iterable = readHDF5Iterable(
									path_to_raw_trainingset, 
									chunksize=chunksize)

	for collection in collection_iterable:	
		# omit labels from normalisation:
		labels, collection_features = dropLabels(collection, feature_type)

		# transform:
		normalised_collection = pd.DataFrame(scaler.transform(collection_features))

		# now fit an incremental PCA to this chunk:
		pca.partial_fit(normalised_collection)
		
	
	# # uncomment to figure out ~ how many dims to retain for 95% VE.
	# # can't use n_components=0.95 in our case because we process in chunks :(

	# ve_ratios = pca.explained_variance_ratio_
	# ve_counter = 0
	# ve_cumulative = 0
	# for ve in ve_ratios:
	# 	if not ve_cumulative >= 0.95:			
	# 		ve_cumulative += ve
	# 		ve_counter += 1
	# print("Keep", ve_counter, "to retain", ve_cumulative*100, "of variance explained.")

	###########################################################################################
	# now with the completed PCA object; go over iterable one last time;
	# apply normalisation and transform by PCA and save to individual files:
	print("Making third pass (normalise and PCA transform)..")
	collection_iterable = readHDF5Iterable(
									path_to_raw_trainingset, 
									chunksize=chunksize)
	
	if os.path.exists(path_to_save_loc+feature_type+"/data.h5"):
		os.remove(path_to_save_loc+feature_type+"/data.h5")
	store = pd.HDFStore(path_to_save_loc+feature_type+"/data.h5")


	for collection in collection_iterable:
		
		# this is our final transform; save perturbation names:
		perturbation_indeces = collection.index

		# omit labels from normalisation:
		labels, collection_features = dropLabels(collection, feature_type)

		# normalise transform:
		normalised_collection = pd.DataFrame(scaler.transform(collection_features))

		# PCA transform to finish preprocessing:
		processed_collection = pca.transform(normalised_collection)

		# prettify the np matrix back into DF and append to HDF:
		num_PCA_dims = len(processed_collection[0])
		PCA_column_headers = [ "PC"+str(dim) for dim in range(num_PCA_dims)]

		pca_data_df = pd.DataFrame(
									processed_collection, 
									index=perturbation_indeces, 
									columns=PCA_column_headers
									)

		complete_preprocessed_df = pd.concat([pca_data_df, labels], 
												axis=1, sort=False)

		store.append(
					path_to_save_loc+feature_type+"/data.h5", 
					complete_preprocessed_df,
					format="table",
					index=False,
					)
	store.close()
	# finally, save both the standardscaler and PCA object for transforming test datasets:
	pickle.dump(scaler, open(path_to_save_loc+"PICKLES/"+feature_type+"_scaler.pkl","wb"))
	pickle.dump(pca, open(path_to_save_loc+"PICKLES/"+feature_type+"_pca.pkl","wb"))


###################################################
###################################################
#################### LEARNERS #####################
###################################################
###################################################

def importDataSet(training_label, path_to_trainingset, path_to_testset=None):
	"""
	Import a pre-processed training set (i.e. by hydra_preprocess) and 
	return X and y.

	--args
	training_label (str): labels to train on; will be removed from test set
	path_to_trainingset (str): path from which the processed training set (HDF5) can
	be read. If no testset is specified, trainingset will be split into 80/20.
	path_to_testset (str, optional): path from which processed testset (HDF5) can be read. 

	--returns
	X_train (pandas dataframe): dataframe containing features for training; 
	contains perturbation names as index. 
	y_train (pandas dataframe): dataframe containing labels for training; 
	contains perturbation names as index. 

	X_test (pandas dataframe): dataframe containing features for testing; 
	contains perturbation names as index. 
	y_test (pandas dataframe): dataframe containing labels for testing; 
	contains perturbation names as index. 
	"""
	trainingset = pd.read_hdf(path_to_trainingset)

	# for now, drop uncertainties from trainingset. Might use later for some probabilistic learning.
	trainingset = trainingset.drop("unc", axis=1)

	# now extract labels and return dataframes separately:
	y_train = trainingset[[training_label]]
	X_train = trainingset.drop([training_label], axis=1)


	if path_to_testset:
		# if external testset is specified, subtract X/y_test from it:
		testset = pd.read_csv(path_to_testset, index_col=0)

		y_test = testset[[training_label]]
		X_test = testset.drop([training_label], axis=1)

		return X_train, y_train, X_test, y_test


	elif not path_to_testset:
		# if not specified, make a 20% testset from trainingset:
		X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
														test_size=0.2, 
														random_state=42
														)
		return X_train, y_train, X_test, y_test


def denseNN(X_train, y_train, X_test, y_test, feature_type):
	"""
	Given traininset and testsets and feature-type specification,
	train a densely connected neural network.

	--args
	X_train (numpy 2D array): trainingset with features only of dim=(n,i)
	y_train (numpy 2D array): trainingset with labels only of dim=(n,1)
	X_test (numpy 2D array): testset with features only of dim=(n,i)
	y_test (numpy 2D array): testset with labels only of dim=(n,1)

	--returns
	fitness (fn): function that contains training protocol
	dimensions (list): list of variables referencing SKOPT hyperparameter ranges
	n_calls (int): number of SKOPT hyperparameter optimisation repeats to run
	default_parameters (list): list of floats/ints/str of initial values that fall in 
							   the "dimensions" list hyperparameter ranges

	"""

	# clean slate stats output for convergence data:
	stat_output_path = "output/"+feature_type+"_skopt_conv_data.csv"
	if os.path.exists(stat_output_path):
		open(stat_output_path).close()
	stats_per_skopt_call = []

	def create_model(
		num_dense_layers_base, 
		num_dense_nodes_base,
		num_dense_layers_end, 
		num_dense_nodes_end, 
		learning_rate,
		adam_b1,
		adam_b2,
		adam_eps,
		num_batch_size):


		model = keras.Sequential()

	# Add input layer of length of the dataset columns:
		model.add(keras.layers.Dense(len(X_train.columns), input_shape=[len(X_train.keys())]))

	# Generate n number of hidden layers (base, i.e. first layers):
		for i in range(num_dense_layers_base):
			model.add(keras.layers.Dense(num_dense_nodes_base,
			activation=keras.activations.relu
			))

	# Generate n number of hidden layers (end, i.e. last layers):
		for i in range(num_dense_layers_end):
			model.add(keras.layers.Dense(num_dense_nodes_end,
			activation=keras.activations.relu
			))

	# Add output layer:

		model.add(keras.layers.Dense(1, activation=keras.activations.linear))

		optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=adam_b1, beta_2=adam_b2, epsilon=adam_eps)

		model.compile(
			loss="logcosh",
			#loss="mae",
			optimizer=optimizer,
			metrics=["mean_absolute_error"]
			)
		return model


	# Set hyperparameter ranges, append to list:
	dim_num_dense_layers_base = Integer(low=1, high=2, name='num_dense_layers_base')
	dim_num_dense_nodes_base = Categorical(categories=list(np.linspace(5,261, 10, dtype=int)), name='num_dense_nodes_base')
	dim_num_dense_layers_end = Integer(low=1, high=2, name='num_dense_layers_end')
	dim_num_dense_nodes_end = Categorical(categories=list(np.linspace(5,261, 10, dtype=int)), name='num_dense_nodes_end')


	learning_rate = Categorical(categories=list(np.linspace(0.001,0.1,10)), name="learning_rate")
	dim_adam_b1 = Categorical(categories=list(np.linspace(0.8,0.99,11)), name="adam_b1")
	dim_adam_b2 = Categorical(categories=list(np.linspace(0.8,0.99,11)), name="adam_b2")
	dim_adam_eps = Categorical(categories=list(np.linspace(0.0001, 0.5, 11)), name="adam_eps")
	dim_num_batch_size = Categorical(categories=list(np.linspace(16, 30, 8, dtype=int)), name='num_batch_size')

	dimensions = [
				dim_num_dense_layers_base,
				dim_num_dense_nodes_base,
				dim_num_dense_layers_end,
				dim_num_dense_nodes_end,
				learning_rate,
				dim_adam_b1,
				dim_adam_b2,
				dim_adam_eps,
				dim_num_batch_size]	

	@use_named_args(dimensions=dimensions)
	def fitness(
		num_dense_layers_base, 
		num_dense_nodes_base, 
		num_dense_layers_end, 
		num_dense_nodes_end,
		learning_rate,
		adam_b1,
		adam_b2,
		adam_eps,
		num_batch_size):
		early_stopping = keras.callbacks.EarlyStopping(
														monitor='val_loss', 
														mode='min', 
														patience=30,
														verbose=0)
	# Create the neural network with these hyper-parameters:
		model = create_model(
							num_dense_layers_base=num_dense_layers_base,
							num_dense_nodes_base=num_dense_nodes_base,
							num_dense_layers_end=num_dense_layers_end,
							num_dense_nodes_end=num_dense_nodes_end,
							learning_rate=learning_rate,
							adam_b1=adam_b1,
							adam_b2=adam_b2,
							adam_eps=adam_eps,
							num_batch_size=num_batch_size)



		history = model.fit(
			X_train, y_train,
		epochs=500, 
		validation_split=0.1,
		verbose=0,
		callbacks=[
					early_stopping, 
					#PrintDot(),			# uncomment for verbosity on epochs
					], 		
		batch_size=num_batch_size)

		hist = pd.DataFrame(history.history)
		hist['epoch'] = history.epoch
		val_loss = hist["val_loss"].tail(5).mean()

		#################
		# calculate some statistics on test set:
		perts_list = y_test.index.tolist()
		prediction = model.predict(X_test)

		prediction_1_list = [ item[0] for item in prediction ]
		exp_1_list = y_test.iloc[:,0].values.tolist()

		# in case of multitask:
		#prediction_2_list = [ item[1] for item in prediction ]
		#exp_2_list = y_test.iloc[:,1].values.tolist()


		# For plotting test set correlations:
		tuples_result = list(zip(perts_list, exp_1_list, prediction_1_list))
		nested_list_result = [ list(elem) for elem in tuples_result ]
		exp_vs_pred_ddGoffset_df = pd.DataFrame(nested_list_result, 
										columns=["Perturbation", "Exp1", "Pred1"])

		


		###################
		#DEFINE SKOPT COST FUNCTION HERE:


		# compute r on test set:
		test_r = abs(stats.pearsonr(exp_1_list, prediction_1_list)[0])

		# append stats to skopt convergence data:
		stats_per_skopt_call.append([val_loss, test_r])

		# sometimes, r is nan or 0; adjust:
		if not type(test_r) == np.float64 or test_r == 0:
			test_r = 0.1

		# SKOPT API is easier when minimizing a function, so return the inverse of r:
		test_r_inverse = 1/test_r


		# Append data with best performing model.
		global startpoint_error

		if test_r_inverse < startpoint_error:
			
			startpoint_error = test_r_inverse

			# # write all model files:
			model.save_weights("models/"+feature_type+"_HYDRA_weights.h5")
			with open("models/"+feature_type+"_HYDRA_architecture.json", "w") as file:
				file.write(model.to_json())

			exp_vs_pred_ddGoffset_df.to_csv("output/"+feature_type+"_top_performer.csv")

			# make a classic loss plot and save:
			plt.figure()
			plt.plot(hist['epoch'], hist['loss'], "darkorange", label="Training loss")
			plt.plot(hist['epoch'], hist['val_loss'], "royalblue", label="Validation loss")
			plt.xlabel("Epoch")
			plt.ylabel("Loss / MAE on OS")
			plt.ylim(0, 0.002)
			plt.legend()
			plt.savefig("output/"+feature_type+"_top_performer_loss_plot.png", dpi=300)

		
		del model
		tf.keras.backend.clear_session()
		K.clear_session()		
		
		return test_r_inverse

	# Bayesian Optimisation to search through hyperparameter space. 
	# Prior parameters were found by manual search and preliminary optimisation loops. 
	default_parameters = [
							2, 			# first half n connected layers
							33, 		# n neurons in first half connected layers
							1, 			# second half n connected layers
							90, 		# n neurons in second half connected layers
							0.1,		# learning rate
							0.971, 		# adam beta1
							0.895, 		# adam beta2
							1.0000e-04, # adam epsilon
							20			# batch size
							]
	return fitness, dimensions, default_parameters


def trainCorrector(fitness, dimensions, n_calls, default_parameters):
	"""
	Train a ML model by calling supporting functions. In jupyter NB, this
	function outputs the progress bar based on 
	https://github.com/scikit-optimize/scikit-optimize/issues/674

	--args
	fitness (fn): function that contains training protocol
	dimensions (list): list of variables referencing SKOPT hyperparameter ranges
	n_calls (int): number of SKOPT hyperparameter optimisation repeats to run
	default_parameters (list): list of floats/ints/str of initial values that fall in 
							   the "dimensions" list hyperparameter ranges
	model_type (str): type of ML function to run

	--returns
	search_result (object): SKOPT class that offers some functionalities.

	"""

	# make a quick progress bar class:
	class tqdm_skopt(object):
	    def __init__(self, **kwargs):
	        self._bar = tqdm(**kwargs)
	        
	    def __call__(self, res):
	        self._bar.update()

	# run the SKOPT optimiser:
	search_result = gp_minimize(func=fitness,
						dimensions=dimensions,
						acq_func='EI', #Expected Improvement.
						n_calls=n_calls,
						x0=default_parameters,
						callback=[tqdm_skopt(total=n_calls, desc="Training")])

	# should we note down optimal hyperparams anywhere?




	#### DO THIS NEXT::::
	# should append this with the previous training function so we can generate hyperparam convergence

	# with open(stat_output_path, "w") as filepath:
	# 	writer = csv.writer(filepath)
	# 	for stats_row in stats_per_skopt_call:
	# 		writer.writerow(stats_row)
	# 	writer.writerow(search_result.x)


















if __name__ == "__main__":
	perturbation_paths = [
			["mobley_9185328~mobley_9185328"],
			["mobley_9209581~mobley_9565165"],
			["mobley_9209581~mobley_9821936"]]
	#computeLigMolProps("./input/ligands/")
	#computePertMolProps(perturbation_paths=perturbation_paths)

	#### don't use this, instead manually form labels.csv:
	# with open("labels/mbar_labels.txt", "w") as file:
	# 	writer = csv.writer(file)

	# 	for path in tqdm(perturbation_paths):
	# 		overlap_matrix, OS, MBAR_freenrg, MBAR_error = retrieveMBAROutput(path+"/free/freenrg-MBAR.dat", verbose=False)
	# 		pert_name = path.replace("input/input_data", "")
	# 		writer.writerow([pert_name, MBAR_error, MBAR_freenrg, OS])

	# buildTrainingSet(
	# 	"labels/labels.csv", 
	# 	"features/MOLPROPS/featurised_molprops.h5", 
	# 	"trainingsets/MOLPROPS_trainingset.h5", 
	# 	)

	# normaliseDataset(
	# 				path_to_raw_trainingset="trainingsets/MOLPROPS_trainingset.h5",
	# 				path_to_save_loc="trainingsets_prepared/",
	# 				feature_type="MOLPROPS", 
	# 				chunksize=6000)


	X_train, y_train, X_test, y_test = importDataSet(
						"ddGoffset", 
						"trainingsets_prepared/MOLPROPS/data.h5")


	startpoint_error = np.inf
	fitness, dimensions, default_parameters = denseNN(
											X_train, 
											y_train, 
											X_test, 
											y_test, 
											"MOLPROPS")

	trainCorrector(fitness, dimensions, 11, default_parameters)

	# if so, can we make the progress bar in jupyter?













