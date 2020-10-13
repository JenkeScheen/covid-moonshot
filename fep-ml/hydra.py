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



if __name__ == "__main__":
	perturbation_paths = [
			["mobley_9185328~mobley_9185328"],
			["mobley_9209581~mobley_9565165"],
			["mobley_9209581~mobley_9821936"]]
	#computeLigMolProps("./input/ligands/")
	#computePertMolProps(perturbation_paths=perturbation_paths)

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

	normaliseDataset(
					path_to_raw_trainingset="trainingsets/MOLPROPS_trainingset.h5",
					path_to_save_loc="trainingsets_prepared/",
					feature_type="MOLPROPS", 
					chunksize=6000)


















