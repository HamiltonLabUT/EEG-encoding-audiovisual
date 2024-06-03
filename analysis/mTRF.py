import sys
# sys.path.append('/Users/maansidesai/Desktop/git/ECoG_NaturalisticSounds/preproc')
# from ECoG_phn_alignment_tools import *

import scipy.io 
import h5py # For loading hf5 files
import mne # For loading BrainVision files (EEG)
from mne import io
import numpy as np
from numpy.polynomial.polynomial import polyfit
from audio_tools import spectools, fbtools, phn_tools #use custom functions for linguistic/acoustic alignment
from scipy.io import wavfile
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import os
import re
import pingouin as pg #stats package 
import pandas as pd
import traceback
import textgrid as tg

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from matplotlib_venn import venn3, venn3_circles, venn2
from scipy.stats import wilcoxon

from ridge.utils import make_delayed, counter, save_table_file
from ridge.ridge import ridge_corr, bootstrap_ridge, bootstrap_ridge_shuffle, eigridge_corr

import random
import itertools as itools
np.random.seed(0)
random.seed(0)

from scipy import stats
import scipy.optimize

import logging
from ridge.utils import make_delayed

from ridge.ridge import bootstrap_ridge
import math
import seaborn as sns


#loading ICA data 
def load_raw_EEG(subject, block, datadir, file='postICA_rejected'):
	eeg_file = '%s/%s/downsampled_128/%s_B%d_%s.fif'%(datadir, subject, subject, block, file)
	#eeg_file = '%s/%s/downsampled_128/%s_B%d_EOGs.fif'%(datadir, subject, subject, block)
	raw = mne.io.read_raw_fif(eeg_file, preload=True)	
	return raw

def loadEEGh5(subject, data_dir, condition, 
	eeg_epochs=True, resp_mean = True, binarymat=False, binaryfeatmat = False, envelope=False, pitch=False, gabor_pc10=False, 
	spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename='full_AV_matrix'):
	"""
	Load contents saved per subject from large .h5 created, which contains EEG epochs based on stimulus type 
	and corresponding speech features. 
	
	Parameters
	----------
	subject : string 
		subject ID (i.e. MT0021)
	filename : string
		either: 'mT-A-only' or 'mT_AV' or 'mT_V-only'
	data_dir : string 
		-change this to match where .h5 is along with EEG data 
	condition : string
		- AV (audiovisual)
		- V (visual only)
		- A (audio only)
	eeg_epochs : bool
		determines whether or not to load EEG epochs per stimulus type per participant
		(default : True)
	resp_mean : bool
		takes the mean across epochs for stimuli played more than once 
		(default : True)
	binarymat : bool
		determines whether or not to load 59 unique individual phoneme types 
		(deafult : False)
	binaryfeatmat : bool
		determines whether or not to load 14 unique phonological features 
		(default : True)
	envelope : bool
		determines whether or not to load the acoustic envelope of each stimulus type 
		(default : True)
	pitch : bool
		determines whether or not to load the pitch of each stimulus type 
	binned_pitches: bool
		load pitch which are binned base on frequency 
	gabor_pc10 : bool
		inclusion of visual feature from the gabor wavelets  
		(default : False)
	spectrogram : bool
		load the spectrogram of a sound 
		(default : False)
	scene_cut : bool 
		load visual data from Praat textgrid with transcribed scene cuts 
		(default : False )

	mouthing : bool 
		load mouth movement (congruent and incongruent) visual data from Praat textgrid (2D matrix)
		(default : False )
	
	filename : string
		either 'EOG_AV_matrix' or 'full_AV_matrix'

	Returns
	-------
	stim_dict : dict
		generates all features for the desired stimulus_class for a given subject as a array within the dict
		the shape of all features are resampled to the shape of phnfeat (phonological features)

	resp_dict : dict
		generates all epochs of features for the desired stimulus_class for a given subject as a array within the dict
		the shape of all epochs are resampled to the shape of phnfeat (phonological features)
	"""	 

	stim_dict = dict()
	resp_dict = dict()
	with h5py.File('%s/%s.hf5'%(data_dir, filename),'r') as fh:
		all_stim = [k for k in fh[condition].keys()]
		print(all_stim)
			
		for idx, wav_name in enumerate(all_stim): 
			print(wav_name)
			stim_dict[wav_name] = []
			resp_dict[wav_name] = []
			try:
				epochs_data = fh['%s/%s/resp/%s/epochs' %(condition, wav_name, subject)][:]
				# phnfeatmat = fh['%s/%s/stim/phn_feat_timings' %(condition, wav_name)][:]
				# ntimes = phnfeatmat.shape[1] #always resample to the size of phnfeat 
				gabors = fh['%s/%s/stim/gabor_pc10' %(condition, wav_name)][:]
				ntimes = gabors.shape[0]
				if binarymat:
					phnmat = fh['%s/%s/stim/phn_timings' %(condition, wav_name)][:] 
					stim_dict[wav_name].append(phnmat)
					ntimes = phnmat.shape[1]
					print('phnmat shape is:')
					print(phnmat.shape)
				if binaryfeatmat:
					phnfeatmat = fh['%s/%s/stim/phn_feat_timings' %(condition, wav_name)][:]
					phnfeatmat = scipy.signal.resample(phnfeatmat, ntimes, axis=1)
					stim_dict[wav_name].append(phnfeatmat)
					print('phnfeatmat shape is:')
					print(phnfeatmat.shape)
				if envelope:
					envs = fh['%s/%s/stim/envelope' %(condition, wav_name)][:] 
					envs = scipy.signal.resample(envs, ntimes) #resampling to size of phnfeat
					stim_dict[wav_name].append(envs.T)
					print('envs shape is:')
					print(envs.shape)
				if pitch:
					pitch_mat = fh['%s/%s/stim/pitches' %(condition, wav_name)][:] 
					pitch_mat = scipy.signal.resample(pitch_mat, ntimes) #resample to size of phnfeat
					pitch_mat = np.atleast_2d(pitch_mat)
					stim_dict[wav_name].append(pitch_mat)
					print('pitch_mat shape is:')
					print(pitch_mat.shape)	
			
				if gabor_pc10:
					#ntimes = phnmat.shape[1]
					gabor_pc10_mat = fh['%s/%s/stim/gabor_pc10' %(condition, wav_name)][:]
					#scipy.signal.resample(gabor_pc10_mat, ntimes, axis=1)
					stim_dict[wav_name].append(gabor_pc10_mat.T)
					print('gabor_mat shape is:')
					print(gabor_pc10_mat.T.shape)  
				if spectrogram:
					specs = fh['%s/%s/stim/spec' %(condition, wav_name)][:] 
					specs = scipy.signal.resample(specs, ntimes, axis=1)
					new_freq = 15 #create new feature size, from 80 to 15. Easier to fit STRF with the specified time delay
					specs = scipy.signal.resample(specs, new_freq, axis=0)
					stim_dict[wav_name].append(specs)
					print('specs shape is:')
					print(specs.shape)
					freqs = fh['%s/%s/stim/freqs' %(condition, wav_name)][:]
				if spectrogram_scaled:
					specs = fh['%s/%s/stim/spec' %(condition, wav_name)][:] 
					specs = scipy.signal.resample(specs, ntimes, axis=1)
					new_freq = 15 #create new feature size, from 80 to 15. Easier to fit STRF with the specified time delay
					specs = scipy.signal.resample(specs, new_freq, axis=0)
					specs  = specs/np.abs(specs).max()
					stim_dict[wav_name].append(specs)
					print('specs shape is:')
					print(specs.shape)

					#return freqs once
					freqs = fh['%s/%s/stim/freqs' %(condition, wav_name)][:]
				if scene_cut:
					s_cuts = fh['%s/%s/stim/scene_cut' %(condition, wav_name)][:] 
					s_cuts = scipy.signal.resample(s_cuts, ntimes, axis=1)
					stim_dict[wav_name].append(s_cuts)
					print('scene cut shape is:')
					print(s_cuts.shape)
			
				if mouthing:
					mouth = fh['%s/%s/stim/mouthing' %(condition, wav_name)][:] 
					mouth = scipy.signal.resample(mouth, ntimes, axis=0)
					stim_dict[wav_name].append(mouth.T)
					print('mouthing shape is:')
					print(mouth.T.shape)	#printing transposed version			
					
			except Exception:
				traceback.print_exc()
				
			if eeg_epochs:
				try: 
					epochs_data = fh['%s/%s/resp/%s/epochs' %(condition, wav_name, subject)][:]
					if resp_mean:
						print('taking the mean across repeats')
						epochs_data = epochs_data.mean(0)
						epochs_data = scipy.signal.resample(epochs_data.T, ntimes).T #resample to size of phnfeat
					else:
						epochs_data = scipy.signal.resample(epochs_data, ntimes, axis=2)
					print(epochs_data.shape)
					resp_dict[wav_name].append(epochs_data)
					
				except Exception:
					traceback.print_exc()
					# print('%s does not have neural data for %s'%(subject, wav_name))

					# epochs_data = []

	if spectrogram:
		return resp_dict, stim_dict, freqs

	if spectrogram_scaled:
		return resp_dict, stim_dict, freqs
		
	else:
		return resp_dict, stim_dict

def strf_features(subject, block, data_dir, condition, full_gabor = False, full_gabor_sc=False,
			full_model = False, pitchUenvs = False, pitchUphnfeat = False, envsUphnfeat = False, phnfeat_only = False, envs_only = False, 
			pitch_only = False, gabor_only = False, scene_cut=False, scene_cut_gaborpc=False, mouthUphnfeat=False, mouth_sc_phnfeat=False, phnfeat_SC=False,
			mouth_sc_gabor=False, full_gabor_sc_mouth=False, mouthing_sc = False, gabor_phnfeat=False, phnfeat_envs=False, phnfeat_pitch=False,
			delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0, 
			noICA=False, filename='full_AV_matrix'):

	"""
	Run your TRF or mTRF, depending on the number of features you input (phnfeat, envs, pitch)
	Test data is always set for TIMIT and Movie Trailers -- see stimulus_class
	
	Default STRF will always run the full model (phnfeat + pitch + envs) for each condition (TIMIT or MovieTrailers). 
	To change this, adjust the loadEEGh5 function to update True/False for the features (combination of features) you want to run. 


	Parameters
	----------
	subject : string 
		subject ID (i.e. MT0002)
	block : int
		valid inputs: 1, 2, 3
		To differentiate the listening/watching condition based on block number of EEG recording.
	filename : string 
		either: 'mT-A-only' or 'mT_AV' or 'mT_V-only' 
		This is to differentiate the listening/watching condition
	data_dir : string 
		-change this to match where .h5 is along with EEG data 
	full_model : bool
		- load envelope, phonological features, and pitch per subject for STRF analysis
		(default : True)
	pitchUenvs : bool
		- load phonological features and pitch per subject for STRF analysis (pair-wise models)
		(default : False)
	pitchUphnfeat : bool
		- load phonological features and pitch per subject for STRF analysis (pair-wise models)
		(default : False)
	envsUphnfeat : bool
		- load phonological features and envelope per subject for STRF analysis (pair-wise models)
		(default : False)
	phnfeat_only : bool 
		- only load phonological features from .h5 file
		(default : False)
	envs_only : bool 
		- only load acoustic envelope from .h5 file
		(default : False)
	pitch_only : bool 
		- only load pitch from .h5 file
		(default : False)
	gabor_only : bool 
		- only load gabor from .h5 file -- will only work for MovieTrailers
		(default : False)
	scene_cut : bool  
		- scene cut visual features only 
		(default : False)
	scene_cut_gaborpc : bool 
		- scene cut + gabor wavelet (10 features decomposed)
		(default : False)

	Returns
	-------
	wt : numpy array
	corrs : numpy array
	valphas : numpy array
	allRcorrs : numpy array
	all_corrs_shuff : list

	"""	
	if phnfeat_envs:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, 
									 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename=filename)
		strf_output = 'phnfeat_envs'		

	
	if phnfeat_pitch:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=False, pitch=True, gabor_pc10=False, 
									 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename=filename)
		strf_output = 'phnfeat_pitch'	
	if  gabor_phnfeat:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, 
									 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename=filename)
		strf_output = 'gabor_phnfeat'		

	if full_gabor: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=True, filename=filename)
		strf_output = 'pitchenvsphnfeatgabor10pc'

	if full_gabor_sc: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=True, 
									 spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False, filename=filename)
		strf_output = 'pitchenvsphnfeatgabor10pc_sc'

	if full_gabor_sc_mouth: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=True, 
									 spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=True, filename=filename)
		strf_output = 'pitchenvsphnfeatgabor10pc_sc_mouthing'
	if full_model:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=False, 
									 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename=filename)
		strf_output = 'pitchenvsphnfeat'

	if pitchUenvs: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=True, pitch=True, gabor_pc10=False, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename=filename)
		strf_output = 'envspitch'

	if pitchUphnfeat: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=False, pitch=True, gabor_pc10=False, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename=filename)
		strf_output = 'pitchphnfeat'

	if envsUphnfeat: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename=filename)
		strf_output = 'envsphnfeat'

	if phnfeat_only: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename=filename)
		strf_output = 'phnfeat'

	if envs_only: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename=filename)
		strf_output = 'envs'

	if pitch_only: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=False, pitch=True, gabor_pc10=False, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename=filename)
		strf_output = 'pitch'

	if gabor_only:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename=filename)
		strf_output = 'gabor_only'
	

	if scene_cut:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False, filename=filename)
		strf_output = 'scene_cut'

	if scene_cut_gaborpc:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False, filename=filename)
		strf_output = 'scene_cut_gabor'

	if mouthUphnfeat:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=True, filename=filename)
		strf_output = 'mouthing_phnfeat'		
	
	if mouth_sc_phnfeat:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=True, filename=filename)
		strf_output = 'mouthing_SC_phnfeat'

	if mouth_sc_gabor:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=True, filename=filename)
		strf_output = 'mouthing_SC_gabor'

	if phnfeat_SC:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False, filename=filename)
		strf_output = 'phnfeat_SC'		
	if mouthing_sc:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=True, filename=filename)
		strf_output = 'mouth_SC'			

	stim_list = []
	for key in resp_dict.keys():
		stim_list.append(key)
	# Populate stim and resp lists (these will be used to get tStim and tResp, or vStim and vResp) -- based on TIMIT or MT from loading h5 above
	stim = stim_dict 
	resp = resp_dict 

	if condition == 'AV':
		test_set =['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav']

	if condition == 'A':
		test_set =['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav']

	if condition == 'V':
		test_set = ['paddington-2-trailer-1_a720p_visualonly_notif.wav', 'insideout-tlr2zzyy32_a720p_visualonly_notif.wav']

	if subject == 'MT0034':
		if condition == 'A':
			test_set = ['insideout-tlr2zzyy32_a720p.wav']

	if subject == 'MT0035':
		if condition == 'AV':
			test_set = ['insideout-tlr2zzyy32_a720p.wav']	
		if condition == 'V':
			test_set = ['paddington-2-trailer-1_a720p_visualonly_notif.wav']		


	print(test_set)

	# if condition == 'AV':
	# 	test_set = ['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav']

			
	# if condition == 'A':
	# 	test_set = ['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav']

	all_stimuli = [k for k in (stim_dict).keys() if len(resp_dict[k]) > 0]
	training_set = np.setdiff1d(all_stimuli, test_set)

		
	val_inds = np.zeros((len(all_stimuli),), dtype=np.bool)
	train_inds = np.zeros((len(all_stimuli),), dtype=np.bool)
	for i in np.arange(len(all_stimuli)):
		if all_stimuli[i] in test_set:
			print(all_stimuli[i])
			val_inds[i] = True
		else:
			train_inds[i] = True

	print("Total number of training sentences:")
	print(sum(train_inds))
	print("Total number of validation sentences:")
	print(sum(val_inds))

	train_inds = np.where(train_inds==True)[0]
	val_inds = np.where(val_inds==True)[0]

	print("Training indices:")
	print(train_inds)
	print("Validation indices:")
	print(val_inds)
	
	# For logging compute times, debug messages
	
	logging.basicConfig(level=logging.DEBUG) 

	#time delays used in STRF
	delays = np.arange(np.floor((delay_min-wt_pad)*fs), np.ceil((delay_max+wt_pad)*fs), dtype=int) #create array to pass time delays in

	print("Delays:", delays)

	# Regularization parameters (alphas - also sometimes called lambda)
	alphas = np.hstack((0, np.logspace(2,8,20))) # Gives you 15 values between 10^2 and 10^8

	nalphas = len(alphas)
	use_corr = True # Use correlation between predicted and validation set as metric for goodness of fit
	single_alpha = True # Use the same alpha value for all electrodes (helps with comparing across sensors)
	nboots = 20 # How many bootstraps to do. (This is number of times you take a subset of the training data to find the best ridge parameter)

	all_wts = [] # STRF weights (this will be ndelays x channels)
	all_corrs = [] # correlation performance of length [nchans]
	all_corrs_shuff = [] # List of lists, how good is a random model

	# train_inds and val_inds are defined in the cell above, and is based on specifying stimuli heard more than once, 
	# which will be in the test set, and the remaining stimuli will be in the training set 
	current_stim_list_train = np.array([all_stimuli[r][0] for r in train_inds])
	current_stim_list_val = np.array([all_stimuli[r][0] for r in val_inds])

	# Create training and validation response matrices
	print(resp_dict[training_set[0]][0].shape)
	print(test_set)
	
	print(len(training_set))
	for r in training_set:
		print(r)

	
	tResp = np.hstack([resp_dict[r][0] for r in training_set]).T
	vResp = np.hstack([resp_dict[r][0] for r in test_set]).T


	# Create training and validation stimulus matrices

	tStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in training_set]))
	vStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in test_set]))
	tStim_temp = tStim_temp/tStim_temp.max(0)
	vStim_temp = vStim_temp/vStim_temp.max(0)


	print('**********************************')
	print(tStim_temp.max(0).shape)
	print(vStim_temp.max(0).shape)
	print('**********************************')

	tStim = make_delayed(tStim_temp, delays)
	vStim = make_delayed(vStim_temp, delays)

	chunklen = int(len(delays)*3) # We will randomize the data in chunks 
	nchunks = np.floor(0.2*tStim.shape[0]/chunklen).astype('int')

	nchans = tResp.shape[1] # Number of electrodes/sensors
	
	#plot neural data responses and binary feature matrix to identify timing for all phonological features based on stimulus 
	plt.figure(figsize=(15,3))
	elec = 15
	nsec_start = 390
	nsec_end = 410
	plt.figure(figsize=(10,5))
	plt.subplot(2,1,1)
	plt.plot(tResp[int(fs*nsec_start):int(fs*nsec_end),42]/tResp[:int(fs*nsec_start),elec].max(), 'r') #this is the response itself - EEG data (training)
	plt.subplot(2,1,2)
	plt.imshow(tStim_temp[int(fs*nsec_start):int(fs*nsec_end),:].T, aspect='auto', vmin=-1, vmax=1,interpolation='nearest', cmap=cm.RdBu) #envelope of trianing sound stimuli
	plt.colorbar()
	
	print('*************************')
	print(vStim.shape)
	print('*************************')
	print('printing vResp: ')
	print(vResp.shape)
	print('*************************')
	
	# Fit the STRFs
	wt, corrs, valphas, allRcorrs, valinds, pred, Pstim = bootstrap_ridge(tStim, tResp, vStim, vResp, 
																		  alphas, nboots, chunklen, nchunks, 
																		  use_corr=use_corr,  single_alpha = single_alpha, 
																		  use_svd=False, corrmin = 0.05,
																		  joined=[np.array(np.arange(nchans))])

	print('*************************')
	print('pred value is: ')
	print(pred)
	print('*************************')
	print(wt.shape)
	#when wt padding is 0:
	if wt_pad > 0:

		good_delays = np.ones((len(delays), 1), dtype=np.bool)
		good_delays[np.where(delays<0)[0]] = False
		good_delays[np.where(delays>=np.ceil(delay_max*fs))[0]] = False
		good_delays = good_delays.ravel()



		print("Reshaping weight matrix to get rid of padding on either side")
		wt2 = wt.reshape((len(delays), -1, wt.shape[1]))[len(np.where(delays<0)[0]):-(len(np.where(delays<0)[0])),:,:]
		wt2 = wt2.reshape((wt2.shape[0]*wt2.shape[1], wt2.shape[2]))
	else:
		wt2 = wt

	print(wt2.shape)
	all_wts.append(wt2)
	all_corrs.append(corrs)
	

	plt.figure()
	# Plot correlations of model by electrode
	plt.plot(all_corrs[0])
	all_wts[0].shape[0]/14

	plt.figure()
	
	
	#BOOTSTRAPPING BELOW 

	# Determine whether the correlations we see for model performance are significant
	# by shuffling the data and re-fitting the models to see what "random" performance
	# would look like.
	#
	# How many bootstraps to do for determining bootstrap significance
	# The minimum p-value you can get from this is 1/nboots_shuffle
	# So for nboots_shuffle = 100, you can get p_values from 0.01 to 1.0
	nboots_shuffle = 100 

	nresp, nelecs = tStim.shape
	allinds = range(nresp)

	print("Determining significance of the correlation values using a bootstrap with %d iterations"%(nboots_shuffle))
	for n in np.arange(nboots_shuffle):
		print("Bootstrap %d/%d"%(n+1, nboots_shuffle))
		indchunks = list(zip(*[iter(allinds)]*chunklen))
		random.shuffle(indchunks)
		shuff_inds = list(itools.chain(*indchunks[:nchunks]))
		tStim_shuff = tStim.copy()
		tResp_shuff = tResp.copy()
		tStim_shuff = tStim_shuff[shuff_inds,:]
		tResp_shuff = tResp_shuff[:len(shuff_inds),:]

		corrs_shuff = eigridge_corr(tStim_shuff, vStim, tResp_shuff, vResp, [valphas[0]], corrmin = 0.05)
		all_corrs_shuff.append(corrs_shuff)

	# all_corrs_shuff is a list of length nboots_shuffle
	# Each element is the correlation for a random model for each of the 64 electrodes for that iteration
	# We use this to figure out [nboots_shuffle] possible values of random correlations for each electrode,
	# then use this to determine if the correlations we're actually measuring with the non-shuffled data are 
	# significantly higher than this
	
	
	# Get the p values of each of the significant correlations
	all_pvals = [] 

	all_c_s=np.vstack((all_corrs_shuff)) # Get the shuffled random correlations for this model

	# Is the correlation of the model greater than the shuffled correlation for random data?
	h_val = np.array([all_corrs[0] > all_c_s[c] for c in np.arange(len(all_c_s))])
	print(h_val.shape)

	# Count the number of times out of nboots_shuffle that the correlation is greater than 
	# random, subtract from 1 to get the bootstrapped p_val (one per electrode)
	p_val = 1-h_val.sum(0)/nboots_shuffle

	all_pvals.append(p_val)
	
	#load in your ICA data for your particular subject - will be used to fit significant responses on topo maps 
	

	if filename=='full_AV_matrix_noICA':
		eeg_file=f'{data_dir}{subject}/downsampled_128/{subject}_B{block}_rejection_mas_raw.fif'
		raw=mne.io.read_raw_fif(eeg_file, preload=True)
		raw.drop_channels(['hEOG', 'vEOG'])
	else:
		
		raw = load_raw_EEG(subject, block, data_dir, file='postICA_rejected')
		if 'STI 014' in raw.info['ch_names']:
			raw.drop_channels(['vEOG', 'hEOG', 'STI 014'])
		else:
			raw.drop_channels(['vEOG', 'hEOG'])

	chnames = raw.info['ch_names']
	chnames = np.array(chnames)

	# Find the maximum correlation across the shuffled and real data
	max_corr = np.max(np.vstack((all_corrs_shuff[0], all_corrs[0])))+0.01 #why is this structured the way it is? Adding 0.01?

	# Plot the correlations for each channel separately
	plt.figure(figsize=(15,3))
	plt.plot(all_corrs[0])

	#Plot an * if the correlation is significantly higher than chance at p<0.05
	for i,p in enumerate(all_pvals[0]):
		if p<0.05:
			plt.text(i, max_corr, '*')

	# Plot the shuffled correlation distribution
	shuffle_mean = np.vstack((all_corrs_shuff)).mean(0) #correlation output form model -- which electrodes are correlated w/ each other, take average of this
	shuffle_stderr = np.vstack((all_corrs_shuff)).std(0)/np.sqrt(nboots_shuffle) #normalization of which electrodes are correlated w/ each other

	# plt.fill_between(np.arange(nchans), shuffle_mean-shuffle_stderr, #normalization here
	# 				 shuffle_mean+shuffle_stderr, color=[0.5, 0.5, 0.5])
	# plt.plot(shuffle_mean, color='k')
	# plt.gca().set_xticks(np.arange(len(all_corrs[0])))
	# plt.gca().set_xticklabels(chnames, rotation=90)
	# plt.xlabel('Channel')
	# plt.ylabel('Model performance')
	# plt.legend(['Actual data','Null distribution'])
	# plt.savefig('%s/%s/%s_ch_distribution_%s_EOG.pdf' %(data_dir, subject, strf_output, condition)) #save fig

	
	#plot the significant correlations for participant on topo map 
	significant_corrs = np.array(all_corrs[0])
	significant_corrs[np.array(all_pvals[0])>0.05] = 0

	# plt.figure(figsize=(5,5))
	# print(['eeg']*2)
	# info = mne.create_info(ch_names=raw.info['ch_names'][:64], sfreq=raw.info['sfreq'], ch_types=64*['eeg'])
	# raw2 = mne.io.RawArray(np.zeros((64,10)), info)
	# montage = mne.channels.read_custom_montage('%s/montage/AP-128.bvef' %(data_dir))
	# raw2.set_montage(montage) #set path for MNE montage file
	# mne.viz.plot_topomap(significant_corrs, raw2.info, vmin=0, vmax=max_corr)
	#plt.savefig('%s/%s/%s_topomap_%s.png' %(data_dir, subject, strf_output, stimulus_class)) #save fig

	#plt.savefig('Topomap_MT.png')
	print(np.array(all_wts).shape)

	#save STRF as .h5 file based on condition type:

	#strf_file = '%s/%s/%s_STRF_by_pitchenvsphnfeat_MT.hf5'%(data_dir, subject, subject)
	if noICA:
		name='noICA'
		strf_file = '%s/%s/%s_STRF_by_%s_%s_%s.hf5'%(data_dir, subject, subject, strf_output, condition,name)
	else:
		strf_file = '%s/%s/%s_STRF_by_%s_%s.hf5'%(data_dir, subject, subject, strf_output, condition)
	print("Saving file to %s"%(strf_file))
	with h5py.File(strf_file, 'w') as f:
		f.create_dataset('/wts_mt', data = np.array(all_wts[0])) #weights for MT/timit
		f.create_dataset('/corrs_mt', data = np.array(all_corrs[0])) #correlations for MT/timit
		f.create_dataset('/train_inds_mt', data = train_inds) #training sets for MT/timit
		f.create_dataset('/val_inds_mt', data = val_inds) #validation sets for MT (test set)/timit
		f.create_dataset('/pvals_mt', data = all_pvals) #save all pvals 
		f.create_dataset('/delays_mt', data = delays) #save delays
		f.create_dataset('/valphas_mt', data = valphas) #save alpha value used for bootstrapping
		f.create_dataset('/allRcorrs_mt', data = allRcorrs) 
		f.create_dataset('/all_corrs_shuff_mt', data = all_corrs_shuff) #

	#THIS PLOT SHOWS THE DISTRIBUTION OF THE PREDICTED VS ACTUAL CORRELATIONS FOR EACH STIMULUS SET RUN
	np.vstack((all_corrs_shuff)).ravel().shape


	# plt.figure()

	# plt.hist(np.hstack((all_corrs_shuff)).ravel(), bins=np.arange(-0.2,max_corr,0.005), alpha=0.5, density=True)
	# plt.hist(all_corrs[0], bins=np.arange(-0.2,max_corr,0.005), alpha=0.5, density=True)
	# plt.xlabel('Model fits (r-values)')
	# plt.ylabel('Number')
	# plt.title('Correlation histograms')
	# plt.legend(['Null distribution', 'EEG data'])
	if noICA:
		plt.savefig('%s/%s/%s_corrHistogram_%s_%s.pdf' %(data_dir, subject, condition, strf_output, name)) #save fig
	else:
		plt.savefig('%s/%s/%s_corrHistogram_%s.pdf' %(data_dir, subject, condition, strf_output))
	#Number of data points for a given bin that occurred 

	return wt, corrs, valphas, allRcorrs, all_corrs_shuff

def eog_strf_features(subject, data_dir, condition, gabor_only = False, scene_cut=False, scene_cut_gaborpc=False, 
					  delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0, filename='EOG_AV_matrix'):

	"""
	Run your TRF or mTRF, depending on the number of features you input (phnfeat, envs, pitch)
	Test data is always set for TIMIT and Movie Trailers -- see stimulus_class
	
	Default STRF will always run the full model (phnfeat + pitch + envs) for each condition (TIMIT or MovieTrailers). 
	To change this, adjust the loadEEGh5 function to update True/False for the features (combination of features) you want to run. 


	Parameters
	----------
	subject : string 
		subject ID (i.e. MT0002)
	block : int
		valid inputs: 1, 2, 3
		To differentiate the listening/watching condition based on block number of EEG recording.
	filename : string 
		either: 'mT-A-only' or 'mT_AV' or 'mT_V-only' 
		This is to differentiate the listening/watching condition
	data_dir : string 
		-change this to match where .h5 is along with EEG data 

	gabor_only : bool 
		- only load gabor from .h5 file -- will only work for MovieTrailers
		(default : False)
	scene_cut : bool  
		- scene cut visual features only 
		(default : False)
	scene_cut_gaborpc : bool 
		- scene cut + gabor wavelet (10 features decomposed)
		(default : False)

	Returns
	-------
	wt : numpy array
	corrs : numpy array
	valphas : numpy array
	allRcorrs : numpy array
	all_corrs_shuff : list

	"""	

	if gabor_only:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False, filename='EOG_AV_matrix')
		strf_output = 'gabor_only'
	

	if scene_cut:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False, filename='EOG_AV_matrix')
		strf_output = 'scene_cut'

	if scene_cut_gaborpc:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False, filename='EOG_AV_matrix')
		strf_output = 'scene_cut_gabor'

		

	stim_list = []
	for key in resp_dict.keys():
		stim_list.append(key)
	# Populate stim and resp lists (these will be used to get tStim and tResp, or vStim and vResp) -- based on TIMIT or MT from loading h5 above
	stim = stim_dict 
	resp = resp_dict 

	if condition == 'AV':
		test_set =['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav']

	if condition == 'A':
		test_set =['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav']

	if condition == 'V':
		test_set = ['paddington-2-trailer-1_a720p_visualonly_notif.wav', 'insideout-tlr2zzyy32_a720p_visualonly_notif.wav']

	if subject == 'MT0034':
		if condition == 'A':
			test_set = ['insideout-tlr2zzyy32_a720p.wav']

	if subject == 'MT0035':
		if condition == 'AV':
			test_set = ['insideout-tlr2zzyy32_a720p.wav']	
		if condition == 'V':
			test_set = ['paddington-2-trailer-1_a720p_visualonly_notif.wav']		


	print(test_set)


	all_stimuli = [k for k in (stim_dict).keys() if len(resp_dict[k]) > 0]
	training_set = np.setdiff1d(all_stimuli, test_set)

		
	val_inds = np.zeros((len(all_stimuli),), dtype=np.bool)
	train_inds = np.zeros((len(all_stimuli),), dtype=np.bool)
	for i in np.arange(len(all_stimuli)):
		if all_stimuli[i] in test_set:
			print(all_stimuli[i])
			val_inds[i] = True
		else:
			train_inds[i] = True

	print("Total number of training sentences:")
	print(sum(train_inds))
	print("Total number of validation sentences:")
	print(sum(val_inds))

	train_inds = np.where(train_inds==True)[0]
	val_inds = np.where(val_inds==True)[0]

	print("Training indices:")
	print(train_inds)
	print("Validation indices:")
	print(val_inds)
	
	# For logging compute times, debug messages
	
	logging.basicConfig(level=logging.DEBUG) 

	#time delays used in STRF
	delays = np.arange(np.floor((delay_min-wt_pad)*fs), np.ceil((delay_max+wt_pad)*fs), dtype=int) #create array to pass time delays in

	print("Delays:", delays)

	# Regularization parameters (alphas - also sometimes called lambda)
	alphas = np.hstack((0, np.logspace(2,8,20))) # Gives you 15 values between 10^2 and 10^8

	nalphas = len(alphas)
	use_corr = True # Use correlation between predicted and validation set as metric for goodness of fit
	single_alpha = True # Use the same alpha value for all electrodes (helps with comparing across sensors)
	nboots = 20 # How many bootstraps to do. (This is number of times you take a subset of the training data to find the best ridge parameter)

	all_wts = [] # STRF weights (this will be ndelays x channels)
	all_corrs = [] # correlation performance of length [nchans]
	all_corrs_shuff = [] # List of lists, how good is a random model

	# train_inds and val_inds are defined in the cell above, and is based on specifying stimuli heard more than once, 
	# which will be in the test set, and the remaining stimuli will be in the training set 
	current_stim_list_train = np.array([all_stimuli[r][0] for r in train_inds])
	current_stim_list_val = np.array([all_stimuli[r][0] for r in val_inds])

	# Create training and validation response matrices
	print(resp_dict[training_set[0]][0].shape)
	print(test_set)
	
	print(len(training_set))
	for r in training_set:
		print(r)

	
	tResp = np.hstack([resp_dict[r][0] for r in training_set]).T
	vResp = np.hstack([resp_dict[r][0] for r in test_set]).T


	# Create training and validation stimulus matrices

	tStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in training_set]))
	vStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in test_set]))
	tStim_temp = tStim_temp/tStim_temp.max(0)
	vStim_temp = vStim_temp/vStim_temp.max(0)


	print('**********************************')
	print(tStim_temp.max(0).shape)
	print(vStim_temp.max(0).shape)
	print('**********************************')

	tStim = make_delayed(tStim_temp, delays)
	vStim = make_delayed(vStim_temp, delays)

	chunklen = int(len(delays)*3) # We will randomize the data in chunks 
	nchunks = np.floor(0.2*tStim.shape[0]/chunklen).astype('int')

	nchans = tResp.shape[1] # Number of electrodes/sensors
	
	
	print('*************************')
	print(vStim.shape)
	print('*************************')
	print('printing vResp: ')
	print(vResp.shape)
	print('*************************')
	
	# Fit the STRFs
	wt, corrs, valphas, allRcorrs, valinds, pred, Pstim = bootstrap_ridge(tStim, tResp, vStim, vResp, 
																		  alphas, nboots, chunklen, nchunks, 
																		  use_corr=use_corr,  single_alpha = single_alpha, 
																		  use_svd=False, corrmin = 0.05,
																		  joined=[np.array(np.arange(nchans))])

	print('*************************')
	print('pred value is: ')
	print(pred)
	print('*************************')
	print(wt.shape)
	#when wt padding is 0:
	if wt_pad > 0:

		good_delays = np.ones((len(delays), 1), dtype=np.bool)
		good_delays[np.where(delays<0)[0]] = False
		good_delays[np.where(delays>=np.ceil(delay_max*fs))[0]] = False
		good_delays = good_delays.ravel()



		print("Reshaping weight matrix to get rid of padding on either side")
		wt2 = wt.reshape((len(delays), -1, wt.shape[1]))[len(np.where(delays<0)[0]):-(len(np.where(delays<0)[0])),:,:]
		wt2 = wt2.reshape((wt2.shape[0]*wt2.shape[1], wt2.shape[2]))
	else:
		wt2 = wt

	print(wt2.shape)
	all_wts.append(wt2)
	all_corrs.append(corrs)
	

	plt.figure()
	# Plot correlations of model by electrode
	plt.plot(all_corrs[0])
	all_wts[0].shape[0]/14
	
	
	#BOOTSTRAPPING BELOW 

	# Determine whether the correlations we see for model performance are significant
	# by shuffling the data and re-fitting the models to see what "random" performance
	# would look like.
	#
	# How many bootstraps to do for determining bootstrap significance
	# The minimum p-value you can get from this is 1/nboots_shuffle
	# So for nboots_shuffle = 100, you can get p_values from 0.01 to 1.0
	nboots_shuffle = 100 

	nresp, nelecs = tStim.shape
	allinds = range(nresp)

	print("Determining significance of the correlation values using a bootstrap with %d iterations"%(nboots_shuffle))
	for n in np.arange(nboots_shuffle):
		print("Bootstrap %d/%d"%(n+1, nboots_shuffle))
		indchunks = list(zip(*[iter(allinds)]*chunklen))
		random.shuffle(indchunks)
		shuff_inds = list(itools.chain(*indchunks[:nchunks]))
		tStim_shuff = tStim.copy()
		tResp_shuff = tResp.copy()
		tStim_shuff = tStim_shuff[shuff_inds,:]
		tResp_shuff = tResp_shuff[:len(shuff_inds),:]

		corrs_shuff = eigridge_corr(tStim_shuff, vStim, tResp_shuff, vResp, [valphas[0]], corrmin = 0.05)
		all_corrs_shuff.append(corrs_shuff)

	# all_corrs_shuff is a list of length nboots_shuffle
	# Each element is the correlation for a random model for each of the 64 electrodes for that iteration
	# We use this to figure out [nboots_shuffle] possible values of random correlations for each electrode,
	# then use this to determine if the correlations we're actually measuring with the non-shuffled data are 
	# significantly higher than this
	
	
	# Get the p values of each of the significant correlations
	all_pvals = [] 

	all_c_s=np.vstack((all_corrs_shuff)) # Get the shuffled random correlations for this model

	# Is the correlation of the model greater than the shuffled correlation for random data?
	h_val = np.array([all_corrs[0] > all_c_s[c] for c in np.arange(len(all_c_s))])
	print(h_val.shape)

	# Count the number of times out of nboots_shuffle that the correlation is greater than 
	# random, subtract from 1 to get the bootstrapped p_val (one per electrode)
	p_val = 1-h_val.sum(0)/nboots_shuffle

	all_pvals.append(p_val)
	
	#load in your ICA data for your particular subject - will be used to fit significant responses on topo maps 
	

	# Find the maximum correlation across the shuffled and real data
	max_corr = np.max(np.vstack((all_corrs_shuff[0], all_corrs[0])))+0.01 #why is this structured the way it is? Adding 0.01?

	# Plot the correlations for each channel separately
	plt.figure(figsize=(15,3))
	plt.plot(all_corrs[0])

	# Plot an * if the correlation is significantly higher than chance at p<0.05
	for i,p in enumerate(all_pvals[0]):
		if p<0.05:
			plt.text(i, max_corr, '*')

	# Plot the shuffled correlation distribution
	shuffle_mean = np.vstack((all_corrs_shuff)).mean(0) #correlation output form model -- which electrodes are correlated w/ each other, take average of this
	shuffle_stderr = np.vstack((all_corrs_shuff)).std(0)/np.sqrt(nboots_shuffle) #normalization of which electrodes are correlated w/ each other



	
	#plot the significant correlations for participant on topo map 
	significant_corrs = np.array(all_corrs[0])
	significant_corrs[np.array(all_pvals[0])>0.05] = 0



	#plt.savefig('Topomap_MT.png')
	print(np.array(all_wts).shape)

	#save STRF as .h5 file based on condition type:

	#strf_file = '%s/%s/%s_STRF_by_pitchenvsphnfeat_MT.hf5'%(data_dir, subject, subject)
	strf_file = '%s/%s/%s_STRF_by_%s_%s_EOG.hf5'%(data_dir, subject, subject, strf_output, condition)
	print("Saving file to %s"%(strf_file))
	with h5py.File(strf_file, 'w') as f:
		f.create_dataset('/wts_mt', data = np.array(all_wts[0])) #weights for MT/timit
		f.create_dataset('/corrs_mt', data = np.array(all_corrs[0])) #correlations for MT/timit
		f.create_dataset('/train_inds_mt', data = train_inds) #training sets for MT/timit
		f.create_dataset('/val_inds_mt', data = val_inds) #validation sets for MT (test set)/timit
		f.create_dataset('/pvals_mt', data = all_pvals) #save all pvals 
		f.create_dataset('/delays_mt', data = delays) #save delays
		f.create_dataset('/valphas_mt', data = valphas) #save alpha value used for bootstrapping
		f.create_dataset('/allRcorrs_mt', data = allRcorrs) 
		f.create_dataset('/all_corrs_shuff_mt', data = all_corrs_shuff) #

	#THIS PLOT SHOWS THE DISTRIBUTION OF THE PREDICTED VS ACTUAL CORRELATIONS FOR EACH STIMULUS SET RUN
	np.vstack((all_corrs_shuff)).ravel().shape


	plt.figure()

	plt.hist(np.hstack((all_corrs_shuff)).ravel(), bins=np.arange(-0.2,max_corr,0.005), alpha=0.5, density=True)
	plt.hist(all_corrs[0], bins=np.arange(-0.2,max_corr,0.005), alpha=0.5, density=True)
	plt.xlabel('Model fits (r-values)')
	plt.ylabel('Number')
	plt.title('Correlation histograms')
	plt.legend(['Null distribution', 'EEG data'])
	plt.savefig('%s/%s/%s_corrHistogram_%s_EOG.pdf' %(data_dir, subject, condition, strf_output)) #save fig
	#Number of data points for a given bin that occurred 

	return wt, corrs, valphas, allRcorrs, all_corrs_shuff

def predict_response(wt, vStim, vResp):
	''' Predict the response to [vStim] given STRF weights [wt],
	compare to the actual response [vResp], and return the correlation
	between predicted and actual response.
	Inputs:
		wt: [features x delays] x electrodes, your STRF weights
		vStim: time x [features x delays], your delayed stimulus matrix
		vResp: time x electrodes, your true response to vStim
	Outputs:
		corr: correlation between predicted and actual response
		pred: prediction for each electrode [time x electrodes]
	'''
	nchans = wt.shape[1]
	print('Calculating prediction...')
	pred = np.dot(vStim, wt)

	print('Calculating correlation')
	corr = np.array([np.corrcoef(vResp[:,i], pred[:,i])[0,1] for i in np.arange(nchans)])

	return corr, pred

def cross_prediction(data_dir, subject_list, test_cond='V', train_cond='AV', test_feature='scene_cut_gabor', 
			 train_feature='scene_cut_gabor', delay_min=0.0, delay_max=0.6, wt_pad=0.0, fs=128., av_sigs=True):
	'''
	Parameters:
	----------
	data_dir : string
		- path where data exist
	subject_list : list
		- list of strings containing subject IDs
	test_cond : 'string
		- either A or V which is the unimodal stim used as the test set
	train_cond : 'string
		- default is AV where weights from the multimodal stimuli will be used 
	test_feature : string
		- either pitchenvsphnfeat or scene_cut_gabor as features to use for A and V, respectively. Must be same as train_feature
	train_feature : string
		- either pitchenvsphnfeat or scene_cut_gabor as features to use for A and V, respectively. Must be same as test_feature
	delay_min : float
		- default of 0.0 based on STRF time lag fit previously
	delay_max : float
		- default of 0.6 based on STRF time lag fit previously
	wt_pad : float
		- default of 0.0 based on STRF wt_pad used to previously fit model
	fs : float
		- sampling rate of EEG, default for 128.0 Hz
	av_sigs : bool
		- Default: False
			- Set audio only or visual only (uni-imodal) stimulus correlation threshold of r > 0.05 as significance
			- If True: Set audiovisual (multimodal) stimulus correlation threshold of r > 0.05 as significance
	'''	
	fig = plt.figure(figsize=(15,9))

	corrs = []
	corrs_sig = []
	corrs_nonsig = []

	data = []

	for idx, s in enumerate(subject_list):
		with h5py.File(f'{data_dir}/{s}/{s}_STRF_by_{train_feature}_{train_cond}.hf5', 'r') as fh:
			wts_train = fh['wts_mt'][:]
			print(wts_train.shape)
		if train_feature == 'pitchenvsphnfeatgabor10pc_sc':
			print('using full AV model')
			nfeats=27
			wts2 = wts_train.reshape(np.int(wts_train.shape[0]/nfeats),nfeats,wts_train.shape[1] )
			if test_feature == 'pitchenvsphnfeat':
				aud_feats = wts2[:,:16,:]
				wts_train = aud_feats.reshape((aud_feats.shape[0]*aud_feats.shape[1], aud_feats.shape[2]))
			elif test_feature == 'scene_cut_gabor':
				vis_feats = wts2[:,16:,:]
				wts_train = vis_feats.reshape((vis_feats.shape[0]*vis_feats.shape[1], vis_feats.shape[2]))
			elif test_feature == 'gabor_only':
				vis_feats = wts2[:,16:-1,:]
				wts_train = vis_feats.reshape((vis_feats.shape[0]*vis_feats.shape[1], vis_feats.shape[2]))
			elif test_feature == 'scene_cut':
				vis_feats = wts2[:,-1:,:]
				wts_train = vis_feats.reshape((vis_feats.shape[0]*vis_feats.shape[1], vis_feats.shape[2]))
			else:
				print('test_feature not defined')
		else:
			print('Not using pitchenvsphnfeatgabor10pc_sc full AV pretained weights ')
		
		print(wts_train.shape)

		with h5py.File(f'{data_dir}/{s}/{s}_STRF_by_{test_feature}_{test_cond}.hf5', 'r') as fh:
			corrs_test = fh['corrs_mt'][:]	#correlation values from A-only corrs with pitchenvsphnfeat
			validation_test = fh['val_inds_mt'][:]
			pval_test = fh['pvals_mt'][:]
			test_nonsig = np.where(corrs_test < 0.05)
			test_sig = np.where(corrs_test > 0.05)
			# test_nonsig = np.where(pval_test[0] > 0.05)
			# test_sig = np.where(pval_test[0] < 0.05)	

		print('*************************')
		print('Now processing %s' %(s))
		print('*************************')
		delays = np.arange(np.floor((delay_min-wt_pad)*fs), np.ceil((delay_max+wt_pad)*fs), dtype=int)

		if test_feature == 'pitchenvsphnfeat':
			resp_dict, stim_dict = loadEEGh5(s, data_dir, test_cond, eeg_epochs=True, resp_mean = True,
									binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=False, 
									spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False)
		elif test_feature == 'scene_cut_gabor':
			resp_dict, stim_dict = loadEEGh5(s, data_dir, test_cond, eeg_epochs=True, resp_mean = True,
									binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, 
									spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False)
		elif test_feature == 'gabor_only':
			resp_dict, stim_dict = loadEEGh5(s, data_dir, test_cond, eeg_epochs=True, resp_mean = True,
									binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, 
									spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False)

		elif test_feature == 'scene_cut':
			resp_dict, stim_dict = loadEEGh5(s, data_dir, test_cond, eeg_epochs=True, resp_mean = True,
									binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
									spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False)									
		else:
			print('test feature not defined')

		all_stimuli = []
		for i in resp_dict.keys():
			x = (i if resp_dict[i] else 'False')
			if x != 'False':
				all_stimuli.append(x)
		vResp = np.hstack([resp_dict[r][0] for r in [all_stimuli[m] for m in validation_test]]).T

		vStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in [all_stimuli[m] for m in validation_test]]))
		vStim = make_delayed(vStim_temp, delays)

		print('******************************')
		print(wts_train.shape)
		print(vStim.shape)
		print(vResp.shape)
		print('******************************')
		test_condition_corrs, train_condition_pred = predict_response(wts_train, vStim, vResp) 

		ds = load_raw_EEG(s, 1, data_dir)

		ds.drop_channels(['vEOG', 'hEOG'])
		chnames = ds.info['ch_names']
		chnames = np.array(chnames)


		corrs.append([corrs_test, test_condition_corrs])



		
		if av_sigs:
			corrs_nonsig.append([corrs_test[test_nonsig], test_condition_corrs[test_nonsig]])
			corrs_sig.append([corrs_test[test_sig], test_condition_corrs[test_sig]])
			title_name = 'av significant, corr > 0.05'
		else:
			test_sig = np.where(test_condition_corrs>0.05)
			test_nonsig = np.where(test_condition_corrs<0.05)
			corrs_nonsig.append([corrs_test[test_nonsig], test_condition_corrs[test_nonsig]])
			corrs_sig.append([corrs_test[test_sig], test_condition_corrs[test_sig]])
			title_name = f'unimodal {test_cond} significance, corr > 0.05'

			#add to data list to create dataframe
	for subj in subject_list:
			for idx, corr in enumerate(corrs_test):
				data.append({'subject': subj, 'unimodal': corrs_test[idx], 'multimodal': test_condition_corrs[idx]})
	df_xpred = pd.DataFrame(data) #create dataframe
	df_xpred.to_csv(f'xpred_{train_feature}_{train_cond}_{test_cond}.csv')
	
	for idx, m in enumerate(subject_list):
		if test_feature == 'gabor_only':
			plt.plot(corrs[idx][0], corrs[idx][1], '.', color='red', label=m)
		elif test_feature == 'scene_cut':
			plt.plot(corrs[idx][0], corrs[idx][1], '.', color='blue', label=m)
		else:
			plt.plot(corrs[idx][0], corrs[idx][1], '.', color='grey', label=m)
		# plt.plot(corrs_sig[idx][0], corrs_sig[idx][1], '.', label=m, alpha=0.8)
		# #plt.plot(corrs_nonsig[idx][0], corrs_nonsig[idx][1], '.', color='#DCDBDB', alpha=0.7)
		# plt.plot(corrs_nonsig[idx][0], corrs_nonsig[idx][1], '.', label=m, alpha=0.8)

	#plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
	plt.xlabel('Predict %s from %s' %(test_cond, test_cond))
	plt.ylabel('Predict %s from %s' %(test_cond, train_cond))
	plt.axis('square')
	plt.axvline(color='black')
	plt.axhline(color='black')
	plt.title(f'{title_name}')

	[slope, intercept, r_value, p_value, std_err] = scipy.stats.linregress(np.hstack(corrs)[0], np.hstack(corrs)[1])

 
	if test_feature == 'pitchenvsphnfeat':
		plt.plot([-0.05, 0.125], [-0.05, 0.125], 'black', label='unity', linestyle='dashed')
		plt.plot([-0.05,0.125], [-0.05*slope + intercept, 0.125*slope+intercept], color='red')

	elif test_feature == 'scene_cut_gabor':
		plt.plot([-0.15, 0.35], [-0.15, 0.35], 'black', label='unity', linestyle='dashed')
		plt.plot([-0.15,0.35], [-0.15*slope + intercept, 0.35*slope+intercept], color='red')
		plt.gca().set_xticks([-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]) 
		plt.gca().set_yticks([-0.15, -0.1,-0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
	print(r_value) #correlation with person 
	print(p_value) #high sigificant

	#Wilcoxon test:
	res = wilcoxon(np.concatenate(np.array(corrs)[:,0,:]), np.concatenate(np.array(corrs)[:,1,:]), alternative='greater')
	print(res)
	plt.savefig(f'{save_dir}/{test_feature}_{train_feature}_xpred.pdf')

	return corrs

def get_roi():
	rois = {'frontal': ['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'FC6', 'FC2', 
			 				'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AF4', 'AFz', 'F1', 'F5',  'FC3', 'FCz', 'FC4', 'F6', 'F2', 'AF2', 'AF8'], 
		 		'central': ['C3','CP5', 'CP1', 'CP6', 'Cz', 'C4', 'CP2', 'C1', 
							'C5', 'CP3', 'CPz', 'CP4', 'C6', 'C2'],
		 		'parietal': ['Pz', 'P3', 'P7', 'P4', 'P8', 'P1', 'P5', 'P6', 'P2'],
				 'temporal': ['T7', 'TP9', 'TP10', 'T8', 'TP7', 'TP8', 'FT8','FT7', 'FT9', 'FT10'],
				 'occipital': ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4',  'PO8']}
	
	# color_dict = {'frontal': '#E3D26F', 
	# 			'central': '#CA895F', 
	# 			'parietal': '#A15E49', 
	# 			'temporal': '#4E3822',
	# 			'occipital': '#2F1B25'}

	# Easier to tell apart
	color_dict = {'frontal': '#1b9e77', 
				'central': '#d95f02', 
				'parietal': '#7570b3', 
				'temporal': '#e7298a',
				'occipital': '#66a61e'}
	return rois, color_dict

def roi_cross_prediction(data_dir, subject_list, test_feature, train_feature,
			 test_cond, train_cond='AV', delay_min=0.0, 
			 delay_max=0.6, wt_pad=0.0, fs=128., av_sigs=False):
	'''
	Parameters:
	----------
	data_dir : string
		- path where data exist
	subject_list : list
		- list of strings containing subject IDs
	test_cond : 'string
		- either A or V which is the unimodal stim used as the test set
	train_cond : 'string
		- default is AV where weights from the multimodal stimuli will be used 
	test_feature : string
		- either pitchenvsphnfeat or scene_cut_gabor as features to use for A and V, respectively. Must be same as train_feature
	train_feature : string
		- either pitchenvsphnfeat or scene_cut_gabor as features to use for A and V, respectively. Must be same as test_feature
	delay_min : float
		- default of 0.0 based on STRF time lag fit previously
	delay_max : float
		- default of 0.6 based on STRF time lag fit previously
	wt_pad : float
		- default of 0.0 based on STRF wt_pad used to previously fit model
	fs : float
		- sampling rate of EEG, default for 128.0 Hz
	av_sigs : bool
		- Default: False
			- Set audio only or visual only (uni-imodal) stimulus correlation threshold of r > 0.05 as significance
			- If True: Set audiovisual (multimodal) stimulus correlation threshold of r > 0.05 as significance
	'''
	fig = plt.figure(figsize=(15,9))


	for idx, s in enumerate(subject_list):
		with h5py.File(f'{data_dir}/{s}/{s}_STRF_by_{train_feature}_{train_cond}.hf5', 'r') as fh:
			wts_train = fh['wts_mt'][:]
		
		if train_feature == 'pitchenvsphnfeatgabor10pc_sc':
			nfeats=27
			wts2 = wts_train.reshape(np.int(wts_train.shape[0]/nfeats),nfeats,wts_train.shape[1] )
			if test_feature == 'pitchenvsphnfeat':
				aud_feats = wts2[:,:16,:]
				wts_train = aud_feats.reshape((aud_feats.shape[0]*aud_feats.shape[1], aud_feats.shape[2]))
			elif test_feature == 'scene_cut_gabor':
				vis_feats = wts2[:,16:,:]
				wts_train = vis_feats.reshape((vis_feats.shape[0]*vis_feats.shape[1], vis_feats.shape[2]))
			else:
				print('test_feature not defined')
		else:
			print('Not using pitchenvsphnfeatgabor10pc_sc full AV pretained weights ')


		with h5py.File(f'{data_dir}/{s}/{s}_STRF_by_{test_feature}_{test_cond}.hf5', 'r') as fh:
			corrs_test = fh['corrs_mt'][:]	#correlation values from A-only corrs with pitchenvsphnfeat
			validation_test = fh['val_inds_mt'][:]
			pval_test = fh['pvals_mt'][0][:]
			# test_nonsig = np.where(corrs_test < 0.05)
			# test_sig = np.where(corrs_test > 0.05)

		print('*************************')
		print('Now processing %s' %(s))
		print('*************************')
		delays = np.arange(np.floor((delay_min-wt_pad)*fs), np.ceil((delay_max+wt_pad)*fs), dtype=int)

		if test_feature == 'pitchenvsphnfeat':
			resp_dict, stim_dict = loadEEGh5(s, data_dir, test_cond, eeg_epochs=True, resp_mean = True,
									binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=False, 
									spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False)
		elif test_feature == 'scene_cut_gabor':
			resp_dict, stim_dict = loadEEGh5(s, data_dir, test_cond, eeg_epochs=True, resp_mean = True,
									binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, 
									spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False)
		else:
			print('test feature not defined')

		all_stimuli = []
		for i in resp_dict.keys():
			x = (i if resp_dict[i] else 'False')
			if x != 'False':
				all_stimuli.append(x)
		vResp = np.hstack([resp_dict[r][0] for r in [all_stimuli[m] for m in validation_test]]).T

		vStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in [all_stimuli[m] for m in validation_test]]))
		vStim = make_delayed(vStim_temp, delays)

		print('******************************')
		print(wts_train.shape)
		print(vStim.shape)
		print(vResp.shape)
		print('******************************')
		test_condition_corrs, train_condition_pred = predict_response(wts_train, vStim, vResp) 

		ds = load_raw_EEG(s, 1, data_dir)
		ds.drop_channels(['vEOG', 'hEOG'])
		chnames = ds.info['ch_names']
		chnames = np.array(chnames)

		rois, color_dict = get_roi()
		
		plots = [(test_condition_corrs[idx], corrs_test[idx], pval_test[idx]) for idx, t in enumerate(ds.info['ch_names']) for i in rois.keys() if t in rois[i]]
		colors = [color_dict[i] for i in rois.keys() for t in ds.info['ch_names'] if t in rois[i]]
		for test, train, pval in plots:
			if av_sigs:
				title_name = 'av significance, corr > 0.05'
				if train > 0.05: #plotting av significance
					print(test, train)
					plt.plot(train, test, '.', color=colors.pop(0))
				else:
					print('test < 0.05, plotting opaque color')	
					plt.plot(train, test, '.', color='#DCDBDB', alpha=0.7)		
			else:
				title_name = f'unimodal {test_cond} significance, corr > 0.05'
				#if (test > 0.05) or (train > 0.05): #plotting unimodal significance
				#if True: # (pval < 0.05):
				print(test, train)
				plt.plot(train, test, '.', color=colors.pop(0))
				# else:
				# 	print('test < 0.05, plotting opaque color')	
				# 	plt.plot(train, test, '.', color='grey', alpha=0.4)		
		handles = [plt.Line2D([], [], marker='.', color=color, linestyle='None') for color in color_dict.values()]
		labels = list(color_dict.keys())
		plt.legend(handles, labels)

		plt.xlabel('Predict %s from %s' %(test_cond, test_cond))
		plt.ylabel('Predict %s from %s' %(test_cond, train_cond))
		plt.axis('square')
		plt.axvline()
		plt.axhline()
		plt.title(f'{title_name}')

	[slope, intercept, r_value, p_value, std_err] = scipy.stats.linregress(np.hstack(corrs)[0], np.hstack(corrs)[1])
	if test_feature == 'pitchenvsphnfeat':
		plt.plot([-0.05, 0.125], [-0.05, 0.125], 'black', label='unity')
	
	elif test_feature == 'scene_cut_gabor':
		plt.plot([-0.05, 0.35], [-0.05, 0.35], 'black', label='unity')
		plt.gca().set_xticks([-0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
		plt.gca().set_yticks([-0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])

	#plot regression line
	

	plt.savefig(f'{save_dir}/{test_feature}_xpred.pdf')

def channel_cond_corr(subject_list, data_dir):

	subj_frame = dict()
	subj_frame['ch_name'] = []
	subj_frame['subj'] = []
	subj_frame['av_v'] = []
	subj_frame['av_a'] = []

	picks = np.arange(64)
	raw = mne.io.read_raw_fif(f'{data_dir}/MT0028/downsampled_128/MT0028_B1_postICA_rejected.fif')
	ch_names = raw.info['ch_names'][:64]
	rois, color_dict = get_roi()

	for subject in subject_list:
		with h5py.File(f'{data_dir}/full_AV_matrix.hf5', 'r') as f:
			for i in f['AV'].keys():
				print(i)
				print('*******************')
				visual_trailer = i.strip('.wav')+'_visualonly_notif.wav'
				print(visual_trailer)
				try:
					print(f['V'].keys())
					ep_v = f[f'V/{visual_trailer}/resp/{subject}/epochs'][:].mean(0)
					ep_a = f[f'A/{i}/resp/{subject}/epochs'][:].mean(0)
					ep_av = f[f'AV/{i}/resp/{subject}/epochs'][:].mean(0)
					print(ep_v.shape)
					print(ep_a.shape)
					print(ep_av.shape)

					if ep_av.shape[1] != ep_a.shape[1] or ep_av.shape[1] != ep_v.shape[1]:
						ep_a = scipy.signal.resample(ep_a, ep_av.shape[1], axis=1)
						ep_v = scipy.signal.resample(ep_v, ep_av.shape[1], axis=1)
						ep_av = f[f'AV/{i}/resp/{subject}/epochs'][:].mean(0)
						print('resampling size')

					else:
						print('not resampling size')
						visual_trailer = i.strip('.wav')+'_visualonly_notif.wav'
						ep_a = f[f'A/{i}/resp/{subject}/epochs'][:].mean(0)
						ep_v = f[f'A/{visual_trailer}/resp/{subject}/epochs'][:].mean(0)
						ep_av = f[f'AV/{i}/resp/{subject}/epochs'][:].mean(0)

					for elec in picks:
						subj_frame['ch_name'].append(ch_names[elec])
						av_a_corr_notnorm = np.corrcoef(ep_a[elec], ep_av[elec])[0,1]
						av_v_corr_notnorm = np.corrcoef(ep_v[elec], ep_av[elec])[0,1]					
						subj_frame['subj'].append(subject)

						subj_frame['av_v'].append(av_v_corr_notnorm) 
						subj_frame['av_a'].append(av_a_corr_notnorm)  

					grand_max = np.max([np.array(subj_frame['av_v']).max(), np.array(subj_frame['av_a']).max()])# LH: max across both
					print('*******************')
					print('printing grand max: ')
					print(grand_max)
					print('*******************')

					#These are now both normalized in the same way
					# rather than by different numbers depending on the condition
					subj_frame['av_v'] = [x/grand_max for x in subj_frame['av_v']]
					subj_frame['av_a'] = [x/grand_max for x in subj_frame['av_a']]

				except:
					print(f'{subject} does not contain stimulus info')

	corr_info = pd.DataFrame(data=subj_frame)
	corr_info['roi_color'] = [color_dict[i] for i in rois.keys() for t in corr_info['ch_name'] if t in rois[i]] #get colors
	corr_info['roi'] = [next((key for key, values in rois.items() if ch_name in values), None) for ch_name in corr_info['ch_name']] #get roi

	print(len(corr_info))

	#then draw unity line
	plt.figure()
	sns.scatterplot(data=corr_info, x="av_v", y="av_a", hue="roi")
	plt.plot([-0.3, 1.0], [-0.3, 1.0], 'black', label='unity')
	plt.axis('square')

def covariance_feature_mat(data_dir, save_dir, subject='MT0028'):
	'''
	Correlation matrix plot to identify how correlate all of the auditory features are to one another 
	Function also creates a scatter plot to compare the correlations between auditory features for both TIMIT and movie trailers 

	Parameters
	----------
	stimulus_class : string 
		- 'TIMIT' or 'MovieTrailers'
	data_dir : string 
		- path to data
	save_dir : string 
		- path to save figures 
	subject : string 
		(default : MT0002) - can be any participant because stimulus dictionary is the same across all subjects 

	Returns
	-------
	1) Confusion matrix of stimulus feature correlations 
	2) Scatter plot comparing correlation between auditory features for both stimulus sets 

	'''

	#load all auditory features from stimulus played 
	resp_dict, stim_dict = loadEEGh5(subject, data_dir, 'AV', eeg_epochs=True, resp_mean = True,
									binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=True, 
									spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False, filename='full_AV_matrix')


	test_set =['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav']

	all_stimuli = [k for k in (stim_dict).keys() if len(resp_dict[k]) > 0]
	training_set = np.setdiff1d(all_stimuli, test_set)

		
	val_inds = np.zeros((len(all_stimuli),), dtype=np.bool)
	train_inds = np.zeros((len(all_stimuli),), dtype=np.bool)
	for i in np.arange(len(all_stimuli)):
		if all_stimuli[i] in test_set:
			print(all_stimuli[i])
			val_inds[i] = True
		else:
			train_inds[i] = True

	print("Total number of training sentences:")
	print(sum(train_inds))
	print("Total number of validation sentences:")
	print(sum(val_inds))

	training_set = np.setdiff1d(all_stimuli, test_set)

	tStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in training_set]))
	tStim = make_delayed(tStim_temp, [0])

	print(tStim.shape)

	feat_labels = ['sonorant','obstruent','voiced','back','front','low','high','dorsal',
			   'coronal','labial','syllabic','plosive','fricative','nasal', 'envs', 'pitch'] + ['SC'] + np.arange(10).tolist() 

	correlate_mat = np.corrcoef(tStim.T)
	plt.imshow(correlate_mat, aspect='auto', cmap=cm.RdBu_r, vmin=-abs(correlate_mat).max(), vmax=abs(correlate_mat).max())
	plt.gca().set_xticks(np.arange(correlate_mat.shape[0]))
	plt.gca().set_xticklabels(feat_labels, rotation=90)

	plt.gca().set_yticks(np.arange(correlate_mat.shape[0]))
	plt.gca().set_yticklabels(feat_labels)
	plt.colorbar()
	plt.title('Feature Covariances' )
	plt.savefig('%s/feature_covarMatrix.pdf' %(save_dir))
	plt.show()

	return correlate_mat

	resp_dict, stim_dict = loadEEGh5(subject, data_dir, 'AV', eeg_epochs=True, resp_mean = True,
									binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=True, 
									spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False, filename='full_AV_matrix')

def face_congruency(data_dir):
	zeros = []
	congruent_timings = []
	incongruent_timings = []



	all_timings = []
	with h5py.File(f'{data_dir}/full_AV_matrix.hf5', 'r') as f:
		for i in list(f['AV']):
			x=f[f'AV/{i}/stim/phn_feat_timings'][:]
			#print(x.shape)
			#phn_timings.append(np.where(x==1)[1].shape[0])
			all_timings.append(x.shape[1])

			y = f[f'AV/{i}/stim/mouthing'][:]
			#print(y.shape)
			congruent_timings.append(np.where(y[:,0] ==1)[0].shape[0])
			incongruent_timings.append(np.where(y[:,1] ==1)[0].shape[0])
			#zeros.append(np.intersect1d(np.where(y[:,0] ==0), np.where(y[:,1] ==0)))
			zeros.append(np.intersect1d(np.where(y[:,0] ==0), np.where(y[:,1] ==0)).shape[0])



			#phn_congruent.append(np.intersect1d(np.where(x==1)[1], np.where(y[:,0] ==1)[0]).shape[0])
			#phn_incongruent.append(np.intersect1d(np.where(x==1)[1], np.where(y[:,1] ==1)[0]).shape[0])


	
	total_samples = np.sum(all_timings)

	print(f'The number of samples for all of the 8 trailers presented is: {total_samples}')
	print(f'The number of samples for NO speech instances are: {np.sum(zeros)}')
	print(f'The number of samples for CONGRUENT face instances is: {np.sum(congruent_timings)}')
	print(f'The number of samples for INCONGRUENT face instances is: {np.sum(incongruent_timings)}')
	print('********************************')
	print(f'The percentage of NO Speech instances: {(np.sum(zeros)/total_samples)*100}')
	print(f'The percentage of congruent instances: {(np.sum(congruent_timings)/total_samples)*100}')
	print(f'The percentage of incongruent instances: {(np.sum(incongruent_timings)/total_samples)*100}')


def phnfeat_av_a(subject_list, data_dir):

	# elecs_dict = {'visual': ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'], 'auditory': ['AFz', 'Cz','FCz','CPz','C1','C2','FC1','FC2']}
	elecs_dict = {'visual': ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'], 
				'auditory': ['AFz', 'Cz','FCz','CPz','C1','C2', 'C6', 'FC1','FC2', 'FC3', 'FC5', 'FC4', 'CP5', 'CP3', 'CP4', 'CP6', 'Fz']}
	raw = load_raw_EEG(subject_list[0], 1, data_dir)
	chnames = raw.info['ch_names']
	vis_idx = [i for i, item in enumerate(chnames) if item in elecs_dict['visual']]
	aud_idx = [i for i, item in enumerate(chnames) if item in elecs_dict['auditory']]

	corrs_av = []
	corrs_v = []

	visual_elecs_av = [] 
	auditory_elecs_av = []

	visual_elecs_v  = []
	auditory_elecs_v= []

	for idx, i in enumerate(subject_list):
		with h5py.File(f'{data_dir}/{i}/{i}_STRF_by_phnfeat_A.hf5', 'r') as fh:
			with h5py.File(f'{data_dir}/{i}/{i}_STRF_by_phnfeat_AV.hf5', 'r') as h:
				vis = fh['corrs_mt'][:] #visual only
				av = h['corrs_mt'][:] #AV condition


				v_v = vis[vis_idx] #visual only visual elecs
				v_a = vis[aud_idx] #visual only auditory elecs

				av_v = av[vis_idx] #AV visual elecs
				av_a = av[aud_idx] #AV auditory elecs

		#all corrs
		corrs_v.append(vis)
		corrs_av.append(av)

		#AV condition
		visual_elecs_av.append(av_v) 
		auditory_elecs_av.append(av_a)

		#Visual only condition
		visual_elecs_v.append(v_v)
		auditory_elecs_v.append(v_a)

	plt.figure()
	for idx, m in enumerate(subject_list):
		plt.plot(corrs_av[idx], corrs_v[idx], '.', color='#808080', alpha=0.3)

		plt.plot(visual_elecs_av[idx], visual_elecs_v[idx], '.', color='purple', alpha=0.7) #AV v V condition visual elecs
		plt.plot(auditory_elecs_av[idx], auditory_elecs_v[idx], '.', color='green', alpha=0.7) #AV v V condition auditory elecs

	plt.plot([-0.04, 0.13], [-0.04, 0.13], 'black', label='unity')
	plt.xlabel('audiovisual phnfeat (r)')
	plt.ylabel('audio phnfeat (r)')
	plt.title('Phnfeat: AV vs. V')
	plt.legend(['all', 'visual_elecs','aud_elecs'])


def scatters(subject_list, user, data_dir, x, y, condition1, condition2,  annotate_plot=False):
	'''
	Create scatter plots for specified conditions conditions only using two contrasting feature sub-spaces
	e.g. phnfeat vs. mouthing_phnfeat (x vs. y)
	e.g. AV vs. AV or AV vs. A or A vs. V, etc. 


	Inputs x and y must be strings which are the same name as how the STRF files are saved
	Function will also compute two-tailed Wilcoxon sign_rank test
	'''
	strf1_common = []
	strf2_common = []

	# strf1_diff = []
	# strf2_diff = []

	#for all corrs
	strf1 = []
	strf2 = []

	data = []

	for idx, i in enumerate(subject_list):
		with h5py.File(f'{data_dir}/{i}/{i}_STRF_by_{x}_{condition1}.hf5', 'r') as fh:
			corrs1 = fh['corrs_mt'][:]
			corrs1_pvals = fh['pvals_mt'][:]
			c=np.where(corrs1_pvals < 0.05)[1]
			strf1_common.append(corrs1[c])
			strf1.append(corrs1)
			for corr_value in corrs1:
				data.append({'subject': i, 'condition': condition1, 'correlation': corr_value})

		with h5py.File(f'{data_dir}/{i}/{i}_STRF_by_{y}_{condition2}.hf5', 'r') as h:
			corrs2 = h['corrs_mt'][:]
			corrs2_pvals = h['pvals_mt'][:]
			
			d=np.where(corrs2_pvals < 0.05)[1]	
			strf2_common.append(corrs2[d])
			strf2.append(corrs2)
			for corr_value in corrs2:
				data.append({'subject': i, 'condition': condition2, 'correlation': corr_value})

	df = pd.DataFrame(data)
	df.to_csv(f'{x}_{y}_{condition1}_{condition2}_correlations_data.csv', index=False)

	print(f'processing stats for {condition1} {x} vs. {condition2} {y}: ')
	res = wilcoxon(np.concatenate(strf1_common), np.concatenate(strf2_common), alternative='two-sided')
	print(res.statistic)
	print(res.pvalue) 

	plt.figure()
	for idx, m in enumerate(subject_list):
		plt.plot(strf1_common[idx], strf2_common[idx], '.', color='red', alpha=0.7)
		#plt.plot(strf1_diff[idx], strf2_diff[idx], '.', color='grey', alpha=0.5)

	x_coord = input('Enter x-coordinate value: ')
	x_coord = float(x_coord)

	y_coord = input('Enter y-coordinate value: ')
	y_coord = float(y_coord)

	plt.plot([x_coord, y_coord], [x_coord, y_coord], 'black', label='unity', linestyle='dotted')
	plt.xlabel(f'{condition1} {x} (r)')
	plt.ylabel(f'{condition2} {y} (r)')
	plt.title(f'{x} vs. {y}')
	plt.axis('square')

	if annotate_plot:
		plt.annotate(f'W={res.statistic}', xy=(0.0, 0.1),  xycoords='data',
				xytext=(0.10, 0.00))

		plt.annotate(f'p={res.pvalue}', xy=(0.0, 0.1),  xycoords='data',
				xytext=(0.10, -0.01))	

	save_fig=f'/Users/{user}/Box/Desai_Conference_and_Journal_Submissions/interspeech-2023_conference/'
	plt.savefig(f'{save_fig}/python_figs/{x}_{y}_{condition1}_vs_{condition2}.pdf')

	return strf1, strf2

def ica_noica_scatter(data_dir, subject_list, condition, feature):

	rois, color_dict = get_roi()

	ds = load_raw_EEG('MT0028', 1, data_dir)
	chnames = ds.info['ch_names']
	chnames = np.array(chnames)
	
	roi_correlations_strf1 = {roi: [] for roi in rois}
	roi_correlations_strf2 = {roi: [] for roi in rois}

	ica=[]
	no_ica = []

	data = []

	roi_correlations_strf1 = {roi: [] for roi in rois}
	roi_correlations_strf2 = {roi: [] for roi in rois}
	# Iterate through each subject
	for subject in subject_list:
		# Read the banded correlation data
		with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_{feature}_{condition}.hf5', 'r') as f:
			corrs1 = f['corrs_mt'][:][:64]
			ica.append(corrs1)

		# Read the ridge correlation data
		with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_{feature}_{condition}_noICA.hf5', 'r') as f:
			corrs2 = f['corrs_mt'][:][:64]
			no_ica.append(corrs2)

		# Collect correlations for each subject and condition
		for corrs, roi_correlations, corr_type in zip(
			[corrs1, corrs2],
			[roi_correlations_strf1, roi_correlations_strf2],
			['ICA', 'no_ICA']
		):
			for channel, corr_value in zip(chnames, corrs):
				for roi, channels in rois.items():
					if channel in channels:
						roi_correlations[roi].append(corr_value)
						data.append({'subject': subject, 'condition': condition, 'roi': roi, 'correlation': corr_value, 'corr_type': corr_type})
						break

	# Create DataFrame
	df = pd.DataFrame(data)

	# Show the first few rows of the DataFrame
	print(df.head())
	df.to_csv(f'ICA-noICA_{feature}_{condition}.csv')

	plt.figure()
	for idx,i in enumerate(subject_list):
		plt.plot(ica[idx], no_ica[idx], '.', label = i)
	plt.title('Correlation Values for ROI')
	plt.xlabel('Correlation Values (with ICA)')
	plt.ylabel('Correlation Values (without ICA)')
	#plt.legend(bbox_to_anchor=(1, 1))
	plt.title(f'{feature} for {condition}')
	#plt.grid(True)
	plt.plot([-0.05, 0.25], [-0.05, 0.25], 'black', label='unity', linestyle='dotted')
	plt.axis('square')

	# Print the correlations for each ROI for STRF1
	for roi, correlations in roi_correlations_strf1.items():
		print(f'{roi} for STRF1: {correlations}')

	# Print the correlations for each ROI for STRF2
	for roi, correlations in roi_correlations_strf2.items():
		print(f'{roi} for STRF2: {correlations}')
		
	roi_colors = {'frontal': 'red', 'central': 'blue', 'parietal': 'green', 'temporal': 'orange', 'occipital': 'purple'}

	# Plotting
	plt.figure(figsize=(8, 6))
	for roi in roi_correlations_strf1.keys():
		x_values = roi_correlations_strf1[roi]
		y_values = roi_correlations_strf2[roi]
		print(x_values)
		print(y_values)

		plt.scatter(x_values, y_values, label=roi, color=roi_colors[roi])

	plt.title('Correlation Values for ROI')
	plt.xlabel('Correlation Values (with ICA)')
	plt.ylabel('Correlation Values (without ICA)')
	plt.legend()
	plt.title(f'{feature} for {condition}')
	#plt.grid(True)
	plt.plot([0.0, 0.25], [0.0, 0.25], 'black', label='unity', linestyle='dotted')
	plt.axis('square')

def scatter_banded_ridge_lmer(data_dir, subject_list, strf_output, condition):
	'''
	strf_output : string
		- scene_cut_gabor
		- phnfeat_gabor
		- phnfeat_envs
		- phnfeat_pitch
	'''
	rois, color_dict = get_roi()

	ds=load_raw_EEG('MT0028', 1, data_dir, file='postICA_rejected')
	chnames = ds.info['ch_names']
	chnames = np.array(chnames)

	data = []
	
	banded_corr_vals = {roi: [] for roi in rois}
	og_ridge_corr_vals = {roi: [] for roi in rois}
	# Iterate through each subject
	for subject in subject_list:
		# Read the banded correlation data
		with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_{strf_output}_{condition}_banded.hf5', 'r') as f:
			corrs_banded = f['corrs_mt_vratios'][:]
			banded_alphas = f['valphas_mt_ratios']
			ratios = f['vratios_mt']

		# Read the ridge correlation data
		with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_{strf_output}_{condition}.hf5', 'r') as f:
			corrs_ridge = f['corrs_mt'][:][:64]
			alphas = f['valphas_mt']

		# Collect correlations for each subject and condition
		for corrs, roi_correlations, corr_type in zip(
			[corrs_banded, corrs_ridge],
			[banded_corr_vals, og_ridge_corr_vals],
			['banded_corrs', 'ridge_corrs']
		):
			for channel, corr_value in zip(chnames, corrs):
				for roi, channels in rois.items():
					if channel in channels:
						roi_correlations[roi].append(corr_value)
						data.append({'subject': subject, 'condition': condition, 'roi': roi, 'correlation': corr_value, 'corr_type': corr_type})
						break

	# Create DataFrame
	df = pd.DataFrame(data)

	# Show the first few rows of the DataFrame
	print(df.head())

	# Plotting
	plt.figure(figsize=(8, 6))

	corr_ridge = []
	corrs_banded = []
	for roi in banded_corr_vals.keys():
		x_values = banded_corr_vals[roi]
		y_values = og_ridge_corr_vals[roi]
		corr_ridge.append(x_values)
		corrs_banded.append(y_values)
		plt.scatter(x_values, y_values, label=roi, color=color_dict[roi])
			

	res = wilcoxon(np.concatenate(corr_ridge), np.concatenate(corrs_banded), alternative='greater')
	print(res)

	plt.xlabel('Ridge regression (r)')
	plt.ylabel('Banded Ridge regression (r)')
	plt.title(f'{strf_output} : {condition}')
	
	plt.plot([0.0, 0.25], [0.0, 0.25], 'black', label='unity', linestyle='dotted')
	plt.legend()
	plt.axis('square')

	df.to_csv(f'banded_ridge_ridge_{strf_output}_{condition}.csv')


def roi_comparison_condition(subject_list, data_dir, strf_output, condition):

	'''
	strf_output : string
		- scene_cut_gabor
		- phnfeat_gabor
		- phnfeat_envs
		- phnfeat_pitch
	'''
	rois, color_dict = get_roi()

	ds=load_raw_EEG('MT0028', 1, data_dir, file='postICA_rejected')
	chnames = ds.info['ch_names']
	chnames = np.array(chnames)
	
	av_corr_vals = {roi: [] for roi in rois}
	v_corr_vals = {roi: [] for roi in rois}

	for subject in subject_list:
		with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_{strf_output}_AV.hf5', 'r') as f:
			av_corrs = f['corrs_mt'][:][:64]
			print(av_corrs.shape)

			#print(f'The banded ridge regression alpha is {banded_alphas} and the best ratio is {ratios}')
		with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_{strf_output}_{condition}.hf5', 'r') as f:
			v_corrs = f['corrs_mt'][:][:64]

		for corrs, roi_correlations in zip([av_corrs, v_corrs], [av_corr_vals, v_corr_vals]):
			for channel, corr_value in zip(chnames, corrs):
				for roi, channels in rois.items():
					if channel in channels:
						roi_correlations[roi].append(corr_value)
						break

	for i in av_corr_vals.keys():
		print(i)
		res = wilcoxon(av_corr_vals[i], v_corr_vals[i], alternative='greater')
		print(len(av_corr_vals[i]), len(v_corr_vals[i]))
		print(res)

def different_condition_topo(data_dir, subject_list, strf_output, condition):
	
	raw=load_raw_EEG('MT0028', 1, data_dir, file='postICA_rejected')
	chnames = raw.info['ch_names']
	chnames = np.array(chnames)[:64]
	

	difference_corrs = []

	for subject in subject_list:
		with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_{strf_output}_AV.hf5', 'r') as f:
			av_corrs = f['corrs_mt'][:][:64]
			print(av_corrs.shape)

		with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_{strf_output}_{condition}_noICA.hf5', 'r') as f:
			v_corrs = f['corrs_mt'][:][:64]

			difference_corrs.append(av_corrs-v_corrs) #get the difference between AV and V for all corrs

	avg_corrs = np.vstack(difference_corrs).mean(0)
	info = mne.create_info(ch_names=list(chnames), sfreq=raw.info['sfreq'], ch_types=64*['eeg'])
	raw2 = mne.io.RawArray(np.zeros((64,10)), info)
	montage = mne.channels.make_standard_montage('standard_1005')
	raw2.set_montage(montage)
	
	fig,ax = plt.subplots(ncols=1)
	im, cm = mne.viz.plot_topomap(avg_corrs, raw2.info, vlim=(avg_corrs.min(),avg_corrs.max()), cmap='RdBu_r')
	plt.savefig(f'/Users/maansidesai/Box/Desai_Conference_and_Journal_Submissions/interspeech-2023_conference/python_figs/{strf_output}_average_difference_corr_AV_{condition}.pdf')

	fig, ax = plt.subplots(1, 1)

	fraction = .05

	fig = plt.figure()
	ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
	cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap='RdBu_r', norm=mpl.colors.Normalize(avg_corrs.min(), avg_corrs.max()))
	


def corrs_topo(corrs, subject_list, data_dir, save_fig, row=6, col=2, individuals=False, average=True):
	model = input('Enter the condition and feature type: ')

	if individuals:
		fig = plt.figure()
		
		fig.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.suptitle(f'{model}')
		for i, s in enumerate(subject_list):
			raw = load_raw_EEG(s, 1, data_dir)
			nchans = raw.info['ch_names']
			
			info = mne.create_info(ch_names=nchans, sfreq=raw.info['sfreq'], ch_types=64*['eeg'])
			raw2 = mne.io.RawArray(np.zeros((64,10)), info)
			montage = mne.channels.make_standard_montage('standard_1005')
			raw2.set_montage(montage)
			plt.subplot(row,col,i+1)

			im,cm = mne.viz.plot_topomap(corrs[i], raw2.info, vmin=corrs[i].min(), vmax=corrs[i].max())
			plt.title(f'{s}',fontsize=8)
			
			cbar_ax = fig.add_axes()
			clb = fig.colorbar(im, cax=cbar_ax)
		plt.savefig(f'{save_fig}/python_figs/{model}_individual.pdf')

		

	if average:
		avg_corrs = np.vstack(corrs).mean(0)
		raw = load_raw_EEG(subject_list[0], 1, data_dir)
		nchans = raw.info['ch_names']
		
		info = mne.create_info(ch_names=nchans, sfreq=raw.info['sfreq'], ch_types=64*['eeg'])
		raw2 = mne.io.RawArray(np.zeros((64,10)), info)
		montage = mne.channels.make_standard_montage('standard_1005')
		raw2.set_montage(montage)
		
		fig,ax = plt.subplots(ncols=1)
		im, cm = mne.viz.plot_topomap(avg_corrs, raw2.info, vmin=avg_corrs.min(), vmax=avg_corrs.max(), cmap=cm.RdBu_r)

		cbar_ax = fig.add_axes()
		clb = fig.colorbar(im, cax=cbar_ax)
	plt.suptitle(f'{model}')
	plt.savefig(f'{save_fig}/python_figs/{model}_average.pdf')
		# fig = plt.figure()
		# ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
		# cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cm.RdBu_r, norm=mpl.colors.Normalize(avg_corrs.min(), avg_corrs.max()))


def unique_correlations(subject_list, data_dir):
	intersect_sig_corrs = []
	for idx, s in enumerate(subject_list):
		with h5py.File(f'{data_dir}/{s}/{s}_STRF_by_pitchenvsphnfeat_AV.hf5', 'r') as h: #full auditory model
			corr_full_av = h['corrs_mt'][:]
			p_val_full_av = h['pvals_mt'][:]
		sig_corrs_full_av = ([np.where (p_val_full_av[0] < 0.05)])
		intersect_sig_corrs.append(list(np.ravel(sig_corrs_full_av)))
		with h5py.File(f'{data_dir}/{s}/{s}_STRF_by_pitchenvsphnfeat_A.hf5', 'r') as h: #full auditory model
			corr_full_a = h['corrs_mt'][:]
			p_val_full_a = h['pvals_mt'][:]

		sig_corrs_full_a = ([np.where (p_val_full_a[0] < 0.05)])
		intersect_sig_corrs.append(list(np.ravel(sig_corrs_full_a)))

	l1=[np.unique(l) for l in intersect_sig_corrs]
	l1=np.concatenate(l1)
	l1=np.sort(l1)
	c=Counter(l1)
	x = [k for k,v in c.items() if v==3]

	return x
	

def errorbars(x, data, color):
	'''
	x = your x axis (vector of x coordinates)
	data = your data value (y)
	'''
	sem_above = data.mean(0) - data.std(0)/np.sqrt(data.shape[0])
	sem_below = data.mean(0) + data.std(0)/np.sqrt(data.shape[0])
	plt.plot(x, data.mean(0), color)
	plt.fill_between(x, sem_below, sem_above, color=color, alpha=0.3)



def scene_cut_erp(data_dir, textgrid_dir, subject_list, block=1, fs=128., uV_scale=1e-6, tmin=-0.3,
	tmax=0.5, frame_rate_sec = 0.04170833333333333, colors=['r', 'b'], plot_grand=True, plot_individual_figs=False):

	all_resp_av = []
	all_resp_v = []

	for subject in subject_list:
		event_file = pd.read_csv(f'{data_dir}/{subject}/audio/{subject}_B{block}_MovieTrailers_events.csv') #timing information of each movie trailer start			
		#event_file = pd.read_csv(f'{data_dir}/{subject}/audio/{subject}_B{block}_MovieTrailers_events_notif_V.csv') #timing information of each movie trailer start			
		
		a = np.where(event_file['condition'] == 'AV')
		evs = event_file.iloc[a][['onset_time', 'offset_time', 'event_id', 'name']].values
		evs[:,:2] = evs[:,:2]*fs

		b = np.where(event_file['condition'] == 'V')
		evs_v = event_file.iloc[b][['onset_time', 'offset_time', 'event_id', 'name']].values
		evs_v[:,:2] = evs_v[:,:2]*fs 
		evs_v[:,:2] = evs_v[:,:2] - (frame_rate_sec *fs) #subtract one frame in 128. Hz sampling rate
		raw = load_raw_EEG(subject, block, data_dir)
		print('****************************')
		print(evs.shape, evs_v.shape)
		print('****************************')
		onsets = []#audiovisual onset/offset info
		offsets = []

		onsets_v = [] #visual ONLY onset/offset info
		offsets_v = []

		for idx, i in enumerate(evs[:,3]): #audiovisual
			tg = tgio.openTextgrid(f'{textgrid_dir}/{i}_corrected_SC.TextGrid')
			sc_tier = tg.tierDict['Scene Cut']
			df = pd.DataFrame([(start, end, label) for (start, end, label) in sc_tier.entryList],columns = ['start','end','label'])
			onsets.append(df['start'].values*fs+evs[idx][0]) #audiovisual
			offsets.append(df['end'].values*fs+evs[idx][1])
		for idx, i in enumerate(evs_v[:,3]): #visual only
			i = i.split('_visualonly_notif')[0]
			print(i)
			tg = tgio.openTextgrid(f'{textgrid_dir}/{i}_corrected_SC.TextGrid')
			sc_tier = tg.tierDict['Scene Cut']
			df = pd.DataFrame([(start, end, label) for (start, end, label) in sc_tier.entryList],columns = ['start','end','label'])
			onsets_v.append(df['start'].values*fs+evs_v[idx][0]) #visual only
			offsets_v.append(df['end'].values*fs+evs_v[idx][1])
		
		ones = np.ones(np.concatenate(offsets).shape[0]) #event ID for audiovisual
		twos = np.ones(np.concatenate(offsets_v).shape[0])*2 #event ID for visual only
	
		sc_events_av = np.vstack((list(np.concatenate(onsets).astype(int)), list(np.concatenate(offsets).astype(int)), list(ones.astype(int)))).T #audiovisual scene cut information
		sc_events_v = np.vstack((list(np.concatenate(onsets_v).astype(int)), list(np.concatenate(offsets_v).astype(int)), list(twos.astype(int)))).T #visual ONLY scene cut information

		epochs_av = mne.Epochs(raw, sc_events_av, event_id=1, baseline=None,reject_by_annotation=True)

		epochs_v = mne.Epochs(raw, sc_events_v, event_id=2, baseline=None, reject_by_annotation=True, event_repeated='merge')

		times = [0.0, 0.05, 0.1, 0.2, 0.3]
		visual_elecs = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']

		data_av = epochs_av.get_data(picks=visual_elecs)
		data_v = epochs_v.get_data(picks=visual_elecs)
		resp_av = data_av.mean(1)/uV_scale  
		resp_v = data_v.mean(1)/uV_scale  
		all_resp_av.append(resp_av.mean(0))
		all_resp_v.append(resp_v.mean(0))


		if plot_individual_figs:

			#plotting topos and scalp ERPs for AV
			evoked_av = epochs_av.average()	
			evoked_av.plot_topomap(times, ch_type='eeg') #plot all electrodes

			#plotting topos and scalp ERPs for Visual ONLY
			evoked_v = epochs_v.average()
			evoked_v.plot_topomap(times, ch_type='eeg') #plot all electrodes

			evoked_av = epochs_av.average(picks=visual_elecs)
			evoked_v = epochs_v.average(picks=visual_elecs)

			mne.viz.plot_compare_evokeds(dict(av=evoked_av, visual=evoked_v),
									legend='upper left', show_sensors='upper right')

	if plot_grand:
		t = np.linspace(tmin,tmax,num=np.array(all_resp_av).shape[1])
		plt.figure()
		errorbars(t, np.array(all_resp_av), colors[0])
		errorbars(t, np.array(all_resp_v), colors[1])
		plt.axvline(0,color='k')
		plt.axhline(0,color='k')
		plt.legend(['av','v'])
		plt.xlabel('Time [s]')
		plt.ylabel('Amplitude [uV]')
	
	return t, all_resp_av, all_resp_v	

def individual_sc_subplot(subject_list, t, all_resp_av, all_resp_v, row=3, col=4):
	fig = plt.figure(figsize=(10,10))
	fig.subplots_adjust(wspace=0.3, hspace=0.4)
	for i, s in enumerate(subject_list):
		plt.subplot(row,col,i+1)
		#y = np.squeeze(all_resp_av[idx])/1e-6
		plt.plot(t, all_resp_av[i], color='red', alpha=0.5)
		plt.plot(t, all_resp_v[i], color='blue', alpha=0.5)
		plt.axvline(0,color='k')
		plt.axhline(0,color='k')
		plt.title(f'{s}',fontsize=12)

		if i == 8:
			plt.xlabel('Time (s)',fontsize=10)
			plt.ylabel('µV',fontsize=10)

	plt.legend(['av','v'], bbox_to_anchor=(1, 1), loc='upper left')

def round_up(n, decimals=0):
	multiplier = 10 ** decimals
	return math.ceil(n * multiplier) / multiplier


def plot_wts(data_dir, subject, condition, strf_output, subplot_r=4, subplot_c=2, fs=128.0, delay_min=0.0, delay_max=0.6, grand_rois=False, avg_rois=True):
	
	feat_labels = ['sonorant','obstruent','voiced','back','front','low','high','dorsal',
			'coronal','labial','syllabic','plosive','fricative','nasal', 'envs']

	elecs_dict = {'visual': ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'], 'auditory': ['AFz', 'Cz','FCz','CPz','C1','C2','FC1','FC2']}
	nfeats = len(feat_labels)

	raw = load_raw_EEG(subject, 1, data_dir)
	chnames = raw.info['ch_names']
	with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_{strf_output}_{condition}_noICA.hf5','r') as hf:
		wts = hf['wts_mt'][:]
		corrs = hf['/corrs_mt'][:]
		pval = hf['pvals_mt'][:]
		# sig_corr_idx = np.where(pval[0]<0.05)
		# sig_corrs = corrs[np.where(pval[0]<0.05)]
		vis_wt_idx = [i for i, item in enumerate(chnames) if item in elecs_dict['visual']]
		aud_wt_idx = [i for i, item in enumerate(chnames) if item in elecs_dict['auditory']]

		#get all ROI weights in matrix
		vis_wts = wts[:,vis_wt_idx] 
		aud_wts = wts[:,aud_wt_idx]

	wts2 = wts.reshape(np.int(wts.shape[0]/nfeats),nfeats,wts.shape[1] )
	print(wts2.shape)
	print(wts.shape)
	print(nfeats)
	delays = np.arange(np.floor((delay_min)*fs), np.ceil((delay_max)*fs), dtype=np.int) #create array to pass time delays in
	print("Delays:", delays)
	t = np.linspace(delay_min, delay_max, len(delays))
	# Check whether the output path to save strfs exists or not
	output_dir = f'/{data_dir}/{subject}/figures'
	isExist = os.path.exists(output_dir)

	if not isExist:
		os.makedirs(output_dir)
		print("The new directory is created!")

	if grand_rois:
		for i in elecs_dict:
			fig = plt.figure(figsize=(10,12))
			for m in range(len((elecs_dict[i]))):
				plt.subplot(subplot_r, subplot_c, m+1)
				strf = wts2[:,:,chnames.index(elecs_dict[i][m])].T
				smax = np.abs(strf).max()
				
				plt.imshow(strf, cmap='RdBu_r', aspect='auto', interpolation='nearest', vmin=-smax, vmax=smax)
				plt.gca().set_xticks([0, (len(delays)-1)/2, len(delays)-1])
				plt.gca().set_xticklabels([t[0], round_up(t[np.int((len(delays)-1)/2)],2), t[len(delays)-1]])
				plt.gca().set_yticks(np.arange(strf.shape[0]))
				plt.gca().set_yticklabels(feat_labels)
				channel_names = str(elecs_dict[i][m])
				plt.title('%s r=%.3g'%(elecs_dict[i][m], corrs[chnames.index(elecs_dict[i][m])]))
				plt.colorbar()
				plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
			plt.tight_layout()
			plt.savefig(f'{output_dir}/{subject}_{i}_chs_wts.pdf')
		
	if avg_rois:
		auditory_wts = aud_wts.reshape(np.int(aud_wts.shape[0]/nfeats),nfeats,aud_wts.shape[1] )
		print(auditory_wts.shape)
		print(aud_wts.shape)
		print(nfeats)
		delays = np.arange(np.floor((delay_min)*fs), np.ceil((delay_max)*fs), dtype=np.int) #create array to pass time delays in
		print("Delays:", delays)

		visual_wts = vis_wts.reshape(np.int(vis_wts.shape[0]/nfeats),nfeats,vis_wts.shape[1] )
		print(visual_wts.shape)
		print(vis_wts.shape)
		print(nfeats)
		delays = np.arange(np.floor((delay_min)*fs), np.ceil((delay_max)*fs), dtype=np.int) #create array to pass time delays in
		print("Delays:", delays)

		#figs, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey='row')
		plt.figure()
		plt.subplot(1,2,2)
		plt.imshow(auditory_wts.mean(2).T, aspect='auto', interpolation='nearest', vmin=-auditory_wts.mean(2).max(), vmax=auditory_wts.mean(2).max(), cmap='RdBu_r')
		plt.gca().set_xticks([0, (len(delays)-1)/2, len(delays)-1])
		plt.gca().set_xticklabels([t[0], round_up(t[np.int((len(delays)-1)/2)],2), t[len(delays)-1]])
		plt.gca().set_yticks(np.arange(auditory_wts.shape[1]))
		plt.gca().set_yticklabels(feat_labels)
		plt.title('Auditory channels')
		plt.colorbar()

		plt.subplot(1,2,1)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
		plt.imshow(visual_wts.mean(2).T, aspect='auto', interpolation='nearest', vmin=-visual_wts.mean(2).max(), vmax=visual_wts.mean(2).max(), cmap='RdBu_r')
		plt.title('Visual channels')
		plt.gca().set_xticks([0, (len(delays)-1)/2, len(delays)-1])
		plt.gca().set_xticklabels([t[0], round_up(t[np.int((len(delays)-1)/2)],2), t[len(delays)-1]])
		plt.gca().set_yticks(np.arange(auditory_wts.shape[1]))
		plt.gca().set_yticklabels(feat_labels)
		plt.xlabel('Time delays [s]')
		plt.colorbar()

		plt.savefig(f'{output_dir}/{subject}_avgROIs_wts.pdf')

def wts_corr(data_dir, save_fig, subject_list, strf_name, fs=128.0, delay_min=0.0, delay_max=0.6):

	if strf_name == 'pitchenvsphnfeat':
		feat_labels = ['sonorant','obstruent','voiced','back','front','low','high','dorsal',
			'coronal','labial','syllabic','plosive','fricative','nasal', 'envs', 'pitch']
		conditions = ['AV', 'A']

	if strf_name == 'scene_cut_gabor':
		feat_labels = list(np.arange(10)) + ['SC']
		conditions = ['AV', 'V']
	#elecs_dict = {'visual': ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'], 'auditory': ['AFz', 'Cz','FCz','CPz','C1','C2','FC1','FC2']}
	nfeats = len(feat_labels)
	print(nfeats)

	raw = load_raw_EEG(subject_list[0], 1, data_dir)
	nchans = raw.info['ch_names'][:64]

	delays = np.arange(np.floor((delay_min)*fs), np.ceil((delay_max)*fs), dtype=np.int) #create array to pass time delays in
	print("Delays:", delays)
	ndelays = len(delays)


	
	for c in conditions:
		if c == 'AV':
			av_mat = np.zeros((nfeats*ndelays, len(nchans), len(subject_list)))
		elif c == conditions[1]:
			uni_mat = np.zeros((nfeats*ndelays, len(nchans), len(subject_list)))
		else:
			print('undefined')

		for idx, s in enumerate(subject_list):
			with h5py.File(f'%s/%s/%s_STRF_by_{strf_name}_%s.hf5'%(data_dir, s, s, c), 'r') as fh: #full model
				if c == 'AV':
					wts = fh['wts_mt'][:]
					wts = wts[:,:64]
				else:
					wts = fh['wts_mt'][:]
				print(wts.shape)
		

				if c == 'AV':
					av_mat[:,:, idx]=wts
				elif c == conditions[1]:
					uni_mat[:,:, idx]=wts
				else:
					print('Undefined input')

	wts_corrs_list = np.zeros((len(nchans), len(subject_list)))
	for ii, t in enumerate(subject_list):
		for channel in np.arange(64):
			audiovisual = av_mat[:,channel,ii] #average audiovisual wts across channels

			unimodal = uni_mat[:,channel,ii] #average visual only wts across channels
			x = np.corrcoef(audiovisual, unimodal)
			wts_corrs_list[channel,ii] = x[0,1] #setting every element of corr matrix to channel and subj

	new_order = np.array(nchans).argsort()
	print(wts_corrs_list.argmax())
	y = wts_corrs_list.argmax()
	[time, ch, nsubj] = np.unravel_index(y, (av_mat.shape))
	topo_input = wts_corrs_list.mean(1)

	#plt.figure() 
	info = mne.create_info(ch_names=nchans, sfreq=raw.info['sfreq'], ch_types=64*['eeg'])
	raw2 = mne.io.RawArray(np.zeros((64,10)), info)
	montage = mne.channels.make_standard_montage('standard_1005')
	raw2.set_montage(montage)
	mne.viz.plot_topomap(topo_input, raw2.info, vlim=(topo_input.min(), topo_input.max()), cmap='Reds')
	plt.savefig(f'{save_fig}/{strf_name}_wts.pdf')



	fig = plt.figure()
	ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
	cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap='Reds', norm=mpl.colors.Normalize(topo_input.min(), topo_input.max()))
	plt.savefig(f'{save_fig}/{strf_name}_wts_colorbar.pdf')


def plot_eog_wts(data_dir, subject, condition, subplot_r=1, subplot_c=2, fs=128.0, delay_min=0.0, delay_max=0.6):

	raw = load_raw_EEG(subject, 1, data_dir, file = 'EOGs')
	chnames = raw.info['ch_names']
	with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_scene_cut_gabor_{condition}_EOG.hf5','r') as hf:
		wts = hf['/wts_mt'][:]
		corrs = hf['/corrs_mt'][:]
		pval = hf['pvals_mt'][:]

	wts2 = wts.reshape(np.int(wts.shape[0]/nfeats),nfeats,wts.shape[1] )
	print(wts2.shape)
	print(wts.shape)
	nfeats = wts2.shape[1]
	print(nfeats)
	delays = np.arange(np.floor((delay_min)*fs), np.ceil((delay_max)*fs), dtype=np.int) #create array to pass time delays in
	print("Delays:", delays)
	t = np.linspace(delay_min, delay_max, len(delays))

	feat_labels = list(np.arange(10)) + ['SC']

	# Check whether the output path to save strfs exists or not
	output_dir = f'/{data_dir}/{subject}/figures'
	isExist = os.path.exists(output_dir)

	if not isExist:
		os.makedirs(output_dir)
		print("The new directory is created!")


	for i in chnames:
		fig = plt.figure(figsize=(10,12))
		for m in range(len((chnames))):
			plt.subplot(subplot_r, subplot_c, m+1)
			strf = wts2[:,:,m].T
			smax = np.abs(strf).max()
			
			plt.imshow(strf, cmap=cm.RdBu_r, aspect='auto', interpolation='nearest', vmin=-smax, vmax=smax)
			plt.gca().set_xticks([0, (len(delays)-1)/2, len(delays)-1])
			plt.gca().set_xticklabels([t[0], round_up(t[np.int((len(delays)-1)/2)],2), t[len(delays)-1]])
			plt.gca().set_yticks(np.arange(strf.shape[0]))
			plt.gca().set_yticklabels(feat_labels)
			channel_names = str(chnames[m])
			plt.title('%s r=%.3g'%(chnames[m], corrs[m]))
			plt.colorbar()
			#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
		plt.tight_layout()
	plt.savefig(f'{output_dir}/{subject}_{i}_chs_wts_EOGs_{condition}.pdf')
		

def eog_barplot(subject_list, data_dir, save_fig):

	v_corrs_heog = []
	av_corrs_heog = []

	v_corrs_veog = []
	av_corrs_veog = []	

	for subject in subject_list:
		raw = load_raw_EEG(subject, 1, data_dir, file = 'EOGs')
		chnames = raw.info['ch_names']
		with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_scene_cut_gabor_AV_EOG.hf5','r') as hf:
			corrs = hf['/corrs_mt'][:]
			av_corrs_heog.append(corrs[0])
			av_corrs_veog.append(corrs[1])
		with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_scene_cut_gabor_V_EOG.hf5','r') as hf:
			corrs = hf['/corrs_mt'][:]
			v_corrs_heog.append(corrs[0])
			v_corrs_veog.append(corrs[1])


	# Calculate mean and standard error for each list of correlations
	v_mean_heog = np.mean(v_corrs_heog)
	av_mean_heog = np.mean(av_corrs_heog)
	v_std_heog = np.std(v_corrs_heog)
	av_std_heog = np.std(av_corrs_heog)

	v_mean_veog = np.mean(v_corrs_veog)
	av_mean_veog = np.mean(av_corrs_veog)
	v_std_veog = np.std(v_corrs_veog)
	av_std_veog = np.std(av_corrs_veog)

	# Data for plotting
	labels = ['v_corrs_heog', 'av_corrs_heog', 'v_corrs_veog', 'av_corrs_veog']
	means = [v_mean_heog, av_mean_heog, v_mean_veog, av_mean_veog]
	stds = [v_std_heog, av_std_heog, v_std_veog, av_std_veog]

	# Colors for the bars
	colors = ['tab:blue', 'tab:blue', 'tab:orange', 'tab:orange']

	# Plotting
	x = np.arange(len(labels))
	width = 0.45

	fig, ax = plt.subplots()
	rects1 = ax.bar(x - width/2, means, width, label='Mean', yerr=stds, capsize=5, color=colors)
	plt.axhline(y=0.0, xmin=0.0, xmax=1.0, color='black')
	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_xlabel('Correlation Lists')
	ax.set_ylabel('Correlation Values')
	ax.set_title('Average Correlation Values with Standard Error')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.legend()
	plt.savefig(f'{save_fig}/EOG_corr-barplots.pdf')



def visualize_wts(subject, data_dir, model, conditions, save_fig, delay_min=0.0, delay_max=0.6, fs=128.):
	for condition in conditions:
		if model == 'scene_cut_gabor':
			feat_labels = list(np.arange(10)) + ['SC']
		if model == 'pitchenvsphnfeat':
			feat_labels=['sonorant','obstruent','voiced','back','front','low','high','dorsal',
					'coronal','labial','syllabic','plosive','fricative','nasal', 'envs', 'pitch']

		nfeats = len(feat_labels)

		raw = load_raw_EEG(subject, 1, data_dir)

		chnames = raw.info['ch_names']

		with h5py.File(f'{data_dir}/{subject}/{subject}_STRF_by_{model}_{condition}.hf5','r') as hf:
			wts = hf['/wts_mt'][:]
			corrs = hf['/corrs_mt'][:]

		wts2 = wts.reshape(np.int(wts.shape[0]/nfeats),nfeats,wts.shape[1] ) #reshape weights since they are not from fitting STRFs
		print(wts2.shape)
		print(wts.shape)
		print(nfeats)
		# print(nfeats.shape)

		# wt_pad = 0.0
		delays = np.arange(np.floor((delay_min)*fs), np.ceil((delay_max)*fs), dtype=np.int) #create array to pass time delays in
		print("Delays:", delays)
		fig = plt.figure(figsize=(10,15))
		for m in range((wts2.shape[2])):
			plt.subplot(8, 8, m+1)
			strf = wts2[:,:,m].T

			smax = np.abs(strf).max()
			t = np.linspace(delay_min, delay_max, len(delays))
			plt.imshow(strf, cmap=cm.RdBu_r, aspect='auto', interpolation='nearest', vmin=-smax, vmax=smax)
			plt.gca().set_xticks([0, (len(delays)-1)/2, len(delays)-1])
			plt.gca().set_xticklabels([t[0], round_up(t[np.int((len(delays)-1)/2)],2), t[len(delays)-1]])

			plt.gca().set_yticks(np.arange(nfeats))
			plt.gca().set_yticklabels(feat_labels)

			plt.title('%s r=%.3g'%(chnames[m], corrs[m]), fontsize=7)
			plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.6)

			plt.suptitle(f'{condition} {model}')
			plt.savefig(f'{save_fig}/python_figs/{subject}_{condition}_{model}_all_chs.pdf')

	return corrs, wts, wts2

def mouth_erp(data_dir, textgrid_talker_dir, save_fig, subject_list, elecs, elec_type, condition='AV', 
	block=1, fs=128., tmin=-0.3, tmax=0.5, uV_scale=1e-6, colors = ['g', 'r']):

	congruent_resp = []
	incongruent_resp = []

	all_congruent_epochs = []
	all_incongruent_epochs = []

	for subject in subject_list:
		event_file = pd.read_csv(f'{data_dir}/{subject}/audio/{subject}_B{block}_MovieTrailers_events.csv') #timing information of each movie trailer start			
		a = np.where(event_file['condition'] == condition)
		evs = event_file.iloc[a][['onset_time', 'offset_time', 'event_id', 'name']].values
		evs[:,:2] = evs[:,:2]*fs
		print(evs[:,:2].astype(int))

		raw = load_raw_EEG(subject, block, data_dir)
		yes_onset = []
		yes_offset = []

		no_onset = []
		no_offset = []

		for idx, i in enumerate(evs[:,3]): 
			if condition == 'V':
				i = i.split('_visualonly_notif')[0]
			else:
				print('processing AV')
			tg = tgio.openTextgrid(f'{textgrid_talker_dir}/{i}_corrected_natsounds.TextGrid')
			sc_tier = tg.tierDict['mouthing']
			df = pd.DataFrame([(start, end, label) for (start, end, label) in sc_tier.entryList],columns = ['start','end','label'])
			yes = np.where(df['label'] == 'yes')
			yes_onset.append((df.iloc[yes]['start'].values*fs+evs[idx][0]).astype(int)) #in sampling rate of 128. Hz
			yes_offset.append((df.iloc[yes]['start'].values*fs+evs[idx][1]+1).astype(int)) #add one for offset time

			no = np.where(df['label'] == 'no')
			no_onset.append((df.iloc[no]['start'].values*fs+evs[idx][0]).astype(int)) #in sampling rate of 128. Hz
			no_offset.append((df.iloc[no]['start'].values*fs+evs[idx][1]+1).astype(int))

		ones = np.ones(np.concatenate(yes_onset).shape[0])
		twos = np.ones(np.concatenate(no_onset).shape[0])*2

		congruent = np.vstack((list(np.concatenate(yes_onset).astype(int)), list(np.concatenate(yes_offset).astype(int)), list(ones.astype(int)))).T #congruent mouthing
		incongruent = np.vstack((list(np.concatenate(no_onset).astype(int)), list(np.concatenate(no_offset).astype(int)), list(twos.astype(int)))).T #incongruent mouthing

		epochs_cong = mne.Epochs(raw, congruent, event_id=1,  tmin=tmin, tmax=tmax, baseline=None,reject_by_annotation=True)
		epochs_incong = mne.Epochs(raw, incongruent, event_id=2,  tmin=tmin, tmax=tmax, baseline=None,reject_by_annotation=True)


		data_congru = epochs_cong.get_data(picks=elecs)
		data_incongru = epochs_incong.get_data(picks=elecs)

		all_congruent_epochs.append(epochs_cong)
		all_incongruent_epochs.append(epochs_incong)

		resp_c = data_congru.mean(1)/uV_scale
		resp_ic = data_incongru.mean(1)/uV_scale
		congruent_resp.append(resp_c.mean(0))
		incongruent_resp.append(resp_ic.mean(0))

		# evoked_v = epochs_v.average()
		# evoked_v.plot_topomap(times, ch_type='eeg')

	#plot grand avg. response
	t = np.linspace(tmin,tmax,num=np.array(congruent_resp).shape[1])
	plt.figure()
	errorbars(t, np.array(congruent_resp), colors[0])
	errorbars(t, np.array(incongruent_resp), colors[1])
	plt.axvline(0,color='k')
	plt.axhline(0,color='k')
	plt.legend(['congruent','incongruent'])
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude [uV]')
	plt.savefig(f'{save_fig}/python_figs/{elec_type}_grandAvg_mouthing-ERPs_{condition}.pdf')

	#plot congruent topo maps
	times = [-.3,-.2,-.1,0,.1,.2,.3,.4, .5]
	rng=0.8
	cat_congr_epochs = mne.concatenate_epochs(all_congruent_epochs)
	cat_congr_epochs.average().plot_topomap(times, vmin=-rng,vmax=rng)
	plt.savefig(f'{save_fig}/python_figs/allSubjs-64ch_congruent_topo_{condition}.pdf')

	#plot incongruent topo maps
	cat_incongr_epochs = mne.concatenate_epochs(all_incongruent_epochs)
	cat_incongr_epochs.average().plot_topomap(times, vmin=-rng,vmax=rng)
	plt.savefig(f'{save_fig}/python_figs/allSubjs-64ch_incongruent_topo_{condition}.pdf')

def lmer_corrs_csv(subject_list, data_dir, conditions, model_types):
	'''
	conditions = ['AV', 'A'] and model_types = ['pitchenvsphnfeat']
	conditions = ['AV', 'V'] and model_types = ['scene_cut_gabor']

	'''
	#conditions = ['AV', 'A', 'V']

	#model_types = ['pitchenvsphnfeat', 'scene_cut_gabor']

	all_subjs = []
	all_models = []

	all_corrs = []
	all_conditions = []

	for c in conditions:
		for model in model_types:
			for s in subject_list:
				with h5py.File(f'{data_dir}/{s}/{s}_STRF_by_{model}_{c}.hf5', 'r') as f:
					corrs=f['corrs_mt'][:]
					if corrs.shape[0]>64:
						print(corrs.shape)
						corrs=corrs[:64]
					else:
						print(corrs.shape)
						corrs=corrs
				for ch in np.arange(len(corrs)):
					all_subjs.append(s)
					all_models.append(model)
					all_corrs.append(corrs[ch])
					all_conditions.append(c)
	print(len(all_subjs))
	print(len(all_models))
	print(len(all_corrs))
	print(len(all_conditions))

	data= {'corrs': np.array(all_corrs).ravel(), 'subject': all_subjs, 'STRF_type': all_models, 'condition': all_conditions}
	df = pd.DataFrame.from_dict(data)
	print(df)
	savepath = os.getcwd()
	df.to_csv(f'/{savepath}/{model}_{conditions[0]}_{conditions[1]}.csv')

def conditional_psd(folder, filepath, subject_list, condition):
	'''
	filepath : string
		- where your data lives
	folder : string
		- name of folder you want to read data in
	subject_list : list of strings
		- a list with subjects in them. For exampme ['MT0028', 'MT0029', 'MT0030'...]
	condition : string
		- AV or V or A which stands for audiovisual, visual or audio
	'''
	all_raws = [] #initialize list which is what you will append all of the neural data to

	#loop through your subject list and find the preprocessed data and also get the file to find the event times 
	#(where all of the stimulus information occurred during the neural recording)
	 
	for subject in subject_list:
		#raw = mne.io.read_raw_brainvision(f'{filepath}/{folder}/{subject}/downsampled_128/{subject}_B1_DS128.vhdr', preload=True)
		raw = mne.io.read_raw_fif(f'{filepath}/{folder}/{subject}/downsampled_128/{subject}_B1_postICA_rejected.fif', preload=True)
		event_times = f'{filepath}/{folder}/{subject}/audio'
		raw.drop_channels(['hEOG', 'vEOG'])


		#read event file and identify the onset and offset information 
		event_file = pd.read_csv(f'{event_times}/{subject}_B1_MovieTrailers_events.csv')
		a = np.where(event_file['condition'] == condition)
		print(a)
		evs = event_file.iloc[a][['onset_time', 'offset_time']].values
		#raw_selection = raw.copy().crop(tmin=evs[0][0], tmax=evs[0][1])
		
		#Use the onset and offset information (in seconds!) to crop the neural data and then append info to the all_raws list
		for idx, i in enumerate(evs):
			raw_selection = raw.copy().crop(tmin=i[0], tmax=i[1]) 
			print(raw_selection)
			all_raws.append(raw_selection)

	#now concatenate all_raws so you have one very long neural recording from all the subjects	
	new_raw = mne.concatenate_raws(all_raws)

	return new_raw	

#######################
# if __name__ == "__main__":
# 	user = 'maansidesai'
# 	data_dir = f'/Users/{user}/Box/MovieTrailersTask/Data/EEG/Participants/'
# 	textgrid_dir = f'/Users/{user}/Box/trailer_AV/textgrids/scene_cut_textGrids/AV_task'
# 	textgrid_talker_dir = f'{textgrid_dir}/talker_tg'
# 	save_fig=f'/Users/{user}/Box/Desai_Conference_and_Journal_Submissions/interspeech-2023_conference/'
# 	subject_list = ['MT0028', 'MT0029', 'MT0030','MT0031' ,'MT0032', 'MT0033', 'MT0034', 'MT0035','MT0036', 'MT0037', 'MT0038']




if __name__ == "__main__":
	user = 'maansidesai'

	data_dir = f'/Users/{user}/Box/MovieTrailersTask/Data/EEG/Participants/'
	save_dir = f'/Users/{user}/Box/Desai_Conference_and_Journal_Submissions/interspeech-2023_conference/python_figs'
	#subject_list = ['MT0028', 'MT0029', 'MT0030','MT0031' ,'MT0032', 'MT0033', 'MT0034', 'MT0035','MT0036', 'MT0037', 'MT0038']
	feature_type = 'visual' #visual or auditory
	plt.ion()
	conditions=['AV', 'V']
	for condition in conditions:
		for subject in subject_list:
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, 1, data_dir, condition, 
																		scene_cut_gaborpc=True, 
																		delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0, noICA=True, filename='full_AV_matrix_noICA')

	if feature_type == 'auditory':
		cross_prediction(data_dir, subject_list, test_cond='A', train_cond='AV', test_feature='pitchenvsphnfeat', 
				train_feature='pitchenvsphnfeat', delay_min=0.0, delay_max=0.6, wt_pad=0.0, fs=128., av_sigs=False)

		cross_prediction(data_dir, subject_list, test_cond='A', train_cond='AV', test_feature='pitchenvsphnfeat', 
				train_feature='pitchenvsphnfeatgabor10pc_sc', delay_min=0.0, delay_max=0.6, wt_pad=0.0, fs=128., av_sigs=False)
		roi_cross_prediction(data_dir, subject_list, test_cond='A', train_cond='AV', test_feature='pitchenvsphnfeat', 
			train_feature='pitchenvsphnfeatgabor10pc_sc', delay_min=0.0, delay_max=0.6, wt_pad=0.0, fs=128., av_sigs=False)

	elif feature_type == 'visual':
		cross_prediction(data_dir, subject_list, test_cond='V', train_cond='AV', test_feature='scene_cut_gabor', 
				train_feature='scene_cut_gabor', delay_min=0.0, delay_max=0.6, wt_pad=0.0, fs=128., av_sigs=False)

	else:
		print('Feature must be auditory or visual ')

	# channel_cond_corr(subject_list, data_dir)

	# raw_av = conditional_psd(user, folder, filepath, subject_list, 'AV')
	# raw_a = conditional_psd(user, folder, filepath, subject_list, 'A')

	# print(raw_av)
	# print(raw_a)

	# #create difference psd plot?
	# raw_a.compute_psd().plot_topomap(normalize=True)
	# raw_av.compute_psd().plot_topomap(normalize=True)
	#elecs = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
# 	#elecs = ['AFz', 'Cz','FCz','CPz','C1','C2','FC1','FC2']
	
# 	# x_aud, y_aud = scatters(subject_list, data_dir, 'pitchenvsphnfeat', 'pitchenvsphnfeat', 'AV', 'A', annotate_plot=True) 
# 	# x_vis, y_vis = scatters(subject_list, data_dir, 'scene_cut_gabor', 'scene_cut_gabor', 'AV', 'V', annotate_plot=True)

# 	# corrs_topo(x_aud, subject_list, row=6, col=2, individuals=False, average=True)
# 	# corrs_topo(y_aud, subject_list, row=6, col=2, individuals=False, average=True)

# 	# corrs_topo(x_vis, subject_list, row=6, col=2, individuals=False, average=True)
# 	# corrs_topo(y_vis, subject_list, row=6, col=2, individuals=False, average=True)

# 	#corrs, wts, wts2 = visualize_wts('MT0032', data_dir, 'pitchenvsphnfeat', ['AV', 'A'], save_fig, delay_min=0.0, delay_max=0.6, fs=128.)
# 	#corrs, wts, wts2 = visualize_wts('MT0032', data_dir, 'scene_cut_gabor', ['AV', 'V'], save_fig, delay_min=0.0, delay_max=0.6, fs=128.)


# 	mouth_erp(data_dir, textgrid_talker_dir, save_fig, subject_list, elecs, elec_type='visual', 
# 		condition='V', block=1, fs=128., tmin=-0.3, tmax=0.5, uV_scale=1e-6, colors = ['green', 'red'])