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
def load_raw_EEG(subject, block, datadir):
	eeg_file = '%s/%s/downsampled_128/%s_B%d_postICA_rejected.fif'%(datadir, subject, subject, block)
	raw = mne.io.read_raw_fif(eeg_file, preload=True)	
	return raw


def loadEEGh5(subject, data_dir, condition,
	eeg_epochs=True, resp_mean = True, binarymat=False, binaryfeatmat = False, envelope=False, pitch=False, gabor_pc10=False, 
	spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False):
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
	with h5py.File('%s/full_AV_matrix.hf5'%(data_dir),'r') as fh:
		all_stim = [k for k in fh[condition].keys()]
		print(all_stim)
			
		for idx, wav_name in enumerate(all_stim): 
			print(wav_name)
			stim_dict[wav_name] = []
			resp_dict[wav_name] = []
			try:
				epochs_data = fh['%s/%s/resp/%s/epochs' %(condition, wav_name, subject)][:]
				#phnfeatmat = fh['%s/%s/stim/phn_feat_timings' %(condition, wav_name)][:]
				#ntimes = phnfeatmat.shape[1] #always resample to the size of phnfeat 
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
					gabor_pc10_mat = fh['%s/%s/stim/gabor_pc10' %(condition, wav_name)][:]
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
			mouth_sc_gabor=False, full_gabor_sc_mouth=False, mouthing_sc = False, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0):

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

	if full_gabor: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=True, 
									 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False)
		strf_output = 'pitchenvsphnfeatgabor10pc'

	if full_gabor_sc: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=True, 
									 spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False)
		strf_output = 'pitchenvsphnfeatgabor10pc_sc'

	if full_gabor_sc_mouth: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=True, 
									 spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=True)
		strf_output = 'pitchenvsphnfeatgabor10pc_sc_mouthing'
	if full_model:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=False, 
									 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False)
		strf_output = 'pitchenvsphnfeat'

	if pitchUenvs: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=True, pitch=True, gabor_pc10=False, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False)
		strf_output = 'envspitch'

	if pitchUphnfeat: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=False, pitch=True, gabor_pc10=False, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False)
		strf_output = 'pitchphnfeat'

	if envsUphnfeat: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False)
		strf_output = 'envsphnfeat'

	if phnfeat_only: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False)
		strf_output = 'phnfeat'

	if envs_only: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False)
		strf_output = 'envs'

	if pitch_only: 
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=False, pitch=True, gabor_pc10=False, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False)
		strf_output = 'pitch'

	if gabor_only:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, 
										 spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=False)
		strf_output = 'gabor_only'
	

	if scene_cut:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False)
		strf_output = 'scene_cut'

	if scene_cut_gaborpc:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False)
		strf_output = 'scene_cut_gabor'

	if mouthUphnfeat:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=False, mouthing=True)
		strf_output = 'mouthing_phnfeat'		
	
	if mouth_sc_phnfeat:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=True)
		strf_output = 'mouthing_SC_phnfeat'

	if mouth_sc_gabor:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=True)
		strf_output = 'mouthing_SC_gabor'

	if phnfeat_SC:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False)
		strf_output = 'phnfeat_SC'		
	if mouthing_sc:
		resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, 
			spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=True)
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
	
	raw = load_raw_EEG(subject, block, data_dir)
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

	# Plot an * if the correlation is significantly higher than chance at p<0.05
	for i,p in enumerate(all_pvals[0]):
		if p<0.05:
			plt.text(i, max_corr, '*')

	# Plot the shuffled correlation distribution
	shuffle_mean = np.vstack((all_corrs_shuff)).mean(0) #correlation output form model -- which electrodes are correlated w/ each other, take average of this
	shuffle_stderr = np.vstack((all_corrs_shuff)).std(0)/np.sqrt(nboots_shuffle) #normalization of which electrodes are correlated w/ each other

	plt.fill_between(np.arange(nchans), shuffle_mean-shuffle_stderr, #normalization here
					 shuffle_mean+shuffle_stderr, color=[0.5, 0.5, 0.5])
	plt.plot(shuffle_mean, color='k')
	plt.gca().set_xticks(np.arange(len(all_corrs[0])))
	plt.gca().set_xticklabels(chnames, rotation=90);
	plt.xlabel('Channel')
	plt.ylabel('Model performance')
	plt.legend(['Actual data','Null distribution'])
	plt.savefig('%s/%s/%s_ch_distribution_%s.pdf' %(data_dir, subject, strf_output, condition)) #save fig

	
	#plot the significant correlations for participant on topo map 
	significant_corrs = np.array(all_corrs[0])
	significant_corrs[np.array(all_pvals[0])>0.05] = 0

	plt.figure(figsize=(5,5))
	print(['eeg']*2)
	info = mne.create_info(ch_names=raw.info['ch_names'][:64], sfreq=raw.info['sfreq'], ch_types=64*['eeg'])
	raw2 = mne.io.RawArray(np.zeros((64,10)), info)
	montage = mne.channels.read_custom_montage('%s/montage/AP-128.bvef' %(data_dir))
	raw2.set_montage(montage) #set path for MNE montage file
	mne.viz.plot_topomap(significant_corrs, raw2.info, vmin=0, vmax=max_corr)
	#plt.savefig('%s/%s/%s_topomap_%s.png' %(data_dir, subject, strf_output, stimulus_class)) #save fig

	#plt.savefig('Topomap_MT.png')
	print(np.array(all_wts).shape)

	#save STRF as .h5 file based on condition type:

	#strf_file = '%s/%s/%s_STRF_by_pitchenvsphnfeat_MT.hf5'%(data_dir, subject, subject)
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


	plt.figure()

	plt.hist(np.hstack((all_corrs_shuff)).ravel(), bins=np.arange(-0.2,max_corr,0.005), alpha=0.5, density=True)
	plt.hist(all_corrs[0], bins=np.arange(-0.2,max_corr,0.005), alpha=0.5, density=True)
	plt.xlabel('Model fits (r-values)')
	plt.ylabel('Number')
	plt.title('Correlation histograms')
	plt.legend(['Null distribution', 'EEG data'])
	plt.savefig('%s/%s/%s_corrHistogram_%s.pdf' %(data_dir, subject, condition, strf_output)) #save fig
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



def cross_prediction(data_dir, subject_list, test_cond='A', train_cond='AV', test_feature='pitchenvsphnfeat', 
			 train_feature='pitchenvsphnfeat', delay_min=0.0, delay_max=0.6, wt_pad=0.0, fs=128., av_sigs=True):
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

	
	plt.savefig(f'{save_dir}/{test_feature}_{train_feature}_xpred.pdf')

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


if __name__ == "__main__":
	user = 'maansidesai'

	data_dir = f'/Users/{user}/Box/MovieTrailersTask/Data/EEG/Participants/'
	save_dir = f'/Users/{user}/Box/Desai_Conference_and_Journal_Submissions/interspeech-2023_conference/python_figs'
	subject_list = ['MT0028', 'MT0029', 'MT0030','MT0031' ,'MT0032', 'MT0033', 'MT0034', 'MT0035','MT0036', 'MT0037', 'MT0038']
	feature_type = 'auditory' #visual or auditory
	plt.ion()

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
		
		cross_prediction(data_dir, subject_list, test_cond='V', train_cond='AV', test_feature='scene_cut_gabor', 
				train_feature='pitchenvsphnfeatgabor10pc_sc', delay_min=0.0, delay_max=0.6, wt_pad=0.0, fs=128., av_sigs=False)		
		# roi_cross_prediction(data_dir, subject_list, test_cond='V', train_cond='AV', test_feature='scene_cut_gabor', 
		# 	 train_feature='scene_cut_gabor', delay_min=0.0, delay_max=0.6, wt_pad=0.0, fs=128., av_sigs=False)
	else:
		print('Feature must be auditory or visual ')

	channel_cond_corr(subject_list, data_dir)

