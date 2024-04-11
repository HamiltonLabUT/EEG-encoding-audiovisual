

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import h5py
import logging
import sys
sys.path.append('/Users/maansidesai/Desktop/git/EEG-encoding-audiovisual/analysis/tikreg/tikreg')
from tikreg import models
from tikreg import utils as tikutils
from tikreg import spatial_priors, temporal_priors
from ridge.utils import make_delayed

from mTRF import load_raw_EEG, loadEEGh5


user='maansidesai'
subject = 'MT0028'
data_dir = f'/Users/{user}/Box/MovieTrailersTask/Data/EEG/Participants/'
condition = 'AV'
delay_min=0.0
delay_max=0.6
wt_pad=0.0
fs=128.

feature_nums=input('Enter the number of features for fitting a banded ridge regression: ')
feature_nums=int(feature_nums)

resp_dict, stim_dict = loadEEGh5(subject, data_dir, condition, eeg_epochs=True, resp_mean = True,
									binaryfeatmat = False, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, 
									spectrogram=False, spectrogram_scaled=False, scene_cut=True, mouthing=False, filename='full_AV_matrix')
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


# Create training and validation stimulus matrices for each feature type
if feature_nums == 2:
	tStim1 = np.atleast_2d(np.vstack([np.vstack(stim_dict[r][0]).T for r in training_set]))
	vStim1 = np.atleast_2d(np.vstack([np.vstack(stim_dict[r][0]).T for r in test_set]))
	tStim1 = tStim1/tStim1.max(0)
	vStim1 = vStim1/vStim1.max(0)

	tStim2 = np.atleast_2d(np.vstack([np.vstack(stim_dict[r][1]).T for r in training_set]))
	vStim2 = np.atleast_2d(np.vstack([np.vstack(stim_dict[r][1]).T for r in test_set]))
	tStim2 = tStim2/tStim2.max(0)
	vStim2 = vStim2/vStim2.max(0)

	print('**********************************')
	print('creating testing and validation stim matricies for feature #1')
	print(tStim1.max(0).shape)
	print(vStim1.max(0).shape)
	print('**********************************')

	print('**********************************')
	print('creating testing and validation stim matricies for feature #2')
	print(tStim2.max(0).shape)
	print(vStim2.max(0).shape)
	print('**********************************')

	# tStim1 = make_delayed(tStim1, delays)
	# vStim1 = make_delayed(vStim1, delays)

	# tStim2 = make_delayed(tStim2, delays)
	# vStim2 = make_delayed(vStim2, delays)

chunklen = int(len(delays)*3) # We will randomize the data in chunks 
#nchunks = np.floor(0.2*tStim.shape[0]/chunklen).astype('int')

nchans = tResp.shape[1] # Number of electrodes/sensors
print('*************************')
print('printing vStim: ')
print(vStim1.shape, vStim2.shape)

print('*************************')
print('printing vResp: ')
print(vResp.shape)
print('*************************')
print('printing tStim: ')
print(tStim1.shape, tStim2.shape)
print('*************************')
print('printing tResp: ')
print(tResp.shape) #training

nfeatures1=vStim1.shape[1]
nfeatures2=vStim2.shape[1]
delays=np.arange(10)

moten_prior = spatial_priors.SphericalPrior(nfeatures1)
obcat_prior = spatial_priors.SphericalPrior(nfeatures2)
print(moten_prior.asarray.shape)
print(obcat_prior.asarray.shape)

temporal_prior = temporal_priors.SphericalPrior(delays=delays)
print(temporal_prior.asarray.shape)

fit_spherical_pop = models.estimate_stem_wmvnp([tStim1, tStim2], tResp, 
											   [vStim1, vStim2],vResp,
											   feature_priors=[moten_prior, obcat_prior],
											   temporal_prior=temporal_prior,
											   ridges=np.logspace(0,4,10),
											   folds=(1,5),
											   performance=True,
											   population_optimal=True,
											   verbosity=2, method='Chol')

# fit_bandedhrf_polar = models.estimate_stem_wmvnp([tStim1, tStim2], tResp, 
#                                                [vStim1, vStim2],vResp,
#                                                feature_priors=[moten_prior, obcat_prior],
#                                                temporal_prior=temporal_prior,
#                                                ridges=np.logspace(0,4,10),
#                                                normalize_hyparams=True,
#                                                folds=(1,5),
#                                                performance=True,
#                                                verbosity=2)
# # Specify fit
options = dict(ridges=np.hstack((0, np.logspace(2,8,20))), weights=True, metric='rsquared')
#fit = models.cvridge(tResp, tStim, vResp, vStim, **options)


# weights_estimate = fit['weights']
# weights_corr = tikutils.columnwise_correlation(weights_true, weights_estimate) #actual, predicted from model
# print(weights_corr.mean())  	# > 0.9
# print(fit['cvresults'].shape) # (5, 1, 11, 2): (nfolds, 1, nridges, nresponses)