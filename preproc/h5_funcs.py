#load modules and functions for phnfeat alignment and creating .h5 file

import textgrid 
import pandas as pd 
import re
import numpy as np 
import scipy.io # For .mat files
import h5py # For loading hf5 files
import mne # For loading BrainVision files (EEG)
import numpy as np
from numpy.polynomial.polynomial import polyfit
from audio_tools import spectools, fbtools, phn_tools
from scipy.io import wavfile
import os
import sys
np.set_printoptions(threshold=sys.maxsize)
import glob
import re
from phn_alignment_tools import get_trailer_phns_event_file
import textgrid as tg

from matplotlib import pyplot as plt
import parselmouth as pm
from parselmouth.praat import call
from praatio import tgio

from scipy import stats



#function to get phoneme + timing event file for movie trailers 
def get_trailer_phns_event_file(datadir, event_file, subject, block, fs=128.0):

	'''
		Run this function for every subject. 
		This will output the phoneme and sample (timing) info for each trailer that a subject heard/watched
		Running this function everytime is important because the subjects do not always hear/watch every trailer, 
		however they do listen to all five blocks of TIMIT 

		Code is here to generate textfile if needed and outputs the following information:

		1st column: phoneme
		2nd column: Time in samples of where phoneme occurs 
		3rd column: category of phoneme
		4th column: Name of trailer

	'''
	# datadir='/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/EEG/MovieTrailers/Participants/%s/downsampled_128'%(subject)
	# event_file = '/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/EEG/MovieTrailers/Participants/%s/audio/%s_MovieTrailers_events.txt'%(subject,subject)
	evs = np.loadtxt(event_file, dtype='f', usecols = (0, 1,2)) #read timing of events
	evs[:,:2] = evs[:,:2]*fs
	evs = evs.astype(np.int) #convert these seconds into samples 
	evnames = np.loadtxt(event_file, dtype=np.str, usecols = (3)) #name of all TIMIT wav files 
	evs_orig = evs.copy()

	basename = [w[:-4] for w in evnames] # This is the name of the wav file without .wav
		#Get all types of phonemes based on family/category: 
	fricatives = ['f','v','th','dh','s','sh','z','zh','hh', 'ch']
	plosives =['p','t','k','b','bcl','d','g', 'v']
	vowels = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'eh', 'ey', 'ih', 'ow', 'iy', 'oy', 'uh', 'uw']
	nasals = ['m', 'n', 'r', 'l', 'y', 'w', 'er', 'ng'] 

	#Creating new categories based on phoneme features:
	

	trailer_phn_start_time = [] #start time of phoneme
	trailer_phn_event_name = [] #each phoneme from text grid transcriptions 
	trailer_name = [] #name of movie trailer that correlates with start time and phoneme 
	trailer_name2 = [] # to append all trailers based on length of phonemes 
	trailer_phon_cat = []


	for idx, b in enumerate(basename):
		tg_dir = '/Users/maansidesai/Box/Stimuli/MovieTrailers/textgrids/Corrected'
		r = open('%s/%s_corrected.TextGrid'%(tg_dir,b))
		tg = textgrid.TextGrid(r.read())		
		tier_names = [t.nameid for t in tg.tiers]
		print('Now reading the file: %s' %(b))
		tier_names_nospace = [t.nameid.replace(" ", "") for t in tg.tiers]
		tier_num = 0
		all_phonemes=[t[2] for t in tg.tiers[tier_num].simple_transcript]
		all_phonemes = [x.lower() for x in all_phonemes]  #need to make all phoneme strings lower case to match TIMIT
		#trailer_phn_event_name.append(all_phonemes)
		print("The unique phonemes are:") #gives all phonemes for each movietrailer in basename
		print(np.unique(all_phonemes))
		print('--------------------------------------')

		phon_group = []
		for phon in all_phonemes:

			if phon in fricatives:
				phon_group.append('fric')
				trailer_name.append(b)

			elif phon in plosives:
				phon_group.append('plos')
				trailer_name.append(b)

			elif phon in vowels:
				phon_group.append('vow')
				trailer_name.append(b)

			elif phon in nasals:
				phon_group.append('nas')
				trailer_name.append(b)

			else:
				phon_group.append('other')
				trailer_name.append(b)

		assert len(all_phonemes) == len(phon_group), 'More labels made than samples'
		trailer_phon_cat.append(phon_group)
		print(phon_group)

		#loop to find any numbers attached to the phonemes and eliminate (i.e. take out 1 from uw1)
		for i, p in enumerate(all_phonemes):
			all_phonemes[i] = re.sub(r'[0-9]+', '', p)

		#converting start times from seconds to samples 
		start_times = [t[0] for t in tg.tiers[tier_num].simple_transcript]
		start_times = np.asarray(start_times, dtype=np.float32)
		start_times = start_times*fs
		start_times = start_times.astype(np.int)
		start_times = start_times + evs[idx,0]

		#appending to arrays 
		trailer_phn_start_time.append(start_times)
		trailer_phn_event_name.append(all_phonemes)
		trailer_name2.append([b]*len(all_phonemes))

		#concatenatate appended arrays (above)
	trailer_phn_event_name = np.concatenate(trailer_phn_event_name)
	trailer_phn_start_time = np.concatenate(trailer_phn_start_time)
	trailer_phon_cat = np.concatenate(trailer_phon_cat)
	print(len(trailer_name))
	trailer_name = np.concatenate([np.expand_dims(i,axis=0) for i in trailer_name])

	#stack all of the arrays and save as textfile 
	phn_sample_trailer_events = np.stack([trailer_phn_event_name, trailer_phn_start_time, trailer_phon_cat, trailer_name], axis=1)
	np.savetxt('%s/%s/audio/%s_B%d_trailer_phn_info.csv' %(datadir, subject, subject, block), phn_sample_trailer_events , fmt='%s\t', delimiter='\t') #output textfile, contains 3 columns 
	np.savetxt('%s/%s/audio/%s_B%d_trailer_phn_info.txt' %(datadir, subject, subject, block), phn_sample_trailer_events , fmt='%s\t', delimiter='\t') #output textfile, contains 3 columns 

	return phn_sample_trailer_events



#function to categorize phoneme categories 
def phn_categories(datadir, subject, block):
		'''
		Initializing phonemes which are the same across both movie trailers and TIMIT (as shown from Liberty's bar plot)
		Assign an index number to each phoneme 
		Append the index number to each phoneme based on phoneme category
		'''	

		phn1 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 
		'ih', 'iy', 'jh', 'k', 'l', 'm','n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v',
		 'w', 'y', 'z', 'zh']

		#assign index to each phoneme in phn1 list:
		assign_num = {i: idx for idx, i in enumerate(phn1)}
		idx_num = [assign_num[i] for i in phn1]

		trailer_dir = f'{datadir}/{subject}/audio/'
		mt = '%s/%s_B%d_trailer_phn_info.csv' %(trailer_dir,subject, block) 
		mt_reader = pd.read_csv(mt,index_col=None, header=0,encoding = "ISO-8859-1")
		mt_reader = mt_reader.dropna(axis=1, how='all') #drop NAs that appear in columns 
		mt_reader.columns = ['phn', 'sample','phn_cat', 'trailer']

		phonemes = mt_reader['phn']

		index = np.empty((mt_reader.shape[0],))
		for i, phon in enumerate(mt_reader['phn']):
			try:
				index[i] = assign_num[phon]
			except:
				index[i] = np.nan

		mt_reader['index'] = index
		np.savetxt('%s/%s_B%d_trailer_phn_info.txt' %(trailer_dir, subject, block), mt_reader , fmt='%s\t', delimiter='\t')

#loading ICA data 
def load_raw_EEG(subject, block, datadir):
	eeg_file = '%s/%s/downsampled_128/%s_B%d_postICA_rejected.fif'%(datadir, subject, subject, block)
	raw = mne.io.read_raw_fif(eeg_file, preload=True)

	# Print which are the bad channels, but don't get rid of them yet...
	raw.pick_types(eeg=True, meg=False, exclude=[])
	bad_chans = raw.info['bads']
	print("Bad channels are: ")
	print(bad_chans)

	# Get onset and duration of the bad segments in samples
	bad_time_onsets = raw.annotations.onset * raw.info['sfreq']
	bad_time_durs = raw.annotations.duration * raw.info['sfreq']

	print(raw._data.shape)

	# Set the bad time points to zero
	for bad_idx, bad_time in enumerate(bad_time_onsets):
		raw._data[:,np.int(bad_time):np.int(bad_time+bad_time_durs[bad_idx])] = 0

	#raw.plot(start=raw.annotations.onset[0])
	
	return raw

#epoching data 
def get_event_epoch(raw, evs, event_id, bef_aft=[0,0], baseline = None, reject_by_annotation=False):

	# Get duration information
	max_samp_dur = np.max(evs[(np.where(evs[:,2] == event_id)),1]-evs[(np.where(evs[:,2] == event_id)),0])
	trial_dur = max_samp_dur/raw.info['sfreq']
	
	epochs = mne.Epochs(raw, evs, event_id=[event_id], tmin=bef_aft[0], tmax=trial_dur+bef_aft[1], baseline=baseline,
						reject_by_annotation=reject_by_annotation)
	ep = epochs.get_data()
		
	return ep		


def load_event_file(subject, datadir, block):
	event_file = '%s/%s/audio/%s_B%d_MovieTrailers_events.txt'%(datadir, subject, subject, block)

		
	# Load the columns with the times    
	evs = np.loadtxt(event_file, dtype='f', usecols = (0, 1, 2))
	evs[:,:2] = evs[:,:2]*128 #128 is the downsampled frequency from EEG data
	evs = evs.astype(np.int) #this takes into account onset and offset times
	wav_id = np.loadtxt(event_file, dtype='<U', usecols = 3) #name of .wav filename
	
	return evs, wav_id


#create binary phoneme and phoneme feature matrices
def binary_phn_mat_stim(datadir, trailer_file, subject, block, wav_name, ep, condition, fs=128.): #get rid of looping through basename

	event_file = pd.read_csv(f'{datadir}/{subject}/audio/{subject}_B{block}_MovieTrailers_events.csv')
		   
	a = np.where(event_file['condition'] == condition)
	#event_file = f'{datadir}/{subject}/audio/{subject}_B{block}_MovieTrailers_events.txt'
	#evs = np.loadtxt(event_file, dtype='f', usecols = (0, 1,2)) #read timing of events
	evs = event_file.iloc[a][['onset_time', 'offset_time', 'event_id']].values
	#evs = evs.astype(np.int)
	evs[:,:2] = evs[:,:2]*fs
	evs = evs.astype(np.int) #convert these seconds into samples 
	evnames = event_file.iloc[a]['name'].values
	#evnames = np.loadtxt(event_file, dtype=np.str, usecols = (3)) #name of all TIMIT wav files 
	evs_orig = evs.copy()

	#read into MT phoneme file:
	trailer_file =f'{datadir}/event_files/trailer_phn_info_index_AV.txt'

	phoneme = np.loadtxt(trailer_file, dtype=np.str, usecols=(0)) #phoneme
	time_samples = np.loadtxt(trailer_file, dtype=np.int, usecols = (1)) #name of TIMIT stimuli/sentence read from 
	phoneme_cat = np.loadtxt(trailer_file, dtype=np.str, usecols = (2)) #phoneme category
	sentence_name = np.loadtxt(trailer_file, dtype=np.str, usecols = (3)) #name of movie trailer stim
	
	#convert from samples to seconds:
	phn_seconds = time_samples

	phn1 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 
	'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 
	'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pcl', 'q', 'r', 's', 'sh', 
	't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
	
	assign_num = {i: idx for idx, i in enumerate(phn1)}
	idx_num = [assign_num[i] for i in phn1]

	timing = dict()
	binary_phn_mat = dict()
	if condition == 'V':
		wav_name = wav_name.split('_visualonly_notif')[0]
	else:
		wav_name = wav_name.split('.wav')[0]
	timing[wav_name] = []
	mat_length = ep.shape[2]
	print(mat_length)
	binary_phn_mat = np.zeros((len(np.unique(phn1)), mat_length))
	print(binary_phn_mat.shape)
	
	for i, s in enumerate(sentence_name):
		#print(wav_name, s)

		if s == wav_name:
			#phn_time = phn_seconds[i]
			phn_time = time_samples[i]
			#phn_time = int(phn_time*128.0)
			phn_time = int(phn_time)
			timing[wav_name].append(phn_time)
			timit_phn = phoneme[i]
			
		   # print(timit_phn, phn_time)
			if timit_phn in phn1:
				phoneme_idx = assign_num[timit_phn]
				#print(phn_time)
				#print(phoneme_idx, phn_time)
				
				binary_phn_mat[phoneme_idx, phn_time] = 1
				#print(phn_time)
				

	binary_feat_mat, fkeys = phn_tools.convert_phn(binary_phn_mat.T, 'features')
	#print(binary_feat_mat)
	 
#     plt.figure(figsize=(20,8))
#     plt.imshow(binary_feat_mat.T, aspect='auto')  
#     #plt.imshow(binary_phn_mat, aspect='auto') 
#     phnlabels = fkeys
#     #phnlabels = phn1.copy()
#     plt.yticks(np.arange(len(phnlabels)), phnlabels);

#     plt.xlabel('Time')
#     #plt.xlim([2000, 2128])
#     plt.ylabel('Phonemes')
#     plt.title(wav_name)
#     plt.grid()
	return binary_feat_mat.T, binary_phn_mat

#Get acoustic envelope for each stimuli
def make_envelopes(wav_dir, wav_name, new_fs, ep, pad_next_pow2=True):    
	print("Sentence: %s"% (wav_name))
	wfs, sound = wavfile.read('%s/%s'%(wav_dir, wav_name))
	sound = sound/sound.max()
	#all_sounds[wav_name] = sound
	envelopes = []

	envelope = spectools.get_envelope(sound, wfs, new_fs, pad_next_pow2=pad_next_pow2)

	return envelope

#Get mel spectrogram: 80 bands. This will be 15 bands later with resampling
def stimuli_mel_spec(wav_dir, wav_name):
	[fs,w] = wavfile.read(wav_dir+'/'+ wav_name)
	w=w.astype(np.float)
	
	mel_spec, freqs = spectools.make_mel_spectrogram(w, fs, wintime=0.025, steptime=1/128.0, nfilts=80, minfreq=0, maxfreq=None)

	return mel_spec, freqs


def get_meanF0s_v2(fileName, steps=1/128.0, f0min=50, f0max=300):
	"""
	Uses parselmouth Sound and Pitch object to generate frequency spectrum of
	wavfile, 'fileName'.  Mean F0 frequencies are calculated for each phoneme
	in 'phoneme_times' by averaging non-zero frequencies within a given
	phoneme's time segment.  A range of 10 log spaced center frequencies is
	calculated for pitch classes. A pitch belongs to the class of the closest
	center frequency bin that falls below one standard deviation of the center
	frequency range.
	​
	"""
	#fileName = wav_dirs + wav_name
	sound =  pm.Sound(fileName)
	pitch = sound.to_pitch(steps, f0min, f0max) #create a praat pitch object
	pitch_values = pitch.selected_array['frequency']
	
	return pitch_values

def scene_cut_feature(h5_dir, textgrid_dir, filename, condition, fs=128.0):
	'''
	Uses the textgrid scene cuts for movie trailers and creates a binary matrix for 
	where a scene cut takes place. 

	Appends instances of scene cuts in samples (fs=128.0) to full_AV_matrix which contains 
	stim and resp for all subjects with corresponding auditory (and visual for MTs) features
	'''
	with h5py.File(f'%s/{filename}.hf5'%(h5_dir), 'a') as g:

		for name in glob.glob('%s/*.TextGrid' %(textgrid_dir)):
			print(name)
			r = open(name)
			grid = tg.TextGrid(r.read())
			tier_names = [t.nameid for t in grid.tiers]
			for r, i in enumerate(tier_names):
				if i == 'Scene cut':

					print(r)

			scene_tier = [t for t in grid.tiers[r].simple_transcript]
			scene_tier = np.array(scene_tier)

			scene_onset = scene_tier[:,0].astype(float)*fs
			scene_onset = scene_onset.astype(int)
			scene_offset = scene_tier[:,1].astype(float)*fs
			scene_offset = scene_offset.astype(int)

			name = name.replace(textgrid_dir, '')
			name = name.replace('_SC.TextGrid', '.wav')
			name = name.replace('_corrected', '')
			name = name.replace('/', '')
			print(name)

			#create matrix length of trailer and append scene onset as 1 
			matrix_values = np.zeros((scene_offset[-1],))
			for i in scene_onset:
				matrix_values[i] = 1 
			print(matrix_values.shape)
			matrix_dims = np.expand_dims(matrix_values, axis=0)

			#append to h5 file /stim/%s/scene_cut as binary matrix 
			if condition == 'V':
				name = name.replace('.wav', '_visualonly_notif.wav')
				g.create_dataset('%s/%s/stim/scene_cut'%(condition, name), data=np.array(matrix_dims, dtype=float))
			else:
				g.create_dataset('%s/%s/stim/scene_cut'%(condition, name), data=np.array(matrix_dims, dtype=float))

def face_talker(data_dir, textgrid_talker_dir, subject, condition, block=1, fs=128., frame_rate_sec=0.04170833333333333):
	'''
	Uses textgrids for trailers to identify if you can see the talker or not as they are speaking.
	2D Binary matrix for congruent (first row) and incongruent (second row) mouth movement.

	Appends instances of mouthing information in samples (fs=128.0) to full_AV_matrix which contains 
	stim and resp for all subjects with corresponding auditory (and visual for MTs) features
	'''

	event_file = pd.read_csv(f'{data_dir}/{subject}/audio/{subject}_B{block}_MovieTrailers_events.csv') #timing information of each movie trailer start			
	a = np.where(event_file['condition'] == condition)
	evs = event_file.iloc[a][['onset_time', 'offset_time', 'event_id']].values
	evs[:,:2] = evs[:,:2]*fs
	print(evs[:,:2].astype(int))
	print(evs[:,1:2].astype(int) - evs[:,:1].astype(int))
	if condition == 'V':
		evs[:,:2] = evs[:,:2] - (frame_rate_sec *fs) #subtract duration of 1 frame in 128.0 Hz sampling rate
	else:
		print('Audiovisual condition is being processed for mouth movement epochs')
		print('********************************************************************')
	evs = evs.astype(np.int)
	print(evs[:,:2].astype(int))
	print(evs[:,1:2] - evs[:,:1])
	wav_id = event_file.iloc[a]['name'].values 

	raw = load_raw_EEG(subject, block, data_dir)
	face_mat = dict()

	for idx, i in enumerate(evs): #audiovisual
		wav_name = wav_id[idx]
		if condition == 'V':
			print('processing visual only condition')
			wav_name = wav_name.split('_visualonly_notif')[0]
			print(i)
		else:
			print('processing audiovisual condition')
		event_id = i[2]		
		epochs = get_event_epoch(raw, evs, event_id)
		mat_length = epochs.shape[2]
		face_matrix = np.zeros((mat_length,2)) #create time x 2 matrix. first row = yes, second row = no
	
		tg = tgio.openTextgrid(f'{textgrid_talker_dir}/{wav_name}_corrected_natsounds.TextGrid')
		sc_tier = tg.tierDict['mouthing']
		df = pd.DataFrame([(start, end, label) for (start, end, label) in sc_tier.entryList],columns = ['start','end','label'])
		yes = np.where(df['label'] == 'yes')
		yes_timings = (df.iloc[yes][['start', 'end']]*fs).astype(int) #in sampling rate of 128. Hz

		no = np.where(df['label'] == 'no')
		no_timings = (df.iloc[no][['start', 'end']]*fs).astype(int) #in sampling rate of 128. Hz

		for on in range(len(yes_timings)): #where there is a time segment for congruent mouth mvmt.
			face_matrix[:,0][yes_timings.values[on][0]:yes_timings.values[on][1]] = 1

		for on in range(len(no_timings)): #where there is a time segment for incongruent mouth mvmt.
			face_matrix[:,1][no_timings.values[on][0]:no_timings.values[on][1]] = 1
		#append to dictionary above:
		face_mat[wav_name] = face_matrix #to save as stim for stim_dict?

		with h5py.File(f'{data_dir}/full_AV_matrix.hf5', 'a') as g:
			if condition == 'V':
				name = wav_name + '_visualonly_notif.wav'
				print(name)
				try:
					g.create_dataset('%s/%s/stim/mouthing'%(condition, name), data=np.array(face_mat[wav_name], dtype=float))
				except:
					print('mouthing V-only stim already exists')
			else:
				name =  wav_name + '.wav'
				print(name)
				try:
					g.create_dataset('%s/%s/stim/mouthing'%(condition, name), data=np.array(face_mat[wav_name], dtype=float))
				except:
					print('mouthing AV stim already exists')
			
	return face_mat


def get_trailer_phns_event_file(subject, user='maansidesai',raw=128.0, fs=128.0, block=1, condition='AV'):

	'''
		Run this function for every subject. 
		This will output the phoneme and sample (timing) info for each trailer that a subject heard/watched
		Running this function everytime is important because the subjects do not always hear/watch every trailer, 
		however they do listen to all five blocks of TIMIT 

		Code is here to generate textfile if needed and outputs the following information:

		1st column: phoneme
		2nd column: Time in samples of where phoneme occurs 
		3rd column: category of phoneme
		4th column: Name of trailer

	'''
	#datadir='/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/EEG/MovieTrailers/Participants/%s/downsampled_128'%(subject)
	datadir = f'/Users/{user}/Box/MovieTrailersTask/Data/EEG/Participants/'
	event_file = pd.read_csv(f'{datadir}/{subject}/audio/{subject}_B{block}_MovieTrailers_events.csv') #timing information of each movie trailer start			

	a = np.where(event_file['condition'] == condition)
	evs = event_file.iloc[a][['onset_time', 'offset_time', 'event_id', 'name']].values
	evs_orig = evs.copy()
	evnames = evs[:,3]
	evs[:,:2] = evs[:,:2]*fs
	evs[:,:2] = evs[:,:2].astype(np.int) #convert these seconds into samples 
	evs = evs[:,:3]

	#evnames = evs[:,3] #name of all TIMIT wav files 
	# evs_orig = evs.copy()

	basename = [w for w in evnames] # This is the name of the wav file without .wav
		#Get all types of phonemes based on family/category: 
	fricatives = ['f','v','th','dh','s','sh','z','zh','hh', 'ch']
	plosives =['p','t','k','b','bcl','d','g', 'v']
	vowels = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'eh', 'ey', 'ih', 'ow', 'iy', 'oy', 'uh', 'uw']
	nasals = ['m', 'n', 'r', 'l', 'y', 'w', 'er', 'ng'] 

	#Creating new categories based on phoneme features:
	

	trailer_phn_start_time = [] #start time of phoneme
	trailer_phn_event_name = [] #each phoneme from text grid transcriptions 
	trailer_phn_end_time = []
	trailer_name = [] #name of movie trailer that correlates with start time and phoneme 
	trailer_name2 = [] # to append all trailers based on length of phonemes 
	trailer_phon_cat = []


	for idx, b in enumerate(basename):
		#tg_dir = '/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/stimuli/MovieTrailers/textgrids/Corrected/'
		tg_dir =  f'/Users/{user}/Box/Stimuli/MovieTrailers/textgrids/Corrected'
		r = open('%s/%s_corrected.TextGrid'%(tg_dir,b))
		tg = textgrid.TextGrid(r.read())		
		tier_names = [t.nameid for t in tg.tiers]
		print('Now reading the file: %s' %(b))
		tier_names_nospace = [t.nameid.replace(" ", "") for t in tg.tiers]
		tier_num = 0
		all_phonemes=[t[2] for t in tg.tiers[tier_num].simple_transcript]
		all_phonemes = [x.lower() for x in all_phonemes]  #need to make all phoneme strings lower case to match TIMIT
		#trailer_phn_event_name.append(all_phonemes)
		print("The unique phonemes are:") #gives all phonemes for each movietrailer in basename
		print(np.unique(all_phonemes))
		print('--------------------------------------')

		phon_group = []
		for phon in all_phonemes:

			if phon in fricatives:
				phon_group.append('fric')
				trailer_name.append(b)

			elif phon in plosives:
				phon_group.append('plos')
				trailer_name.append(b)

			elif phon in vowels:
				phon_group.append('vow')
				trailer_name.append(b)

			elif phon in nasals:
				phon_group.append('nas')
				trailer_name.append(b)

			else:
				phon_group.append('other')
				trailer_name.append(b)

		assert len(all_phonemes) == len(phon_group), 'More labels made than samples'
		trailer_phon_cat.append(phon_group)
		print(phon_group)

		#loop to find any numbers attached to the phonemes and eliminate (i.e. take out 1 from uw1)
		for i, p in enumerate(all_phonemes):
			all_phonemes[i] = re.sub(r'[0-9]+', '', p)

		#converting start times from seconds to samples 
		start_times = [t[0] for t in tg.tiers[tier_num].simple_transcript]
		start_times = np.asarray(start_times, dtype=np.float32)
		start_times = start_times*raw
		start_times = start_times.astype(np.int)
		start_times = start_times + evs[idx,0]

		end_times = [t[1] for t in tg.tiers[tier_num].simple_transcript]
		end_times = np.asarray(end_times, dtype=np.float32)
		end_times = end_times*raw
		end_times = end_times.astype(np.int)
		end_times = end_times + evs[idx,0]


		#appending to arrays 
		trailer_phn_start_time.append(start_times)
		trailer_phn_end_time.append(end_times)
		trailer_phn_event_name.append(all_phonemes)
		trailer_name2.append([b]*len(all_phonemes))


		#concatenatate appended arrays (above)
	trailer_phn_event_name = np.concatenate(trailer_phn_event_name)
	trailer_phn_start_time = np.concatenate(trailer_phn_start_time)
	trailer_phn_end_time = np.concatenate(trailer_phn_end_time)
	trailer_phon_cat = np.concatenate(trailer_phon_cat)
	print(len(trailer_name))
	trailer_name = np.concatenate([np.expand_dims(i,axis=0) for i in trailer_name])

	#stack all of the arrays and save as textfile 
	phn_sample_trailer_events = np.stack([trailer_phn_start_time, trailer_phn_end_time, trailer_phn_event_name, ], axis=1)
	np.savetxt(f'%s/%s/audio/%s_trailer_phn_info_{condition}.csv' %(datadir, subject, subject), phn_sample_trailer_events , fmt='%s\t', delimiter='\t') #output textfile, contains 3 columns 
	np.savetxt(f'%s/%s/audio/%s_trailer_phn_info_{condition}.txt' %(datadir, subject, subject), phn_sample_trailer_events , fmt='%s\t', delimiter='\t') #output textfile, contains 3 columns 


	return phn_sample_trailer_events

def add_index(datadir, subject, condition='AV'):
	#condition = 'AV'
	phn1 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 
	'ih', 'iy', 'jh', 'k', 'l', 'm','n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v',
	 'w', 'y', 'z', 'zh']

	#assign index to each phoneme in phn1 list:
	assign_num = {i: idx for idx, i in enumerate(phn1)}
	idx_num = [assign_num[i] for i in phn1]

	trailer_dir = datadir
	mt = f'%s/%s/audio/%s_trailer_phn_info_{condition}.csv' %(trailer_dir,subject, subject) 
	mt_reader = pd.read_csv(mt,index_col=None, header=0,encoding = "ISO-8859-1")
	mt_reader = mt_reader.dropna(axis=1, how='all') #drop NAs that appear in columns 
	mt_reader.columns = ['sample_start', 'sample_end', 'phn']

	phonemes = mt_reader['phn']

	index = np.empty((mt_reader.shape[0],))
	for i, phon in enumerate(mt_reader['phn']):
		try:
			index[i] = assign_num[phon]
		except:
			index[i] = np.nan

	mt_reader['index'] = index
	np.savetxt(f'%s/%s/audio/%s_trailer_phn_info_{condition}.txt' %(trailer_dir, subject, subject), mt_reader , fmt='%s\t', delimiter='\t')