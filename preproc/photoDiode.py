import mne
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
from h5_funcs import load_raw_EEG, get_event_epoch

'''
This script tests out the use of the photo diode as a form of alignment with the visual only and AV or A-only conditions. 
The photo diode was attached to one of the Aux ports of the amplifier and measures changes in brightness from visual input over time.
We recorded the photo diode response across all conditions (AV, V, A) and assessed if there was a frame being dropped from the visual-only condition.
The photo diode recording confirmed a frame was being dropped for the visual only condition when compared ot AV or A-only. 
The following code below serves as verification, by converting the frame_rate in seconds to sampling rate (in 128 Hz) and subtracting a single
frame-rate in sampling rate from the visual only condition to adjust for the time difference. 

Plots for two trailers (as examples) are also written below.
'''

def photoDiode_comp(data_dir, subject, fs=128., block=1, frame_rate_sec = 0.04170833333333333, add_frame_rate = False):

	condition = ['AV', 'V']
	epochs = dict()
	for i in condition:
		event_file = pd.read_csv(f'{data_dir}/{subject}/audio/{subject}_B{block}_MovieTrailers_events.csv') #timing information of each movie trailer start			
		a = np.where(event_file['condition'] == i)
		evs = event_file.iloc[a][['onset_time', 'offset_time', 'event_id']].values
		evs[:,:2] = evs[:,:2]*fs

		if add_frame_rate:
			if i == 'V':
				evs[:,:2] = evs[:,:2] - (frame_rate_sec *fs) #add frame rate
			else:
				print('audiovisual condition')
		else:
			print('Not adding single frame to AV condition')

		evs = evs.astype(np.int)
		wav_id = event_file.iloc[a]['name'].values 

		raw = load_raw_EEG(subject, block, data_dir)

		
		for idx, i in enumerate(evs): #audiovisual
			wav_name = wav_id[idx]
			if condition == 'V':
				print('processing visual only condition')
				wav_name = wav_name.split('_visualonly_notif')[0]
				print(i)
			else:
				print('processing audiovisual condition')
			event_id = i[2]		
			ep = get_event_epoch(raw, evs, event_id)
			print(ep.shape)
			epochs[wav_name] = ep

	del epochs['thelittleprince-tlr_a720p'] #remove this because there wasn't a visual only trial detected


	plt.figure()
	#plot pele as sample trailer:
	plt.plot(epochs['pele-tlr1_a720p'].mean(0).T[1200:1600], label='AV')
	plt.plot(epochs['pele-tlr1_a720p_visualonly_notif'].mean(0).T[1200:1600], label='V')
	plt.title('Pele from 1200-1600 samples')
	plt.legend()

	plt.figure()
	#plot paddington as sample trailer:
	plt.plot(epochs['paddington-2-trailer-1_a720p'].mean(0).T[1200:1600], label='AV')
	plt.plot(epochs['paddington-2-trailer-1_a720p_visualonly_notif'].mean(0).T[1200:1600], label='V')
	plt.title('Paddington from 1200-1600 samples')
	plt.legend()

	return epochs

#######################
if __name__ == "__main__":
	user = 'maansidesai'
	data_dir = f'/Users/{user}/Box/MovieTrailersTask/Data/EEG/Participants/'
	subject = 'PhotoDiodeTest1_AVMT_20230127'

	epochs = photoDiode_comp(data_dir, subject, fs=128., block=1, frame_rate_sec = 0.04170833333333333, add_frame_rate = True)
	epochs = photoDiode_comp(data_dir, subject, fs=128., block=1, frame_rate_sec = 0.04170833333333333, add_frame_rate = False)