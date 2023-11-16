# make_event.py
#
# Written October 2021 by Liberty Hamilton

import numpy as np
import match_filter
import glob
import librosa 
import os
import pandas as pd
from matplotlib import pyplot as plt

def create_event_file(subj, block, neural_dir, event_times, event_ids, neural_fs=100, etype='sentence'):
    '''
    Create the event file for MNE
    
    Inputs:
    -----------
        subj [str]:         The subject ID, e.g. 'S0007' 
        block [str] :       The block, e.g. 'B3'
        neural_dir [str] :  The path to the neural data (e.g. '/Users/liberty/Box/ECoG_Backup/')
        event_times [list]: A list of the event onset and offset times, in order
        event_ids [list]:   A list of the event IDs
        neural_fs [int] :   Sampling rate of the neural data in Hz (default: 100)

    Outputs:
    ------------
        eve [list] :        List of event onset sample, offset sample, and ID
                            for use in MNE python

    '''
    eve_tmp = np.hstack((event_times, np.atleast_2d(event_ids).T))
    eve_tmp[:,:2] = eve_tmp[:,:2]*neural_fs
    eve = eve_tmp.astype(int)

    eve_file = os.path.join(neural_dir, f'{subj}', 'audio' , f'{subj}_{block}_MovieTrailers_events.txt')
    print(f"Saving to {eve_file}")
    np.savetxt(eve_file, eve, fmt='%d', delimiter='\t')
    return eve


def match_filter_directory(full_audio, wav_dir, audio_format='wav', audio_fs=16000, nreps=1, corr_thresh=0.6, flip_phase=False, show_fig=False):
    '''
    Run the match filter script on a full directory to find instances of each
    wav file in the [full_audio] from an EEG or ECoG experiment. 

    Inputs:
    -----------
        full_audio [np.array] :     Vector of sound waveform
        wav_dir [str] :             Path to the audio files you want to find within your EEG/ECoG
        audio_format [str] :        the extension of the audio files in [wav_dir]. Usually 'wav'
                                    but could also be 'mp3' or otherwise
        audio_fs [int] :            Sampling rate of audio data
        nreps [int] :               Number of repeats expected for each sound in [wav_dir]. For
                                    example, for TIMIT5, there are 10 sentences repeated 10 times,
                                    so nreps should be 10. 

    Outputs:
    -----------    
        ordered_event_names [list]: A list of all of the wavs that were played, in order
        ordered_event_times [list]: A list of the event onset and offset times, in order
        ordered_event_corrs [list]: A list of the correlation between signal and match, in order
        ordered_event_ids [list]:   A list of the event IDs
 
    '''
    # Get the audio files from [wav_dir]
    #audio_files = glob.glob(f'{wav_dir}/*.{audio_format}')
    #audio_files.sort()  # In alphabetical order

    # Put audio files in a specific order
    audio_files = ['pele-tlr1_a720p',
                   'bighero6-tlr3_a720p',
                   'ferdinand-trailer-2_a720p',
                   'insideout-tlr2zzyy32_a720p',
                   'insideout-usca-tlr2_a720p',
                   'paddington-2-trailer-1_a720p',
                   'thelittleprince-tlr_a720p',
                   'deep-trailer-1_a720p',
                   ]

    all_event_times = []
    all_event_names = []
    all_event_corrs = []
    all_event_ids = []
    for trial_type in ['.wav', '_visualonly_notif.wav']:
        for audio_idx, audio_file in enumerate(audio_files):
            corr_thresh_use = corr_thresh
            if trial_type == '_visualonly_notif.wav':
                corr_thresh_use = 0.8
            audio_file = wav_dir + '/' + audio_file + trial_type 
            print(audio_file)
            template_sound, fs = librosa.load(audio_file, sr=audio_fs, mono=True)
            print(template_sound.shape)
            event_time, cc = match_filter.match_filter(template_sound, full_audio, fs,
                                                       corr_thresh=corr_thresh_use, nreps=nreps,
                                                       remove_bad_events=True, debug=False,
                                                       flip_phase=flip_phase, show_fig=show_fig)

            if (len(event_time) > 0) and (len(cc) > 0):
                for ev in event_time:
                    full_audio[int(ev[0]*audio_fs):int(ev[1]*audio_fs)] = 0
                all_event_names.append([audio_file]*len(event_time))
                all_event_times.append(event_time)
                all_event_corrs.append(cc)
                all_event_ids.append([audio_idx]*len(event_time))

    print(all_event_corrs)
    print(all_event_names)
    print(all_event_times)
    if len(all_event_names) > 1:
        stacked_event_names = np.hstack((all_event_names))
        stacked_event_times = np.vstack((all_event_times))
        stacked_event_corrs = np.hstack((all_event_corrs))
        stacked_event_ids = np.hstack((all_event_ids))
    
        #Sort by time
        time_order = np.argsort(stacked_event_times[:,0])
        ordered_event_names = [stacked_event_names[t] for t in time_order]
        ordered_event_times = stacked_event_times[time_order,:]
        ordered_event_corrs = stacked_event_corrs[time_order]
        ordered_event_ids = stacked_event_ids[time_order]
    else:
        ordered_event_times = all_event_times[0]
        ordered_event_names = all_event_names[0]
        ordered_event_corrs = all_event_corrs[0]
        ordered_event_ids = all_event_ids[0]

    return ordered_event_names, ordered_event_times, ordered_event_corrs, ordered_event_ids

def main(subj, block, wav_dir, neural_dir, flip_phase=False, nreps=10,
         corr_thresh=0.6, show_fig=True, neural_fs=100):
    '''
    Run the main routine if this script is called

    Inputs:
    -------------
        subj [str]:             The subject ID, e.g. 'S0007' 
        block [str] :           The block, e.g. 'B3'
        neural_dir [str] :      The path to the neural data (e.g. '/Users/liberty/Box/ECoG_Backup/')
        wav_dir [str] :         Path to the stimuli (e.g.'/Users/liberty/Documents/Austin/code/TIMIT_iPad/Sounds')
        flip_phase [bool] :     Whether to flip the phase of the template (True/False, default: False)
        corr_thresh [float] :   Threshold for finding a match (between 0 and 1, 0.6 is reasonable). Too small
                                will result in false positives
        show_fig [bool] :       Show a debugging figure (True/False, default: True)
        neural_fs [int] :       Sampling rate of neural data in Hz (default: 100)

    Outputs:
    -------------
        ordered_event_names [list]: A list of all of the wavs that were played, in order
        ordered_event_times [list]: A list of the event onset and offset times, in order
        ordered_event_corrs [list]: A list of the correlation between signal and match, in order
        ordered_event_ids [list]:   A list of the event IDs
        eve [list] :                The final list of event onset sample, offset sample, and ID for MNE

    '''
    neural_audio_file = os.path.join(neural_dir, f'{subj}', 'audio', f'{subj}_{block}_16kHz.wav')
    print(f'Loading {neural_audio_file}')
    full_audio, neural_audio_fs = librosa.load(neural_audio_file, sr=16000, mono=True)
    
    plt.ion()
    plt.plot(full_audio)
    plt.title('Full EEG audio - check for clipping or bad signal')

    print('Running match filter to find each sentence played in this block')
    ordered_event_names, ordered_event_times, ordered_event_corrs, ordered_event_ids = match_filter_directory(full_audio, wav_dir, nreps=nreps, corr_thresh=corr_thresh, flip_phase=flip_phase, show_fig=show_fig)
    
    short_names = [p.split('/')[-1].split('.')[0] for p in ordered_event_names]

    # Put the data into a dataframe and save it as a csv
    data = {'name': short_names, 'path_name': ordered_event_names, 
            'event_id': ordered_event_ids, 
            'onset_time': ordered_event_times[:,0],
            'offset_time': ordered_event_times[:,1],
            'match_filt_corr': ordered_event_corrs}
    df = pd.DataFrame(data)
    
    print("Saving data frame csv file with match filter results")
    csv_file = os.path.join(neural_dir, f'{subj}', 'audio', f'{subj}_{block}_MovieTrailers_events.csv')
    mf_fig = os.path.join(neural_dir, f'{subj}', 'audio', f'{subj}_{block}_event_times_corrs.pdf')
    df.to_csv(csv_file)
    df.boxplot(by='name', column='match_filt_corr',rot=90)
    plt.axhline(corr_thresh, color='r')
    plt.gca().set_ylim([0,1.1])
    plt.ylabel('Matched Filter Correlation')
    plt.tight_layout()
    plt.savefig(mf_fig)

    print("Saving the event file")
    eve = create_event_file(subj, block, neural_dir, ordered_event_times, 
                            ordered_event_ids, neural_fs)

    return ordered_event_names, ordered_event_times, ordered_event_corrs, ordered_event_ids, eve

#######################
if __name__ == "__main__":
    subj = 'PhotoDiodeTest1_AVMT_20230127'
    block = 'B1'
    stim_type = 'MovieTrailers'
    #wav_dir = f'/Users/alyssa/Library/CloudStorage/Box-Box/AV_mixtures_MT/Stimuli/{stim_type}'
    wav_dir = f'/Users/maansidesai/Box/AV_mixtures_MT/Stimuli/{stim_type}'
    #neural_dir = '/Users/alyssa/Library/CloudStorage/Box-Box/MovieTrailersTask/Data/EEG/Participants'
    neural_dir = '/Users/maansidesai/Box/MovieTrailersTask/Data/EEG/Participants'
    corr_thresh = 0.2
    nreps = 3
    ordered_event_names, ordered_event_times, ordered_event_corrs, ordered_event_ids, eve = main(subj, block, wav_dir, neural_dir, flip_phase=False, nreps=nreps, corr_thresh=corr_thresh, show_fig=True)

