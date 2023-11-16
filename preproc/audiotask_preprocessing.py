"""Module containing functions for preprocessing EEG/ECOG data from audio-based tasks

Contains the following functions:
    * get_filename - get name of data file
    * get_template_names - get list of template '.wav' files
    * EEGaud_to_hf5 - convert recorded audio to a '.hf5' file
    * load_hf5_aud - load '.hf5' file with audio data and sampling frequency
    * resample - resample audio to specified sampling frequency
    * to_wav - convert audio file to '.wav' file
    * plot_aud_waveform - plot waveform of audio
    * make_sound - create audio from array
    * match_template_to_stim - match template audio files to recording of auditory stimuli played"""

import numpy as np
import mne
import matplotlib.pyplot as plt
import scipy.io
import scipy.io.wavfile
import scipy.signal
from match_filter import match_filter
import sys
import math
import h5py
import os
import glob
import re
import tqdm


##### DEFINE RELEVANT FILE AND FOLDER NAMES #####

def get_filename(datadir, subject, block, filetype, subtype=None):
    """Get name of data file
    
    This function will print and return the complete name and address of a needed file.
    Function prevents the need to update file names and addresses in multiple locations
    or to remember where and how different sorts of files are saved.
    
    Parameters
    ----------
    subject : string
        subject ID
    filetype : string
        type of filename needed
        supported options : aud_vhdr / aud_hf5 / aud_wav / event_file
            -for 'aud_wav', set 'subtype' equal to sampling frequency
            -for 'event_file', set 'subtype' equal to 'template_set'
    subtype : string
        file subtype
        (default : None)
        
    Returns
    -------
    filename : string
        name of file needed, directories included"""
    
    # datadir = '/Volumes/research_projects/Hamilton/users/maansi/data/EEG/MovieTrailers/participants/%s/' %subject
    #datadir = '/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/EEG/MovieTrailers/Participants/%s/' %subject
    #datadir = '/Users/maansidesai/Box/MovieTrailersTask/Data/EEG/Participants/%s/' %subject
    datadir = f'{datadir}/{subject}'

    if filetype == 'aud_vhdr':
        filename = '%s/audio/%s_B%d_Audio.vhdr' %(datadir, subject, block)
    
    elif filetype == 'aud_hf5':
        filename = '%s/audio/%s_B%d_Audio.hf5' %(datadir, subject, block)
        
    elif filetype == 'aud_wav':
        if subtype is None:
            raise ValueError('\'subtype\' must be specified')
        if subtype>1000:
            subtype = int(subtype/1000)
            units = 'kHz'
        else:
            units = 'Hz'
        filename = '%s/audio/%s_B%d_%s%s.wav'%(datadir, subject, block, subtype, units)
        
    elif filetype == 'event_file':
        if subtype is None:
            raise ValueError('\'subtype\' must be specified')
        filename = '%s/audio/%s_B%d_%s_events.txt'%(datadir, subject, block, subtype)

    elif filetype == 'corr_file':
        if subtype is None:
            raise ValueError('\'subtype\' must be specified')
        filename = '%s/audio/%s_B%d_%s_corr.txt'%(datadir, subject, block, subtype)
        
    else:
        raise ValueError('filetype input not supported')
    
    print('%s file: %s' %(filetype, filename))
    return filename

    
def get_template_names(template_set, stim_dir):
    """Get list of template '.wav' files
    
    This function will return the list of template files for a project given the
    project's name.
    
    Parameters
    ----------
    template_set : string
        name of project/set of templates
        supported options : TIMIT1-TIMIT5 or TIMIT_all / MovieTrailers
        
    Returns
    -------
    template_files : list
        list of all the '.wav' template files"""
    
    #stim_dir = '/Volumes/research_projects/Hamilton/stimuli/'
    #stim_dir = '/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/stimuli/'
    #stim_dir = '/Users/maansidesai/Box/Stimuli/'
    if template_set.startswith('MovieTrailers'):
        template_set == 'MovieTrailers'
        template_dir = '%sMovieTrailers/' %stim_dir
        template_wavs = glob.glob(os.path.join(template_dir, '*.wav'))
    return np.unique(template_wavs)


##### MANIPULATE STIMULUS AUDIO #####

def EEGaud_to_hf5(datadir, subject, block, sampling_fs = 25000.):
    """Convert recorded audio to a '.hf5' file
    
    This function loads audio recorded during data presentation via auxiliary audio
    cable and saved via BrainVision, and it saves audio data to a '.hf5' file.
    
    Parameters
    ----------
    subject : string
        subject ID
    
    Saves
    -----
    aud_dat : numpy array
        presented audio data
    sfreq : float
        sampling frequency
    
    Returns
    -------
    aud_dat : numpy array
        presented audio data"""
    
    aud_vhdr_name = get_filename(datadir, subject, block, 'aud_vhdr')  # header file name for EEG audio data
    raw = mne.io.read_raw_brainvision(aud_vhdr_name, preload=True)  # slow

    #%timeit -n 1 -r 1 aud_dat = raw.get_data()

    aud_dat=raw.get_data(picks=[1])  # get data from second channel
    #plot_aud_waveform(ad)
    #plot_aud_waveform(aud_dat)

    # Create an info structure with info about name of Aux channel, sampling freq, and type
    info = mne.create_info(['Aux'], raw.info['sfreq'], ['stim'])

    audio_raw = mne.io.RawArray(aud_dat, info)

    aud_hf5_name = get_filename(datadir, subject, block, 'aud_hf5')
    hf = h5py.File(aud_hf5_name, 'w') # Write to the file
    hf.create_dataset('aud_dat', data=aud_dat.T) # Create a variable named 'aud_dat' and write to the file
    hf.create_dataset('sfreq', data=sampling_fs) # Create variable named 'sfreq' and write to the file
    hf.close() # Close the file because we're done adding stuff to it
    
    return aud_dat


def load_hf5_aud(aud_hf5_name):
    """Load '.hf5' file with audio data and sampling frequency
    
    Parameters
    ----------
    aud_hf5_name : string
        name of '.hf5' file
        
    Returns
    -------
    aud_dat : numpy array
        presented audio data
    sfreq : float
        sampling frequency"""
    
    with h5py.File(aud_hf5_name, 'r') as hf: # Open the file for reading
        aud_dat = hf['aud_dat'][:] # Read just the variable 'aud_dat' (the audio waveform)
        sfreq = hf['sfreq'][:]
        
    return aud_dat, sfreq


def resample(sound, fs, new_filename, new_fs=16000):
    """Resample audio to specified sampling frequency
    
    Parameters
    ----------
    sound : numpy array
        sound to be modified
    fs : int
        sampling frequency of audio to be modified
    new_filename : string
        name of audio file to be saved
    new_fs : int
        sampling frequency of audio to be saved
        (default : 16000)
    
    Saves
    -----
    new_sound : numpy array
        resampled sound
    
    Returns
    -------
    new_sound : numpy array
        resampled sound"""
    
    print('Original audio size: %d' %(sound.size))
    print('Original audio length: %d sec' %(sound.shape[0]/fs))
    print('Resampling from %s to %s' %(fs, new_fs))
    new_sound = scipy.signal.resample(sound, np.int((np.float(sound.shape[0])/fs)*new_fs))
    scipy.io.wavfile.write(new_filename, new_fs, new_sound/new_sound.max())
    print('Resampled audio size: %d' %(new_sound.size))
    print('Resampled audio length: %d sec' %(new_sound.shape[0]/new_fs))
    print('Saved to %s' %new_filename)
    return new_sound
   
    
def to_wav(subject, infile, outfile, new_fs=16000):
    """Convert audio file to '.wav' file
    
    This function loads either a '.hf5' or '.wav' file, downsamples it to the
    specified frequency, and saves the resampled sound as a '.wav' file via the
    'resample' function.
    
    Parameters
    ----------
    subject : string
        subject ID
    infile : string
        name of audio file to be modified
    outfile : string
        name of audio file to be saved
    new_fs : int
        sampling frequency of audio to be saved
        (default : 16000)
    
    Returns
    -------
    new_sound : numpy array
        resampled sound"""
    
    if os.path.isfile(outfile):
        print('Downsampled audio file already exists')
    else:
        print('Creating downsampled audio file...')
        # w : audio waveform / fs : sampling frequency
        if infile.endswith('.hf5'):
            [w, fs] = load_hf5_aud(infile)
        elif infile.endswith('.wav'):
            [fs, w] = scipy.io.wavfile.read(infile)
        else:
            raise ValueError('\'infile\' filetype not supported')
        new_sound = resample(w, fs, outfile, new_fs=new_fs)
    return new_sound


def plot_aud_waveform(aud_dat):
    """Plot waveform of audio
    
    Parameters
    ----------
    aud_dat : numpy array
        presented audio data"""
    
    plt.figure(figsize=(30,10))
    plt.plot(aud_dat.T)
    plt.show()
    
    
def make_sound(aud_dat, rate=44100):
    """Create audio from array
    
    Parameters
    ----------
    aud_dat : numpy array
        audio data
    rate : int
        sampling rate for audio
        (default : 44100)"""
    
    from IPython.display import Audio
    Audio(data=aud_dat.ravel(), rate=rate)

    
##### EVENT DETECTION #####
    
def match_template_to_stim(datadir, stim_dir, subject, block, stim_filename, template_files, template_set, corr_thresh, nreps, debug=False):
    """Match template audio files to recording of auditory stimuli played
    
    Given a recording of the audio played for a subject, this function will find where
    template audio matches the stimulus audio. An events file with the start and end times
    of each trial as well as the associated event numbers and template filenames will be
    saved. A debugging option is available so that matches can be examined.
    
    Parameters
    ----------
    subject : string
        subject ID
    stim_filename : string
        name of stimulus audio file
    template_set : string
        name of project/set of templates (to be passed to get_template_names)
        supported options : TIMIT1-TIMIT5 / movie_trailers
    corr_thresh : float
        threshold of template-stimulus correlation for good matches
    nreps : int
        maximum number of times a template was played to a subject
    debug : bool
        determines whether extra computations and plotting should occur to validate matches
        (default : False)
        
    Saves
    -----
    all_events : numpy array
        list of each trial's start time, end time, event number, and template audio file    
    
    Returns
    -------
    all_events : numpy array
        list of each trial's start time, end time, event number, and template audio file
    max_corr : numpy array
        list of each trial's start time, template audio file, and correlation with the template"""
    
    # check whether event file already exists
    event_file = get_filename(datadir, subject, block, 'event_file', subtype=template_set)
    corr_file = get_filename(datadir, subject, block, 'corr_file', subtype=template_set)
    # if os.path.isfile(event_file):
    #     return ValueError('Event file already exists.')
    
    # print run info
    print('----------')
    print('Comparing template files to stimulus audio...')
    print('----------')
    print('Stim file: %s' %stim_filename)
    print('Template set: %s' %template_set)
    print('Template-stim correlation threshold: %s' %corr_thresh)
    print('Maximum number of repetitions per template: %s' %nreps)
    print('----------')
    
    [stim_fs, stim_w] = scipy.io.wavfile.read(stim_filename)
    print('Stim aud shape: %s' %stim_w.shape)
    
    #template_files = get_template_names(stim_dir, template_set)
    print('Number of template files: %s' %len(template_files))
    print('----------')

    all_events=[]
    max_corrs=[]

    
    file_idx_dict = {}
    count = 0
    print(template_files)
    for num, filename in enumerate(template_files):
        if filename not in file_idx_dict:
            file_idx_dict[filename] = count 
            count += 1
        print('Searching for template %s/%s (%s)' %(num, len(template_files), num/len(template_files)))
        print(os.path.basename(filename))
        
        # get template
        #[template_fs, template_sound] = scipy.io.wavfile.read(filename)
        #define path for TIMIT or movie trailers 
        #stim_dir = '/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/stimuli/'
        stim_dir = '/Users/maansidesai/Box/Stimuli/'
        if template_set.startswith('TIMIT'):
            stim_dir_TIMIT = '%sTIMIT/'%(stim_dir)
            [template_fs, template_sound] = scipy.io.wavfile.read(stim_dir_TIMIT  + filename) #for TIMIT path
        else:
            [template_fs, template_sound] = scipy.io.wavfile.read(filename) #for MovieTrailers path
        
        #[template_fs, template_sound] = scipy.io.wavfile.read('/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/stimuli/TIMIT/' + filename)
        #[template_fs, template_sound] = scipy.io.wavfile.read(filename)
        if len(template_sound.shape)==1:  # if audio is mono...
            template_sound = np.atleast_2d(template_sound).T
        else:  # if audio is stereo...
            template_sound = np.atleast_2d(template_sound[:,0]).T  # arbitrarily pick first column
            
        # search for audio matching template in stimulus
        if debug:
            #[events, cc, stim_signal, matches, match_conv] = match_filter(template_sound, np.atleast_2d(stim_w).T, stim_fs, corr_thresh, nreps, remove_bad_events=False, debug=debug)
            [events, cc, stim_signal, matches, match_conv] = match_filter(template_sound, np.atleast_2d(stim_w).T, stim_fs, corr_thresh, nreps, remove_bad_events=False, debug=debug)
            #[evnts, cc, spk_signal, matches, match_conv] = match_filter.match_filter(template_sound, np.atleast_2d(spk_signal).T, spk_fs, corr_thresh, nreps, remove_bad_events=False, debug = True)
            # calculate the number of matches with a higher correlation than threshold
            # good_match_inds = [ind for ind, single_corr in enumerate(cc) if single_corr>corr_thresh]
            num_good_matches = events.shape[0]
            print('%s, number of events = %s' %(filename, num_good_matches))
            
            # update function output
            for match_ind in range(num_good_matches):
                [trial_name, start_time, stop_time] = [os.path.basename(filename), events[match_ind][0], events[match_ind][1]]
                all_events.append([start_time, stop_time, file_idx_dict[filename], trial_name])
                max_corrs.append([start_time, trial_name, cc[match_ind]])
            
        else:
            #[events, cc, stim_signal] = match_filter(template_sound, np.atleast_2d(stim_w).T, stim_fs, corr_thresh, nreps, remove_bad_events=True, debug=debug)
            [events, cc, stim_signal] = match_filter(template_sound, np.atleast_2d(stim_w).T, stim_fs, corr_thresh, nreps, remove_bad_events=True, debug=debug)
           
            # print number of matches with a higher correlation than threshold
            print(file_idx_dict[filename])
            num_good_matches = events.shape[0]
            print('%s, number of good matches = %s' %(filename, num_good_matches))
        
            # update function output
            for match in range(num_good_matches):
                trial_name, start_time, stop_time = os.path.basename(filename), events[match][0], events[match][1]
                #all_events.append([start_time, stop_time, num, trial_name])
                all_events.append([start_time, stop_time, file_idx_dict[filename], trial_name])
                #all_events.append([start_time, stop_time, file_idx_dict, trial_name])
                max_corrs.append([start_time, trial_name, cc[match]])
        #print(np.array(all_events))        
           
        # only plot if debugging
        # if debug:
        #     all_matches=True  # plot all matches vs. all good matches
            
        #     f = plt.figure(figsize=(10,10)) #10 is width, 3 is height
            
        #     # Plot just the template
        #     pp = 1
        #     if all_matches:
        #         plt.subplot(1+len(matches), 1, pp)
        #     else:
        #         plt.subplot(1+num_good_matches, 1, pp)
        #     plt.plot(template_sound, 'b')
        #     plt.gca().set_title('%s Template'%(trial_name)) # gca() is get current axis
        #     pp=pp+1

        #     # Plot each of the matches
        #     if all_matches:
        #         for match_index in np.arange(len(matches)): # Plot everything, even bad matches
        #             plt.subplot(1+len(matches), 1, pp)
        #             plt.plot(matches[match_index], 'r')
        #             plt.gca().set_title('%s Match %2.2f'%(trial_name, cc[match_index]))
        #             pp=pp+1
        #     else:
        #         for match_index in good_match_inds: # Plot only good matches
        #             plt.subplot(1+num_good_matches, 1, pp)
        #             plt.plot(matches[match_index], 'r')
        #             plt.gca().set_title('%s Match %2.2f'%(trial_name, cc[match_index]))
        #             pp=pp+1

            # Make a separate figure to show the convolution result
            # Peaks in this show where your stimulus apparently ends
            # plt.figure(figsize=(16,5))
            # plt.plot(match_conv[-1])

            # Find the top 10 values of match_conv
            # sorted_indices = np.argsort(match_conv[-1])[::-1][:10]
            # for s in sorted_indices:
            #     yl = plt.ylim()
            #     plt.vlines(s, yl[0], yl[1]) # Plot a vertical line at where the "matches" are

        print('----------')
    
    all_events = np.array(all_events)
    max_corrs = np.array(max_corrs)
    
    # sort detected events according to their start time
    s_inds=np.argsort(all_events[:,0].astype('float'))
    all_events = all_events[s_inds]
    max_corrs = max_corrs[s_inds]
    
    # write the event file
    np.savetxt(event_file, all_events, fmt='%s\t', delimiter='\t')
    np.savetxt(corr_file, max_corrs,fmt='%s\t', delimiter='\t')
    print('Output saved as %s')
    
    return all_events, max_corrs


# DEF: function to join TIMIT event files

# DEF: function to update event #s in event file from pickle dictionary

# DEF: function to validate order of TIMIT events based on order of sentences in TIMIT text files

# DEF: function to plot detected events against template '.wav' files