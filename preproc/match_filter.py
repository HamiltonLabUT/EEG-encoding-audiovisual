import numpy as np
import scipy.signal
import scipy.stats
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

def match_filter(template_sound, anin_signal, fs, corr_thresh=0.8, nreps=1, remove_bad_events=True, debug=False, flip_phase=True, show_fig=False):
    """
    Find instances of the template signal [template_sound] in an analog channel signal [anin_signal].  Both must 
    be sampled at sampling rate [fs], so use resample before passing to this function.
    
    Parameters
    ---------------------------
      template_sound: n-length vector representing sound signal to search for
      anin_signal: m-length vector (where m > n) for ANIN1 or ANIN2 (usually you will want to search using ANIN2, 
                   the speaker channel)
      fs: sampling rate of the signals
      corr_thresh: Threshold for correlation between template and ANIN, used
                   to include or reject events
      nreps: Number of repetitions expected for this sound (if you know that
             it occurs a certain number of times.  Default = 1 repetition.)
      
      remove_bad_events: [True]/False.  Whether to remove event times that are below
                         [corr_thresh]. Sometimes when debugging you may want
                         to keep bad events.  Default = True.
      flip_phase: [True]/False. Flip the phase before performing convolution. You may need to do this or not..
      debug: True/[False]. Returns optional outputs specified below.
    Returns
    ---------------------------
      evnt_time: [nreps x 2]. Start and end time (in seconds) of the sound
                 [template_sound] as it is found in [anin_signal].
      cc: [nreps x 1]. Correlation between [template_sound] and [anin_signal]
          for each repetition. Used to determine whether event is a good
          match.
    Optional Outputs:
      matched_segment: [nreps x len(template_sound)].  Waveform on
                       [anin_signal] that was found as a match.  Should look
                       very similar to [template_sound].
      match_conv: [nreps x len(anin_signal)].  Result of the
                    convolution.  Usually only returned for debugging 
                    purposes.
    Written 2015 by Liberty Hamilton
    Example
    --------------------------
    >>> import scipy.io
    >>> dat = scipy.io.loadmat('/Users/liberty/Documents/UCSF/changrepo/matlab/preprocessing/EventDetection/sample_sounds.mat')
    >>> corr_thresh = 0.8
    >>> nreps = 12
    >>> [ev,cc] = match_filter(dat['template_sound'], dat['anin_signal'], dat['fs'], corr_thresh, nreps)
    Found a match for sentence (62.000-64.000), rep 1, r=0.890
    Found a match for sentence (168.000-170.000), rep 2, r=0.894
    Found a match for sentence (106.000-108.000), rep 3, r=0.873
    Found a match for sentence (199.000-201.000), rep 4, r=0.903
    Found a match for sentence (211.000-213.000), rep 5, r=0.845
    Found a match for sentence (90.000-93.000), rep 6, r=0.925
    Found a match for sentence (121.000-124.000), rep 7, r=0.940
    Found a match for sentence (148.000-150.000), rep 8, r=0.950
    Found a match for sentence (10.000-12.000), rep 9, r=0.968
    Found a match for sentence (33.000-35.000), rep 10, r=0.969
    Could not find a match for rep 11, best correlation was r=0.114
    Removing non-matching events with corr < 0.80
    Could not find a match for rep 12, best correlation was r=0.114
    Removing non-matching events with corr < 0.80
    """

    # Initialize variables
    evnt_time = [] # Start time and end time for each repetition of this template sound
    cc = [] # Correlation coefficient between template and the match found in anin_signal (helps determine whether event detection was successful)
    matched_segment = [] # Matching segment (should look like template)

    signal = np.copy(anin_signal)

    if show_fig:
        plt.ion()
        fig3 = plt.figure(num=2, constrained_layout=True)
        plt.show()

    for r in np.arange(nreps):
        found = False
        # Perform circular convolution
        if flip_phase:
            flip = -1
        else:
            flip = 1
        match_conv = scipy.signal.fftconvolve(flip*template_sound[::-1], signal, mode = 'full') # convolution between template and analog audio signal
            
        # sort by maximum of the convolution, this tells you where the END
        # of the events occurred. We sort in case there is more than one
        # example of the sentence detected in the TDT signal
        end_time = np.argmax(match_conv) + 1
        start_time = end_time - template_sound.shape[0]
        
        if start_time < 0:
            start_time = 1
            end_time = start_time + template_sound.shape[0]
            print('Start time was negative! This is likely bad.')
        
        # Append the start and end times
        evnt_time.append( [ start_time, end_time ] )

        # Get the segment that matches according to the convolution
        # Added negative sign to flip the audio signal for the match that was found,
        # otherwise we get a negative correlation
        matched_segment.append(signal[start_time:end_time])
        
        # correlation between sentence and the "match"
        # Pearson correlation
        print(template_sound.ravel().shape)
        print(signal[start_time:end_time].shape)
        if len(template_sound) == len(signal[start_time:end_time]):
            cc_tmp = np.corrcoef(flip*template_sound.ravel(), signal[start_time:end_time]) 
            cc.append(cc_tmp[0,1])
        
        
            # If the correlation is good enough, consider this a true match
            if (cc[-1] > corr_thresh):# & (evnt_time[-1] not in evnt_time[:-1]): # Find last element in cc list
                print('***Found a match for sentence (%4.3f-%4.3f), rep %d, r=%3.3f'%(evnt_time[-1][0]/fs, evnt_time[-1][1]/fs, r + 1, cc[-1]))
            else:
                print('Could not find a match for rep %d, best correlation was r=%3.3f at %4.3f-%4.3f'%(r + 1, cc[-1], evnt_time[-1][0]/fs, evnt_time[-1][1]/fs))
                #if evnt_time[-1] not in evnt_time[:-1]:
                # match_conv[np.arange(np.int(evnt_time[-1][0]), np.int(evnt_time[-1][1]))] = 0
                # match_conv[np.argmax(match_conv)-np.int(fs*0.1):np.argmax(match_conv)+np.int(fs*0.1)] = 0
                #signal[np.arange(np.int(evnt_time[-1][0]), np.int(evnt_time[-1][1]))] = 0
                if remove_bad_events:
                    print('Removing non-matching events with corr < %2.2f'%(corr_thresh))
                    evnt_time.pop()
                    matched_segment.pop()
                    #match_conv.pop()
                    cc.pop()
                print("Skipping the rest for this sentence...")
                break
    
            if show_fig:
                fig3.clf()
                gs = fig3.add_gridspec(3, 3)
                f3_ax1 = fig3.add_subplot(gs[0,:])
                f3_ax1.plot(signal)
                f3_ax1.axvline(start_time, color='r')
                f3_ax1.axvline(end_time, color='r')
                f3_ax1.set_title('Current signal with matches zeroed out')
                f3_ax2 = fig3.add_subplot(gs[1,:2])
                f3_ax2.plot(template_sound.ravel())
                f3_ax2.set_title('Template sound to find')
                f3_ax3 = fig3.add_subplot(gs[2,:2])
                f3_ax3.plot(matched_segment[-1])
                f3_ax3.set_title('Match, r=%2.2f'%(cc[-1]))
                f3_ax4 = fig3.add_subplot(gs[1:,2:])
                f3_ax4.hist(np.array(cc))
                f3_ax4.set_xlim([0,1])
                f3_ax4.set_xlabel('Correlation')
                plt.pause(0.1)
    
            if cc[-1] > corr_thresh:
                signal[start_time:end_time] = 0
    
    # convert event times from samples to seconds
    evnt_time = np.array(evnt_time)/np.float(fs) 
    
    if debug:
        return evnt_time, cc, signal.T, matched_segment, match_conv
    else:
        return evnt_time, cc

if __name__ == "__main__":
    import doctest
    doctest.testmod()
