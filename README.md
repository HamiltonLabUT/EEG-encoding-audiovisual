# A comparison of EEG encoding models using audiovisual stimuli and their unimodal counterparts

Authors: Maansi Desai, Alyssa Field, Liberty S. Hamilton

## Abstract
Communication in the real world is inherently multimodal. When having a conversation, typically sighted and hearing people use both auditory and visual cues to understand one another. For example, objects may make sounds as they move in space, or we may use the movement of a person's mouth to better understand what they are saying in a noisy environment. Still, many neuroscience experiments rely on unimodal stimuli (visual only or auditory only) to understand encoding of sensory features in the brain. The extent to which visual information may influence encoding of auditory information and vice versa in natural environments is thus unclear. Here, we addressed this question by recording scalp electroencephalography (EEG) in 11 subjects as they listened to and watched movie trailers in audiovisual (AV), visual (V) only, and audio (A) only conditions. We then fit linear encoding models that described the relationship between the brain responses and the acoustic, phonetic, and visual information in the stimuli. We also compared whether auditory and visual feature tuning was the same when stimuli were presented in the original AV format versus when visual or auditory information was removed. We found that auditory feature tuning was similar in the AV and A-only conditions, and similarly, tuning for visual information was similar when stimuli were presented with the audio present (AV) and when the audio was removed (V only). In a cross prediction analysis, we investigated whether models trained on AV data predicted responses to A or V only test data as well as using the unimodal conditions for training. Overall, prediction performance using AV training and V test sets was similar to using V training and V test sets, suggesting that the auditory information has a relatively smaller effect on EEG. In contrast, prediction performance using AV training and A only test set was slightly worse than using matching A only training and test sets. This suggests the visual information has a stronger influence on EEG, though this makes no qualitative difference in the derived feature tuning. In effect, our results show that researchers may benefit from the richness of multimodal datasets, which can then be used to answer more than one research question.


### *Breakdown of folder and contents*

## Preprocessing (Relevant scripts to use below:)

| Filename | Description |
| --- | --- |
| `textgrid.py` | custom-written functions for reading Praat textgrids in order to interface with Python for extracting transcriptions and timing information |
| `phn_alignment_tools.py` | Takes the phoneme information from TIMIT (.phn files) and Movie Trailer (using Praat TextGrids) along with the event files generated from the EEG experiment using match filter in order to align the phonemes and categorize these phonemes into phonological features.|
| `neural_preproc_tools.py` | Set of functions using MNE-python to filter, manually reject artifact, and conduct ICA for neural preprocessing.|
| `create_h5_funcs.py` | Feature extraction file for phonological features, acoustic envelope, spectrogram, F0 pitch, binned pitch. _No changes to the code should be necessary for other ECoG subjects, except for potentially path modifications_.  |
| `EEG_create_h5_file.ipynb` | Jupyter notebook imports `h5_funcs.py`  and creates an .h5 file by reading in each participant's neural data (epoched), and planting the corresponding epoched neural data and speech feature extraction for each Movie trailer conditions (AV, A or V). This .h5 file is available at https://osf.io/tgbk7/ so the creation of this file is not necessary. |

The .h5 file which gets created from the `EEG_create_h5_file.ipynb` file is structured accordingly. It is important that this structure stays the same across all ECoG subjects because the analysis code to fit the encoding models to predict brain data to any given speech or visual feature is programmed based on the data file structure: 

--- `Main` (S000X_ECoG_matrix.hf5):

    --- `A` or `AV` or `V`
        --- `wav_name` : the name of the .wav file (either the trailer wav_name for the A or AV condition OR the trailers name with a slightly longer extension `_notif_sound.wav` for the V only) which was heard/seen
            --- `resp` : neural data
                 --- `epochs` : epoched neural data to the onset of the stimulus based on the event file generated from match_filter
            --- `stim` : stimulus feature extracted from the .wav file 
                 --- `pitches` : fundamental frequency (F0) pitch (1 feature)
                 --- `binned_pitches` : pitch information extracted in 10 different bins, between 50-300Hz (10 features)
                 --- `binned_edges` : edge boundaries for the bins 
                 --- `envelope` : acoustic envelope (1 feature)
                 --- `phn_feat_timings` : phonological features (14 features)
                 --- `phn_timings` : individual phonemes (53 features)
                 --- `gabor_10pc` : 10 features - binary matrix for 10 principle component gabor wavelet filters (AV and V conditions only!)
                 --- `scene_cut` : 1D binary matrix to provide onset where the scene in the movie trailer changed (AV and V conditions only!)

## analysis -- The following filenames and descriptions are for fitting STRFs using the .h5 file structure generated from the preprocessing folder.

| Filename | Description |
| --- | --- |
| `audio_tools` | folder with custom-written functions for auditory feature extractions |
| `mTRF.py` | The main python file which contains functions for fitting mTRFs and for also plotting receptive fields and the correlations generated from the encoding model analysis. Also contains some visualization functions. Change the user and path information at the bottom of this script to run the desired plots|
| `run-Main.ipynb` | Jupyter notebook imports `mTRF.py` to fit any desired model using the stimulus features saved in the .h5 file. You can run a combination of acoustic and linguistic features, pairwise features, or individual features by running the corresponding/desired cell in the notebook. All of the cells call functions in the `mTRF.py` so all of the plotting can also be conducted in this notebook as well. |

 
