
from mTRF import *
import matplotlib

matplotlib.use('Agg')
user = 'maansidesai'

data_dir = f'/Users/{user}/Box/MovieTrailersTask/Data/EEG/Participants/'
block = 1

condition_list = ['AV', 'A', 'V']

subject_list = ['MT0028', 'MT0029', 'MT0030','MT0031' ,'MT0032', 'MT0033', 'MT0034', 'MT0035','MT0036', 'MT0037', 'MT0038']

for condition in condition_list:

	if condition == 'A':
		for subject in subject_list:
			print(subject)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, full_model=True,delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, envs_only = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, pitch_only = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, pitchUenvs = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, phnfeat_only = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, pitchUphnfeat = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, envsUphnfeat = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
	if condition == 'V':
		for subject in subject_list:
			print(subject)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition,gabor_only = True,delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, scene_cut=True,delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, scene_cut_gaborpc=True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, phnfeat_only = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			
			#mouthing and phonological feature - congruent and incongruent talking
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, mouthUphnfeat=True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			
			#mouthing + SC
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, mouthing_sc=True,delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)



	if condition == 'AV':
		for subject in subject_list:
			print(subject)

			#auditory only features
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, full_model=True,delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, envs_only = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, pitch_only = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, pitchUenvs = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, phnfeat_only = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, pitchUphnfeat = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, envsUphnfeat = True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			
			#visual only features
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition,gabor_only = True,delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, scene_cut=True,delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, scene_cut_gaborpc=True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			
			#audiovisual full model
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, full_gabor_sc=True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)

			#audiovisual full model + mouthing
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, full_gabor_sc_mouth=True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			
			#mouthing and phonological feature - congruent and incongruent talking
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, mouthUphnfeat=True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, mouth_sc_phnfeat=True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, mouth_sc_gabor=True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)

			#phnfeat+SC
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, phnfeat_SC=True, delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)

			#mouthing + SC
			wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(subject, block, data_dir, condition, mouthing_sc=True,delay_max=0.6, delay_min=0.0, fs=128., wt_pad=0.0)

