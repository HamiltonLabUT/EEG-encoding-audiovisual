import textgrid 
import pandas as pd 
import re
import numpy as np 

'''
The purpose of these functions are to use them when dealing with your raw EEG data 
See below for concise definitions of each function: 

* TIMIT_phn_groups: ONLY NEED TO RUN THIS FUNCTION ONCE. Reads through all of the TIMIT .phn files and concatenates all sentences with phoneme, phoneme category, and timing information (in seconds)
* get_trailer_phns_event_file: RUN FOR EVERY SUBJECT. Will align phoneme information during task for each movie trailer based on phonemes from textgrids, will convert to correct time in samples 
* get_TIMIT_phns_event_file: RUN FOR EVERY SUBJECT. Align large textfile with all phonemes for each sentence with each subject's TIMIT timings and convert to correct time in samples. 
* phn_categories: appending an index value using pandas to a csv and .txt file and initializes the overlapping/unique phonemes for both conditions (MT/TIMIT)

'''

# def trailer_phn_groups(raw=128.0):
# 	'''
# 	Only run this function once - the file is already made
# 	This output file take all of the phonemes + timings for all movie trailers from the transcribed textgrids, adds phoneme categories
# 	and eventually adds an index number to designate each phoneme 

# 	[output]: a textfile with unique phonemes, phoneme categories, index number and timing information 
# 	'''
# 	#tg_dir = '/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/stimuli/MovieTrailers/textgrids/Corrected/'
# 	#tg_dir = '/Users/maansidesai/Box/Stimuli/MovieTrailers/textgrids/Corrected'
# 	movie_file_name = []
# 	for name in glob.glob(os.path.join(tg_dir, '*_corrected.TextGrid')):
# 		name = os.path.splitext(os.path.basename(name)[:-9])[0]
# 		if name.endswith('_corrected'):
# 			name = name[:-10]
# 			np.array(movie_file_name.append(name))
		
# 	movie_file_name = np.array(movie_file_name)
# 	basename = movie_file_name


# 	#Get all types of phonemes based on family/category: 
# 	fricatives = ['f','v','th','dh','s','sh','z','zh','hh', 'ch']
# 	plosives =['p','t','k','b','bcl','d','g', 'v']
# 	vowels = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'eh', 'ey', 'ih', 'ow', 'iy', 'oy', 'uh', 'uw']
# 	nasals = ['m', 'n', 'r', 'l', 'y', 'w', 'er', 'ng'] 

# 	trailer_phn_start_time = [] #start time of phoneme
# 	trailer_phn_event_name = [] #each phoneme from text grid transcriptions 
# 	trailer_name = [] #name of movie trailer that correlates with start time and phoneme 
# 	trailer_name2 = [] # to append all trailers based on length of phonemes 
# 	trailer_phon_cat = []

# 	#read into MT phoneme .phn files:
# 	for idx, b in enumerate(basename):
# 		tg_dir = '/Users/maansidesai/Box/Stimuli/MovieTrailers/textgrids/Corrected'
# 		#tg_dir = '/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/stimuli/MovieTrailers/textgrids/Corrected/'
# 		r = open('%s/%s_corrected.TextGrid'%(tg_dir,b))
# 		tg = textgrid.TextGrid(r.read())
# 		tier_names = [t.nameid for t in tg.tiers]
		
# 		print('Now reading the file: %s' %(b))
# 		tier_names_nospace = [t.nameid.replace(" ", "") for t in tg.tiers]
# 		tier_num = 0
# 		all_phonemes=[t[2] for t in tg.tiers[tier_num].simple_transcript]
# 		all_phonemes = [x.lower() for x in all_phonemes]  #need to make all phoneme strings lower case to match TIMIT
		
# 		#convert start times to samples from seconds 
# 		start_times = [t[0] for t in tg.tiers[tier_num].simple_transcript]
# 		start_times = np.array(start_times, dtype=np.float32)
# 		start_times = start_times*fs
# 		start_times = start_times.astype(np.int)
		
# 		#because of rounding of float, the int values gives a onset in samples as 1, so need to make this 0 to signify onset time
# 		for x, i in enumerate(start_times):
# 			if i == 1:
# 				start_times[x] = 0

# 		print("The unique phonemes are:") #gives all phonemes for each movietrailer in basename
# 		print(np.unique(all_phonemes))
# 		print(start_times)
# 		print('--------------------------------------')
		
# 		phon_group = []
# 		for phon in all_phonemes:

# 			if phon in fricatives:
# 				phon_group.append('fric')
# 				trailer_name.append(b)

# 			elif phon in plosives:
# 				phon_group.append('plos')
# 				trailer_name.append(b)

# 			elif phon in vowels:
# 				phon_group.append('vow')
# 				trailer_name.append(b)

# 			elif phon in nasals:
# 				phon_group.append('nas')
# 				trailer_name.append(b)

# 			else:
# 				phon_group.append('other')
# 				trailer_name.append(b)

# 		assert len(all_phonemes) == len(phon_group), 'More labels made than samples'
# 		trailer_phon_cat.append(phon_group)

# 		#loop to find any numbers attached to the phonemes and eliminate (i.e. take out 1 from uw1)
# 		for i, p in enumerate(all_phonemes):
# 			all_phonemes[i] = re.sub(r'[0-9]+', '', p)

# 		trailer_phn_start_time.append(start_times)
# 		trailer_phn_event_name.append(all_phonemes)
# 		trailer_name2.append([b]*len(all_phonemes))

# 	trailer_phn_event_name = np.concatenate(trailer_phn_event_name)
# 	trailer_phn_start_time = np.concatenate(trailer_phn_start_time)
# 	trailer_phon_cat = np.concatenate(trailer_phon_cat)

# 	print(len(trailer_name))
# 	trailer_name = np.concatenate([np.expand_dims(i,axis=0) for i in trailer_name])
# 	phn_sample_trailer_events = np.stack([trailer_phn_event_name, trailer_phn_start_time, trailer_phon_cat, trailer_name], axis=1)

# 	#output into csv and txt file 
# 	save_dir = '/Users/maansidesai/Box/MovieTrailersTask/Data/EEG/Participants/event_files/'
# 	np.savetxt('%s/trailer_phn_info.csv' %(save_dir), phn_sample_trailer_events , fmt='%s\t', delimiter='\t') #output textfile, contains 3 columns 
# 	np.savetxt('%s/trailer_phn_info.txt' %(save_dir), phn_sample_trailer_events , fmt='%s\t', delimiter='\t') #output textfile, contains 3 columns 

# 	#ONLY RUN THIS CELL ONCE!
	
# 	phn1 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 
# 	'ih', 'iy', 'jh', 'k', 'l', 'm','n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v',
# 	 'w', 'y', 'z', 'zh']

# 	#assign index to each phoneme in phn1 list:
# 	assign_num = {i: idx for idx, i in enumerate(phn1)}
# 	idx_num = [assign_num[i] for i in phn1]
# 	data_dir = '/Users/maansidesai/Box/MovieTrailersTask/Data/EEG/Participants/event_files/'

# 	mt = '%s/trailer_phn_info.csv' %(data_dir) 
# 	mt_reader = pd.read_csv(mt,index_col=None, header=0,encoding = "ISO-8859-1")
# 	mt_reader = mt_reader.dropna(axis=1, how='all') #drop NAs that appear in columns 
# 	mt_reader.columns = ['phn', 'sample','phn_cat', 'trailer']
# 	phonemes = mt_reader['phn']

# 	index = np.empty((mt_reader.shape[0],))
# 	for i, phon in enumerate(mt_reader['phn']):
# 		try:
# 			index[i] = assign_num[phon]
# 		except:
# 			index[i] = np.nan

# 	mt_reader['index'] = index
# 	np.savetxt('%s/trailer_phn_info.txt' %(data_dir), mt_reader , fmt='%s\t', delimiter='\t')



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

		#path:
		#datadir = '/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis'

		# if stimuli == 'TIMIT':
		# 	timit_dir = '%s/Stimuli/TIMIT'%(datadir)
		# 	read = '%s/TIMIT_phn_info.csv' %(timit_dir)     
		# 	reader = pd.read_csv(read,index_col=None, header=0,encoding = "ISO-8859-1")
		# 	reader = reader.dropna(axis=1, how='all') #drop NAs that appear in columns 
		# 	reader.columns = ['idx_sent', 'idx_phn', 'sample', 'phn', 'phn_cat', 'sentence']

		# 	phonemes = reader['phn']
		# 	index = np.empty((reader.shape[0],))
		# 	for i, phon in enumerate(reader['phn']):
		# 		try:
		# 			index[i] = assign_num[phon]
		# 		except:
		# 			index[i] = np.nan
		# 	reader['index'] = index
		# 	np.savetxt('%s/TIMIT_phn_info_index.txt' %(timit_dir, subject), reader , fmt='%s\t', delimiter='\t')

		#elif stimuli == 'MovieTrailers':
		#trailer_dir = '%s/EEG/MovieTrailers/Participants/%s/downsampled_128'%(datadir,subject)
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

		# else:
		# 	print('Could not detect correct input')



		


