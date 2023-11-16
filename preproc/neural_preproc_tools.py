import mne


# Loading the data 
user = 'maansidesai'
datadir = f'/Users/{user}/Box/MovieTrailersTask/Data/EEG/Participants/'
subject = input('Input subject ID: ')
block = input('Input block number: ')
block = int(block)
#path  = '/Users/alyssa/Library/CloudStorage/Box-Box/FBS_ICAprocessing/FBS10/Session_1/20220401_FBS010_Session_1_DS128.vhdr'  #yourdata path
raw   = mne.io.read_raw_brainvision(f'{datadir}/{subject}/downsampled_128/{subject}_B{block}_DS128.vhdr',preload=True)


# to see the related information of data
raw.info   

raw.plot_psd()   
raw.notch_filter(60)
raw.set_eeg_reference(['TP9','TP10']) 

# filter data from 1 hz to None
raw.filter(l_freq =1,h_freq=None) 


#copy the raw data 
raw_copy = raw.copy() 

#filter data from 1 Hz to None
raw_copy.filter(l_freq =1,h_freq=15)   

raw.plot() #for manual artifact rejection

# Save annotated data
raw.save(f'{datadir}/{subject}/downsampled_128/{subject}_B{block}_rejection_mas_raw.fif', overwrite=True) 

# Save annotated data
raw_copy.save(f'{datadir}/{subject}/downsampled_128/{subject}_B{block}_rejection_mas_raw-1Hzto15.fif', overwrite=True) 



montage = mne.channels.read_custom_montage(f'{datadir}/montage/AP-128.bvef')


picks = mne.pick_types(raw.info,eeg=True,meg=False,eog=False)


#ica = mne.preprocessing.ica(n_components=len(picks),method ='infomax',random_state=12345)
#ica= mne.preprocessing.ICA(n_components=len(picks),method ='infomax')
ica = mne.preprocessing.ICA(n_components=len(picks), random_state=97, max_iter=800)


ica.fit(raw_copy, picks = picks,reject_by_annotation=True)
ica.plot_components()
ica.plot_sources(raw)


veog_epochs = mne.preprocessing.create_eog_epochs(raw,ch_name ='vEOG',tmin=-0.5,tmax=0.5,l_freq=1,h_freq=10)
heog_epochs = mne.preprocessing.create_eog_epochs(raw,ch_name ='hEOG',tmin=-0.5,tmax=0.5,l_freq=1,h_freq=10)


ica.plot_properties(veog_epochs)
ica.plot_properties(heog_epochs)

veog_inds,scores = ica.find_bads_eog(veog_epochs, ch_name ='vEOG',threshold=3.0)
heog_inds,scores = ica.find_bads_eog(heog_epochs, ch_name ='hEOG',threshold=3.0)


ica.plot_properties(veog_epochs,picks=veog_inds)
ica.plot_properties(heog_epochs,picks=heog_inds)
ica.plot_components()
# ica.plot_properties(veog_epochs,picks=[8]) #For example you can choose to see the IC0008 here
# ica.plot_properties(heog_epochs,picks=[8]) #For example you can choose to see the IC0008 here

ica.apply(raw_copy)
ica.apply(raw)
ica.save(f'{datadir}/{subject}/downsampled_128/{subject}_B{block}_ICA.fif')


raw.save(f'{datadir}/{subject}/downsampled_128/{subject}_B{block}_postICA_rejected.fif', overwrite=True) #non filtered
raw_copy.save(f'{datadir}/{subject}/downsampled_128/{subject}_B{block}_postICA_rejected_1-15Hz.fif', overwrite=True) #this is 1-15Hz filtered