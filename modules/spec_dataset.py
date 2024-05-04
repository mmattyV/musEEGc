from modules.constants import *

class SpectroDataset(Dataset):
    # Bin and hot encode our labels for our targets
    # Bins: [high familiarity & high enjoyment, 
    #        high familiarity & low enjoyment, 
    #        low familiarity & high enjoyment, 
    #        low familiarity & low enjoyment]
    # High is >= 2.5
    # Low is < 2.5
    def get_target(self, row):
        # HEHF
        if row[2] >= 2.5 and row[3] >= 2.5:
            return 0, 'HEHF'
        # HELF
        elif row[2] >= 2.5 and row[3] < 2.5:
            return 1, 'HELF'
        # LEHF
        elif row[2] < 2.5 and row[3] >= 2.5:
            return 2, 'LEHF'
        # LELF
        else:
            return 3, 'LELF'
        
    def __init__(self, data_dir, behav_file, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.behav_file = behav_file
        self.transform = transform
        self.target_transform = target_transform
        self.data_dict = {}

        eeg_label_dict = {}
        self.class_counts = {}

        tags = ['HEHF', 'HELF', 'LEHF', 'LELF']
        
        for tag in tags:
            self.class_counts[tag] = 0

        df = pd.read_csv(self.behav_file)
        behav_data = df.values

        total_files = 0
        for entry in os.listdir(self.data_dir):
            # Join the directory path with the entry name to get full file path
            full_path = os.path.join(self.data_dir, entry)
            if os.path.isfile(full_path):
                total_files += 1
    
        progress_bar = tqdm(total=len(behav_data))

        id = 0
        for row in behav_data:
            existing_files = set(os.listdir(self.data_dir))

            data_path = f'pre_eeg_sub-0{row[0]//10}{row[0]%10}_ses-{row[1]//10}{row[1]%10}_eeg.fif'
            if data_path in existing_files:
                data_path = os.path.join(self.data_dir, data_path)
                full_data = mne.io.read_raw_fif(data_path, preload=False)

                # Splitting full EEG recording into 1 second slices
                num_intervals = (full_data.get_data().shape[1] // 3) // (84*5)
                for i in range(num_intervals):
                    slice = [data_path, i*84, i*84+84]
                    
                    target, tag_string = self.get_target(row)

                    self.data_dict[id] = slice

                    eeg_label_dict[id] = target
                    self.class_counts[tag_string] += 1
                    id += 1

            progress_bar.update(1)
        
        progress_bar.close()

        self.items = list(eeg_label_dict.items())
        print('Class counts: ', self.class_counts)

    def get_class_counts(self):
        enum_class_count = {}
        i = 0
        for _, count in self.class_counts.items():
            enum_class_count[i] = count
            i += 1
        return enum_class_count
    
    def get_label(self, idx):
        return self.items[idx][1]

    def __len__(self):
        return (len(self.items) // 2) # 1/2 the amount of data
    
    def __getitem__(self, idx):
        label = self.items[idx][1]
        eeg_index = self.data_dict[self.items[idx][0]]

        full_data = mne.io.read_raw_fif(eeg_index[0], preload=False)
        # Define frequencies of interest (log-spaced)
        frequencies = np.logspace(np.log10(1), np.log10(40), num=20)
        n_cycles = frequencies / 8.  # Different number of cycle per frequency
        # Compute time-frequency representation with Morlet wavelets
        power = tfr_morlet(full_data, freqs=frequencies, n_cycles=n_cycles, use_fft=True, return_itc=False, decim=6, n_jobs=-1)

        eeg_data = power.get_data()[:, :, eeg_index[1] : eeg_index[2]]

        if self.transform:
            eeg_data = self.transform(eeg_data)
        if self.target_transform:
            label = self.target_transform(label)

        eeg_data = eeg_data.permute(1, 2, 0)

        # eeg_data has shape (129, 40, 84), we need to reduce the time bins to 42
        # Reshape eeg_data to merge each pair of time bins
        eeg_data = eeg_data.reshape(eeg_data.shape[0], eeg_data.shape[1], eeg_data.shape[2] // 2, 2)

        # Compute the mean across the last dimension to average adjacent bins
        eeg_data = eeg_data.mean(dim=3)

        new_shape = (eeg_data.shape[0], eeg_data.shape[1] * eeg_data.shape[2])  # (129, 20*42)
        eeg_data = eeg_data.reshape(new_shape)

        return eeg_data, label
        # eeg_data has shape (129, 840)