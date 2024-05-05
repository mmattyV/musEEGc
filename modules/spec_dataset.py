from modules.constants import *  # Assumes necessary constants and modules are defined in this file.

class SpectroDataset(Dataset):
    """
    A dataset class that handles EEG data, targeting machine learning applications.
    It categorizes and encodes EEG recordings based on a combination of enjoyment and familiarity scores.
    """
    def get_target(self, row):
        """
        Encode the labels into four categories based on enjoyment and familiarity levels.
        Args:
        - row: A list or array with elements where index 2 is enjoyment and index 3 is familiarity.
        
        Returns:
        - Tuple of (class index, class description string).
        """
        # Categories are encoded based on the threshold value 2.5
        if row[2] >= 2.5 and row[3] >= 2.5:
            return 0, 'HEHF'  # High Enjoyment, High Familiarity
        elif row[2] >= 2.5 and row[3] < 2.5:
            return 1, 'HELF'  # High Enjoyment, Low Familiarity
        elif row[2] < 2.5 and row[3] >= 2.5:
            return 2, 'LEHF'  # Low Enjoyment, High Familiarity
        else:
            return 3, 'LELF'  # Low Enjoyment, Low Familiarity
        
    def __init__(self, data_dir, behav_file, transform=None, target_transform=None):
        """
        Initialize the dataset object, load and preprocess data.
        Args:
        - data_dir: Directory containing EEG data files.
        - behav_file: Path to a CSV file with behavioral data.
        - transform: Optional transformation to apply to the EEG data.
        - target_transform: Optional transformation to apply to the target labels.
        """
        self.data_dir = data_dir
        self.behav_file = behav_file
        self.transform = transform
        self.target_transform = target_transform
        self.data_dict = {}
        eeg_label_dict = {}
        self.class_counts = {}

        tags = ['HEHF', 'HELF', 'LEHF', 'LELF']
        for tag in tags:
            self.class_counts[tag] = 0  # Initialize class count for each category

        df = pd.read_csv(self.behav_file)
        behav_data = df.values  # Load behavioral data

        progress_bar = tqdm(total=len(behav_data))

        id = 0
        for row in behav_data:
            existing_files = set(os.listdir(self.data_dir))

            data_path = f'pre_eeg_sub-0{row[0]//10}{row[0]%10}_ses-{row[1]//10}{row[1]%10}_eeg.fif'
            if data_path in existing_files:
                data_path = os.path.join(self.data_dir, data_path)
                full_data = mne.io.read_raw_fif(data_path, preload=False)

                # Splitting full EEG recording into 1-second slices
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
        """
        Get the counts of each class.
        Returns:
        - Dictionary mapping class indices to their counts.
        """
        enum_class_count = {}
        i = 0
        for _, count in self.class_counts.items():
            enum_class_count[i] = count
            i += 1
        return enum_class_count
    
    def get_label(self, idx):
        """
        Get the label for a given index.
        Args:
        - idx: Index in the dataset.
        
        Returns:
        - Label associated with the given index.
        """
        return self.items[idx][1]

    def __len__(self):
        """
        Get the length of the dataset.
        Returns:
        - Integer, half the number of items (assumes some form of data reduction or selection).
        """
        return len(self.items) // 2
    
    def __getitem__(self, idx):
        """
        Retrieve an item by index.
        Args:
        - idx: Index of the item.
        
        Returns:
        - Tuple of (processed EEG data, label).
        """
        label = self.items[idx][1]
        eeg_index = self.data_dict[self.items[idx][0]]

        full_data = mne.io.read_raw_fif(eeg_index[0], preload=False)
        # Define and compute time-frequency representation using Morlet wavelets
        frequencies = np.logspace(np.log10(1), np.log10(40), num=20)
        n_cycles = frequencies / 8.  # Define the number of cycles per frequency
        power = tfr_morlet(full_data, freqs=frequencies, n_cycles=n_cycles, use_fft=True, return_itc=False, decim=6, n_jobs=-1)

        eeg_data = power.get_data()[:, :, eeg_index[1] : eeg_index[2]]

        if self.transform:
            eeg_data = self.transform(eeg_data)
        if self.target_transform:
            label = self.target_transform(label)

        eeg_data = eeg_data.permute(1, 2, 0)  # Adjust dimensions for further processing

        # Reshape to merge time bins for reduced data representation
        eeg_data = eeg_data.reshape(eeg_data.shape[0], eeg_data.shape[1], eeg_data.shape[2] // 2, 2)
        eeg_data = eeg_data.mean(dim=3)  # Compute the mean across the last dimension to average adjacent bins

        new_shape = (eeg_data.shape[0], eeg_data.shape[1] * eeg_data.shape[2])
        eeg_data = eeg_data.reshape(new_shape)  # Reshape to final required shape

        return eeg_data, label
