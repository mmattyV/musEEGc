from modules.constants import *  # Assuming this contains necessary constants for the module.

class RawDataset(Dataset):
    """
    A PyTorch Dataset class that processes raw EEG data files and associated behavioral scores.
    It categorizes data based on combinations of enjoyment and familiarity ratings into four bins:
    HEHF (High Enjoyment & High Familiarity),
    HELF (High Enjoyment & Low Familiarity),
    LEHF (Low Enjoyment & High Familiarity),
    LELF (Low Enjoyment & Low Familiarity).
    """

    # Bin and hot encode our labels for our targets
    # Bins: [high familiarity & high enjoyment,
    #        high familiarity & low enjoyment,
    #        low familiarity & high enjoyment,
    #        low familiarity & low enjoyment]
    # High is >= 2.5
    # Low is < 2.5
    def get_target(self, row):
        """
        Determine the target classification based on enjoyment and familiarity ratings.
        Args:
        - row: A row from the behavioral data file containing ratings.
        
        Returns:
        - Tuple (target class index, target class string)
        """
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
        """
        Initialize the dataset by loading and preprocessing the data.
        Args:
        - data_dir: Directory where EEG data files are stored.
        - behav_file: File path for the CSV file containing behavioral data.
        - transform: Optional transform to be applied on a sample.
        - target_transform: Optional transform to be applied on the label.
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
            self.class_counts[tag] = 0

        df = pd.read_csv(self.behav_file)
        behav_data = df.values
    
        progress_bar = tqdm(total=len(behav_data))

        id = 0
        for row in behav_data:
            existing_files = set(os.listdir(self.data_dir))

            data_path = f'pre_eeg_sub-0{row[0]//10}{row[0]%10}_ses-{row[1]//10}{row[1]%10}_eeg.fif'
            if data_path in existing_files:
                data_path = os.path.join(self.data_dir, data_path)
                full_data = mne.io.read_raw_fif(data_path, preload=False)

                # Splitting full EEG recording into 5 second slices
                num_intervals = full_data.get_data().shape[1] // 1250
                for i in range(num_intervals):
                    slice = [data_path, i*1250, i*1250+1250]
                    
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
        Returns a dictionary mapping class labels to their counts.
        """
        enum_class_count = {}
        i = 0
        for _, count in self.class_counts.items():
            enum_class_count[i] = count
            i += 1
        return enum_class_count
    
    def get_label(self, idx):
        """
        Retrieve the label for a specific index.
        Args:
        - idx: Index for which to retrieve the label.
        
        Returns:
        - Label corresponding to the index.
        """
        return self.items[idx][1]

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return (len(self.items) // 2)  # 1/2 the amount of data
    
    def __getitem__(self, idx):
        """
        Retrieve a sample and its label from the dataset.
        Args:
        - idx: Index of the sample to retrieve.
        
        Returns:
        - Tuple (EEG data, label)
        """
        label = self.items[idx][1]
        eeg_index = self.data_dict[self.items[idx][0]]

        full_data = mne.io.read_raw_fif(eeg_index[0], preload=False)
        eeg_data = full_data.get_data()[:, eeg_index[1] : eeg_index[2]]

        if self.transform:
            eeg_data = self.transform(eeg_data)
        if self.target_transform:
            label = self.target_transform(label)

        return eeg_data[0], label
