import numpy as np
import os
import mne
from mne.preprocessing import ICA
from mne import pick_types
from mne.io import read_raw_eeglab
from mne.time_frequency import psd_array_welch
from mne.time_frequency import tfr_morlet
import torch
import multiprocessing
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, WeightedRandomSampler
import json 
from tqdm import tqdm
import time
from torch.autograd import Variable
import copy
import pandas as pd
from sklearn.decomposition import PCA
import logging

logging.getLogger('mne').setLevel(logging.WARNING)

num_sub = 20
num_sess = 12
cuda_device = 0
train_dir = '../prepro_data/train'
val_dir = '../prepro_data/val'
train_behav_file = 'labels/train_behav.csv'
val_behav_file = 'labels/val_behav.csv'
base_lr = 0.0001
decay_weight = 0.1 
epoch_decay = 5 
b_size = 5
n_epochs = 10