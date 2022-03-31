BASE_PATH = "../input/mens-march-mania-2022/MDataFiles_Stage2/"

RESULTS_NAME = "MNCAATourneyDetailedResults.csv"
SEED_NAME = "MNCAATourneySeeds.csv"
RANK_NAME = "MMasseyOrdinals_thruDay128.csv"

TEST_NAME = "MSampleSubmissionStage2.csv"

model_type = 'Autolgbm'  
message='baseline'

train_filename = "./train.csv"
output = "output"
test_filename = "./test.csv"
task = None
targets = ['target']
features = None
categorical_features = None
use_gpu = False
num_folds = 10
seed = 42
num_trials = 10000
time_limit = 7200
fast = False

TEST = False

#Imports
import pandas as pd
import numpy as np
import random
import time
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import matplotlib.pylab as plt
import seaborn as sns
from autolgbm import AutoLGBM



