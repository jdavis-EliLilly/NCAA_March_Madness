

#Config
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
if TEST:
    time_limit = 1000



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

#Load Team Data results
df_TDresults = pd.read_csv(BASE_PATH + RESULTS_NAME)


'''
------------------------------
Creating a train dataset from 
MNCAATourneyDetailedResults (previous tourney results)
------------------------------
'''

#Create a df for features, 
#FGM - Field Goals Made (Shots made?)
#FGA - Field Goals Attempted (Shots total?)
#FGM3 - 3s made
#FTM - Free throws made FTA - Attempted
#OR - Offensive Rebounds DR - Defensive Rebounds
#Ast - Assists TO - Turnovers
#Stl - Steals Blk - Blocks
#Pf - No clue

feat_col = ["FGM","FGA","FGM3","FTM","FTA","OR","DR","Ast","TO","Stl","Blk","PF"]

#Win loss per game to double data per team (may overfit but I want
# a loss features game and a win features game per team per game)
df_boxW = df_TDresults[["Season","WTeamID"]+["W" + col for col in feat_col]]
df_boxL = df_TDresults[["Season","LTeamID"]+["L" + col for col in feat_col]]
df_boxW = df_boxW.rename(columns={"WTeamID":"TeamID"})
df_boxW = df_boxW.rename(columns={("W"+ col):col for col in feat_col})
df_boxL = df_boxL.rename(columns={"LTeamID":"TeamID"})
df_boxL = df_boxL.rename(columns={("L"+ col):col for col in feat_col})
df_box = pd.merge(df_boxW,df_boxL,on = ["Season","TeamID"]+feat_col,how="outer")
df_box = df_box.groupby(["Season","TeamID"])[feat_col].agg(np.mean).reset_index() #agg mean per team Id per season


#Cleaning data and housekeeping for Win loss
#per feature column in df
df_TDresults2 = df_TDresults
df_TDresults = df_TDresults.rename(columns={"WTeamID":"Team1ID","LTeamID":"Team2ID","WScore":"T1Score","LScore":"T2Score"})
df_TDresults = df_TDresults.rename(columns={f"W{col}":f"T1{col}" for col in feat_col})
df_TDresults = df_TDresults.rename(columns={f"L{col}":f"T2{col}" for col in feat_col})
df_TDresults2 = df_TDresults2.rename(columns={"WTeamID":"Team2ID","LTeamID":"Team1ID","WScore":"T2Score","LScore":"T1Score"})

#If team wins give a 1 if losing give 0
features = ["Season","Team1ID","Team2ID","T1Score","T2Score",'target']
df_TDresults['target'] = 1.0
df_TDresults2['target'] = 0.0

#Merge to create a train dataset on past tourney results 
train = pd.merge(df_TDresults,df_TDresults2,on = features,how="outer")
train = train[features]
box_T1 = df_box.copy()
box_T2 = df_box.copy()
box_T1.columns = ['Season','Team1ID'] + ["T1"+col+"_mean" for col in feat_col]
box_T2.columns = ['Season','Team2ID'] + ["T2"+col+"_mean" for col in feat_col]
train = pd.merge(train,box_T1,on = ["Season","Team1ID"],how = "left")
train = pd.merge(train,box_T2,on = ["Season","Team2ID"],how = "left")

'''
------------------------------
Creating a seed dataset as a feature
------------------------------
'''
#Creating Seeding dataframe
# Whomever cretes the "seeding"
#knows more than I do so I will use
#it as a feature
df_seeds = pd.read_csv(BASE_PATH + SEED_NAME)

#Adding seed features to our train dataset
df_seeds["seed"] = df_seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds_T1 = df_seeds[['Season','TeamID','seed']].copy()
seeds_T2 = df_seeds[['Season','TeamID','seed']].copy()
seeds_T1.columns = ['Season','Team1ID','T1_seed']
seeds_T2.columns = ['Season','Team2ID','T2_seed']
train = pd.merge(train,seeds_T1,on = ["Season","Team1ID"],how = "left")
train = pd.merge(train,seeds_T2,on = ["Season","Team2ID"],how = "left")
train["seeddiff"] = train["T1_seed"] - train["T2_seed"]


'''
------------------------------
Creating a ranking dataset as a feature
from ordinal
------------------------------
'''
df_MMOrdinals = pd.read_csv(BASE_PATH + RANK_NAME)
df_rank = df_MMOrdinals.groupby(["Season","TeamID"])["OrdinalRank"].agg(np.mean).reset_index()
ranks_T1 = df_rank.copy()
ranks_T2 = df_rank.copy()
ranks_T1.columns = ['Season','Team1ID','T1_rank_mean']
ranks_T2.columns = ['Season','Team2ID','T2_rank_mean']
train = pd.merge(train,ranks_T1,on = ["Season","Team1ID"],how = "left")
train = pd.merge(train,ranks_T2,on = ["Season","Team2ID"],how = "left")
train["rankdiff"] = train["T1_rank_mean"] - train["T2_rank_mean"]
train = train.drop(columns = ["T1Score","T2Score"])

'''
------------------------------
Formatting test data to resemble 
train data
------------------------------
'''
#Load test data provided
test = pd.read_csv(BASE_PATH + TEST_NAME)
test["Season"] = test['ID'].apply(lambda x: int(x[0:4]))
test["Team1ID"] = test['ID'].apply(lambda x: int(x[5:9]))
test["Team2ID"] = test['ID'].apply(lambda x: int(x[10:14]))
box_T1['Season'] = box_T1['Season'] + 1
box_T2['Season'] = box_T2['Season'] + 1
test = pd.merge(test,box_T1,on = ["Season","Team1ID"],how = "left")
test = pd.merge(test,box_T2,on = ["Season","Team2ID"],how = "left")


#Adding seed features
test["seeddiff"] = test["T1_seed"] - test["T2_seed"]
test = pd.merge(test,ranks_T1,on = ["Season","Team1ID"],how = "left")
test = pd.merge(test,ranks_T2,on = ["Season","Team2ID"],how = "left")
test["rankdiff"] = test["T1_rank_mean"] - test["T2_rank_mean"]
test = test.drop(columns = ['ID','Pred'])

#train and test dataframes to csv
train.to_csv("train.csv",index = None)
test.to_csv("test.csv",index = None)

'''
------------------------------
AutoLGBM setup
------------------------------
'''
features=None
algbm = AutoLGBM(
    train_filename=train_filename,
    output=output,
    test_filename=test_filename,
    task=task,
    targets=targets,
    features=features,
    categorical_features=categorical_features,
    use_gpu=use_gpu,
    num_folds=num_folds,
    seed=seed,
    num_trials=num_trials,
    time_limit=time_limit,
    fast=fast,
)

#Train the algo
algbm.train()


#Write the submission
submission = pd.read_csv(BASE_PATH + TEST_NAME)
autolgb_pred = pd.read_csv("./output/test_predictions.csv")
submission['Pred'] = autolgb_pred['1.0']
# submission.rename(columns = {'1.0':'Pred'}, inplace = True)
submission.to_csv("submission.csv", index=False)

#Profit

