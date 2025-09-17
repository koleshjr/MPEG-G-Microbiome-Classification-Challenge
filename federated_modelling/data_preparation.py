import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

KMERS_TRAIN_PATH = '../data_prep/Data/ProcessedFiles/train_features_with_kmers_new.csv'
KMERS_TEST_PATH = '../data_prep/Data/ProcessedFiles/test_features_with_kmers_new.csv'
ORIGINAL_TRAIN_PATH = '../data_prep/Data/Train.csv'
PROCESSED_TEST_PATH ='Data/test_with_formatted_id.csv'

train = pd.read_csv(KMERS_TRAIN_PATH)
train['ID'] = train['file'].apply(lambda x: x.split('.')[0])
train_labels = pd.read_csv(ORIGINAL_TRAIN_PATH)
train_labels['ID'] = train_labels['filename'].apply(lambda x: x.split('.')[0])
train = train.merge(train_labels[['ID', 'SampleType', 'SubjectID']], on='ID', how='left')
test = pd.read_csv(KMERS_TEST_PATH)
test['ID'] = test['file'].apply(lambda x: x.split('.')[0])

n_splits = 5
method = 'gkf'  # 'sgkf' or 'skf'
gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
train['fold'] = -1

if method == 'gkf':
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X=train, y=train['SampleType'], groups=train['SubjectID'])):
        train.loc[val_idx, 'fold'] = fold
elif method == 'skf':
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=train, y=train['SampleType'])):
        train.loc[val_idx, 'fold'] = fold

for fold in range(n_splits):
    print(train[train['fold']==fold]['SampleType'].value_counts())
    print("-"* 100)

train.to_csv(f'Data/train_with_{n_splits}_folds_{method}.csv', index=False)
test.to_csv(PROCESSED_TEST_PATH, index=False)