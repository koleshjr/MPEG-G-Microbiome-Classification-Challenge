import pandas as pd
from logging import INFO
import xgboost as xgb
from flwr.common import log 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

fds= None 


def custom_train_test_split(partition , test_fraction, seed):
    """Split the data into train and validation sets given split rate"""
    train, valid = train_test_split(partition, test_size=test_fraction, random_state=seed, stratify =partition['SampleType'])
    num_train = len(train)
    num_valid = len(valid)

    return train, valid, num_train, num_valid

def transform_dataset_to_dmatrix(dataset):
    """Transform a pandas dataframe into an xgboost DMatrix"""
    target_col = "SampleType"
    stat_cols = ["num_reads", "avg_read_len", "gc_content", "q20_fraction", "q30_fraction"]
    k5_cols = [col for col in dataset.columns if "k5" in col]

    features = stat_cols + k5_cols + ["A", "T", "G", "C"]
    le = LabelEncoder()
    dataset[target_col] = le.fit_transform(dataset[target_col])
    X = dataset[features].values
    y = dataset[target_col].values
    dmatrix = xgb.DMatrix(X, label=y)
    return dmatrix

def load_data(data_path: str, fold: int, test_fraction: float = 0.2, seed: int = 42):
    """Load and preprocess the dataset."""
    df = pd.read_csv(data_path)
    df = df[df["fold"] == fold].reset_index(drop=True)

    train_data, valid_data, num_train, num_valid = custom_train_test_split(df, test_fraction=test_fraction, seed=seed)
    dtrain = transform_dataset_to_dmatrix(train_data)
    dvalid = transform_dataset_to_dmatrix(valid_data)
    return dtrain, dvalid, num_train, num_valid

def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
