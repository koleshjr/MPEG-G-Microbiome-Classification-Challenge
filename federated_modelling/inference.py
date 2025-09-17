import pandas as pd 
import numpy as np
import xgboost as xgb 
from sklearn.preprocessing import LabelEncoder

from task import load_data
le = LabelEncoder()

TEST_PATH = "Data/test_with_formatted_id.csv"
TRAIN_PATH = "Data/train_with_5_folds_gkf.csv"
MODEL_PATH = "Models/final_global_model_500_5_bagging_gkf.json"


def load_and_transform_test_data(test_path: str):
    """Load and preprocess the test dataset."""
    test_df = pd.read_csv(test_path)
    stat_cols = ["num_reads", "avg_read_len", "gc_content", "q20_fraction", "q30_fraction"]
    k5_cols = [col for col in test_df.columns if "k5" in col]

    features = stat_cols + k5_cols + ["A", "T", "G", "C"]
    X_test = test_df[features].values
    dtest = xgb.DMatrix(X_test)
    
    return dtest, test_df['ID'].values

def load_train_data(train_path: str, fold: int):
    """Load and preprocess the training dataset."""
    df = pd.read_csv(train_path)
    df = df[df["fold"] == fold].reset_index(drop=True)

    target_col = "SampleType"
    stat_cols = ["num_reads", "avg_read_len", "gc_content", "q20_fraction", "q30_fraction"]
    k5_cols = [col for col in df.columns if "k5" in col]

    features = stat_cols + k5_cols + ["A", "T", "G", "C"]
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    
    X = df[features].values
    y = df[target_col].values
    dmatrix = xgb.DMatrix(X, label=y)
    
    return dmatrix, le.classes_

def evaluate_model(model_path: str, train_path: str):
    """Load the model and evaluate it on the training data."""
    bst = xgb.Booster()
    bst.load_model(model_path)
    
    # Load training data
    dtrain, classes = load_train_data(train_path, fold=0)  # Assuming fold 0 for evaluation
    
    # Run evaluation
    eval_results = bst.eval_set(evals=[(dtrain, "train")], iteration=bst.num_boosted_rounds() - 1)
    logloss = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
    
    return logloss, classes

def predict(test_path: str,train_path: str, model_path: str):
    """Load the test data and model, then make predictions IN probs and return a sub file with ID, Mouth, Nasal, Skin, and Stool."""
    _, classes = load_train_data(train_path, fold=0)  # Assuming fold 0 for evaluation
    dtest, test_ids = load_and_transform_test_data(test_path)
    
    # Load the model
    bst = xgb.Booster()
    bst.load_model(model_path)
    
    # Make predictions
    preds = bst.predict(dtest)
    
    # Create a DataFrame for results
    results_df = pd.DataFrame(preds, columns=classes)

    results_df['ID'] = test_ids
    
    return results_df[['ID'] + classes.tolist()]
def predict_fn():
    """Main function to run the prediction."""
    results_df = predict(TEST_PATH,TRAIN_PATH, MODEL_PATH)
    
    # Save results to CSV
    submission_path = f"Subs/{MODEL_PATH.split('/')[-1].split('.')[0]}.csv"
    results_df.to_csv(submission_path, index=False)
    print(f"Predictions saved to {submission_path}")

    return results_df

if __name__ == "__main__":
    logloss, classes = evaluate_model(MODEL_PATH, TRAIN_PATH)
    print(f"Logloss on training data: {logloss}")
    print(f"Classes: {classes}")
    # Run prediction
    predict_fn()