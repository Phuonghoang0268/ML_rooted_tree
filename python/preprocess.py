from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
from models import  feature_cols


def over_resample(enhanced_df):

    X = enhanced_df[feature_cols]
    y = enhanced_df['is_root']

    # Split data into training and testing sets
    groups = enhanced_df['sentence_id']

    # Split data into training and testing sets
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Print class distribution before resampling
    print("Class distribution before resampling:")
    print(pd.Series(y_train).value_counts())

    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled, X_test, y_test, train_idx, test_idx



