from tqdm import tqdm
import itertools
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from evaluate import evaluate_model, evaluate_all_models, predict_test
from models import model_class, feature_cols
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis



class PostProcessedF1Scorer:
    def __init__(self, groups):
        self.groups = groups

    def __call__(self, model, X_val, y_val):
        # Clone model to avoid shared state
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        df = pd.DataFrame({
            'sentence_id': self.groups.loc[X_val.index].values,
            'true_label': y_val,
            'pred_proba': y_pred_proba
        })

        # Apply one-root-per-sentence postprocessing
        final_preds = []
        for sent_id in df['sentence_id'].unique():
            sent_df = df[df['sentence_id'] == sent_id]
            max_idx = sent_df['pred_proba'].idxmax()
            preds = [1 if idx == max_idx else 0 for idx in sent_df.index]
            final_preds.extend(preds)

        report = classification_report(y_val, final_preds, output_dict=True, zero_division=0)
        return report['1']['f1-score']

def custom_group_cv_with_smote( X_train, y_train, groups, model_class, param_grid, smote_random_state=42, n_splits=1000, scoring_class=None, verbose=True):

    # Initialize SMOTE and GroupKFold
    smote = SMOTE(random_state=smote_random_state)
    group_kfold = GroupKFold(n_splits=n_splits)

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    # Initialize results tracking
    best_score = -np.inf
    best_params = None
    all_results = []

    # Loop through all parameter combinations
    for params_tuple in param_combinations:
        # Create parameter dictionary for this combination
        params = dict(zip(param_names, params_tuple))

        # Scores for this parameter set
        param_scores = []

        # Cross-validation
        for train_fold_idx, val_fold_idx in group_kfold.split(X_train, y_train, groups=groups):
            # Get fold data
            X_fold_train = X_train.iloc[train_fold_idx]
            y_fold_train = y_train.iloc[train_fold_idx]
            X_fold_val = X_train.iloc[val_fold_idx]
            y_fold_val = y_train.iloc[val_fold_idx]

            # Apply SMOTE to training fold only
            X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train, y_fold_train)

            # Initialize and train model with current parameters
            model = model_class(**params)
            model.fit(X_fold_train_resampled, y_fold_train_resampled)

            # Score using custom scorer if provided
            if scoring_class is not None:
                val_fold_groups = groups.iloc[val_fold_idx]
                fold_scorer = scoring_class(groups=val_fold_groups)
                fold_score = fold_scorer(model, X_fold_val, y_fold_val)
            else:
                # Default to accuracy if no scorer provided
                y_pred = model.predict(X_fold_val)
                from sklearn.metrics import accuracy_score
                fold_score = accuracy_score(y_fold_val, y_pred)

            param_scores.append(fold_score)

        # Average score for these parameters across all folds
        avg_score = np.mean(param_scores)

        # Store results
        result = {
            'params': params,
            'avg_score': avg_score,
            'fold_scores': param_scores
        }
        all_results.append(result)

        # Keep track of best parameters
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

        if verbose:
            print(f"Params: {params}, Avg Score: {avg_score:.4f}")

    # Sort results by score (descending)
    all_results.sort(key=lambda x: x['avg_score'], reverse=True)

    if verbose:
        print("\nBest parameters:", best_params)
        print("Best cross-validation score:", best_score)

    # Train final model with best parameters on full training set
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train final model
    best_model = model_class(**best_params)
    best_model.fit(X_train_resampled, y_train_resampled)

    return {
        'best_model': best_model,
        'model_class': model_class,
        'best_params': best_params,
        'best_cv_score': best_score,
        'all_results': all_results
    }

def cross_validation(X_train, y_train, X_test, y_test, train_idx, test_idx, model_class, groups, df):
    results = {}

    for model_name in tqdm(model_class):
        print(model_name)
        class_model = model_class[model_name]
        model_c = class_model['model_class']
        param_grid = class_model['param_grid']

        best = custom_group_cv_with_smote(
            X_train, y_train, groups.iloc[train_idx],
            model_c, param_grid, n_splits=5, scoring_class=PostProcessedF1Scorer
        )
        results[model_name] = best


    # Sort and get top 3 by post_recall
    sorted_models = sorted(results.items(), key=lambda x: x[1]['best_cv_score'], reverse=True)
    top3_results = {model: metrics for model, metrics in sorted_models[:3]}

    # Print
    print("\nTop 3 models based on post-processed score:")
    for i, (model_name, info) in enumerate(top3_results.items(), 1):
        print(f"{i}. {model_name}")
        print(f"   Model class: {info['model_class']}")
        print(f"   Post F1: {info['best_cv_score']}")
        print(f"   Best Params: {info['best_params']}")

    return top3_results

if __name__ == '__main__':
    # Load data
    enhanced_df_raw = pd.read_csv('../python_data_pre/engineered_features_train_nor_s.csv')
    enhanced_df = enhanced_df_raw[enhanced_df_raw['language_code'].isin([17])]


    X = enhanced_df[feature_cols]
    y = enhanced_df['is_root']

    # Split data into training and testing sets
    groups = enhanced_df['sentence_id']


    # Split data into training and testing sets
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # # Standardize features
    # scaler = StandardScaler()
    #
    # #
    # # Fit scaler on training data
    # X_train_np = scaler.fit_transform(X_train)
    # X_test_np = scaler.transform(X_test)
    # X_train_scaled = pd.DataFrame(X_train_np, index=X_train.index, columns=X_train.columns)
    # X_test_scaled = pd.DataFrame(X_test_np, index=X_test.index, columns=X_test.columns)
    #
    # # Apply SMOTE to balance the training data
    # smote = SMOTE(random_state=42)
    # X_train_resampled_scaled, y_train_resampled_scaled = smote.fit_resample(X_train_scaled, y_train)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    top3 = cross_validation(X_train, y_train, X_test, y_test, train_idx, test_idx, model_class, groups, enhanced_df)

    for model_name in top3:
        model_class_test = top3[model_name]['model_class']
        model_params = top3[model_name]['best_params']
        model_full = model_class_test(**model_params)
        print(evaluate_model(model_name, model_full, X_train_resampled, y_train_resampled, X_test, y_test, test_idx, enhanced_df))
        # print(evaluate_model(model_name, model_full, X_train_resampled_scaled, y_train_resampled_scaled, X_test, y_test, test_idx,
        #                enhanced_df))

    # best=custom_group_cv_with_smote(X_train, y_train, groups.iloc[train_idx], QuadraticDiscriminantAnalysis,
    #      {
    #         'reg_param': [0.0, 0.1, 1.0],
    #         'tol': [1e-4, 1e-3, 1e-2],
    #         'store_covariance': [True, False]
    #     }, n_splits=5, scoring_class=PostProcessedF1Scorer)
    #
    # model_c=QuadraticDiscriminantAnalysis
    # model_name='qda'
    #
    # all = []
    # all.append(
    #     evaluate_model(model_name, model_c(**best['best_params']), X_train_resampled_scaled, y_train_resampled_scaled,
    #                    X_test, y_test, test_idx, enhanced_df)['post_recall'])
    # all.append(
    #     evaluate_model(model_name, model_c(**best['best_params']), X_train_resampled, y_train_resampled, X_test,
    #                    y_test, test_idx, enhanced_df)['post_recall'])
    #
    # print(all)
    #
    # print(evaluate_model('ada boost', QuadraticDiscriminantAnalysis(reg_param=1.0, tol=0.0001, store_covariance=True),
    #                X_train_resampled, y_train_resampled, X_test, y_test, test_idx, enhanced_df))
