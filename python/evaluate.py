# Create a function to evaluate a single model with post-processing
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from models import models, feature_cols
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import itertools



def predict_test(model, test_df):
    """
    Predict the test set using the trained model and ensure one root per sentence
    Returns a new DataFrame with sentence_id, vertex, and root prediction
    Parameters:
    -----------
    model : trained classifier model
        The model used for prediction
    test_df : pandas DataFrame
        Test data with features and sentence_ids
    Returns:
    --------
    result_df : pandas DataFrame
        DataFrame with language_code, sentence_id, vertex, and is_root columns
    """
    # Extract features
    X_test = test_df[feature_cols]
    id=test_df['id'].values
    language = test_df['language'].values
    language_codes = test_df['language_code'].values
    sentence_ids = test_df['sentence_id'].values
    vertices = test_df['vertex'].values

    print(model.predict_proba(X_test))
    # Get raw probability predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Initialize final predictions as all zeros (non-roots)
    y_pred_final = np.zeros(len(X_test))

    # Create a combined key for uniqueness: language_code + sentence_id
    combined_keys = np.array([f"{lang}_{sid}" for lang, sid in zip(language_codes, sentence_ids)])

    # Group by the combined key and find the vertex with highest probability per unique combination
    unique_keys = np.unique(combined_keys)

    for unique_key in unique_keys:
        # Get indices for this language_code + sentence_id combination
        key_indices = np.where(combined_keys == unique_key)[0]

        # Get probabilities for this combination
        key_probs = y_pred_proba[key_indices]

        # Find the index with the highest probability within this combination
        max_prob_idx = key_indices[np.argmax(key_probs)]

        # Mark this vertex as the root (1)
        y_pred_final[max_prob_idx] = 1

    # Create a new DataFrame with only the required columns
    result_df = pd.DataFrame({
        'id': id,
        'language': language,
        'language_code': language_codes,
        'sentence_id': sentence_ids,
        'vertex': vertices,
#        'raw_root': test_df['is_root'].values,
        'is_root': y_pred_final.astype(int)
    })

    return result_df



def evaluate_model(name, model, X_train, y_train, X_test, y_test, test_idx, enhanced_df):
    """
    Train and evaluate a model with post-processing to ensure one root per sentence
    """
    print(f"\n{'='*50}")
    print(f"Evaluating {name}")
    print(f"{'='*50}")

    # Train the model
    model.fit(X_train, y_train)

    # Make raw predictions
    y_pred_raw = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of being a root

    # Evaluate raw model performance
    print("\nRaw Classification Report (before ensuring one root per sentence):")
    print(classification_report(y_test, y_pred_raw))

    # Post-processing to ensure only one root per sentence
    # Create a DataFrame with test indices, sentence_ids, and prediction probabilities
    test_data = pd.DataFrame({
        'idx': test_idx,
        'sentence_id': enhanced_df.iloc[test_idx]['sentence_id'].values,
        'vertex': enhanced_df.iloc[test_idx]['vertex'].values,
        'true_label': y_test,
        'pred_proba': y_pred_proba
    })

    # For each sentence, select only the vertex with highest probability as root
    final_predictions = []
    root_indices = []  # to keep track of which indices are predicted as roots

    # Process each sentence group
    for sentence_id in test_data['sentence_id'].unique():
        # Get data for this sentence
        sentence_data = test_data[test_data['sentence_id'] == sentence_id]

        # Find the index with maximum probability
        max_prob_idx = sentence_data['pred_proba'].idxmax()

        # For evaluation, mark this vertex as root (1) and others as non-root (0)
        for idx in sentence_data.index:
            if idx == max_prob_idx:
                final_predictions.append(1)
                root_indices.append(idx)
            else:
                final_predictions.append(0)

    # Convert to numpy array for evaluation
    final_predictions = np.array(final_predictions)

    # Evaluate post-processed model performance
    # print("\nPost-processed Classification Report (one root per sentence):")
    # print(classification_report(y_test, final_predictions))

    # Return metrics for comparison
    raw_report = classification_report(y_test, y_pred_raw, output_dict=True)
    post_report = classification_report(y_test, final_predictions, output_dict=True)

    return {
        'name': name,
        'raw_f1': raw_report['1']['f1-score'],
        'post_f1': post_report['1']['f1-score'],
        'raw_precision': raw_report['1']['precision'],
        'post_precision': post_report['1']['precision'],
        'raw_recall': raw_report['1']['recall'],
        'post_recall': post_report['1']['recall']
    }


# Main script to evaluate all models
def evaluate_all_models(X_train_resampled, y_train_resampled, X_test_scaled, y_test, test_idx, enhanced_df):
    results = []

    # Evaluate all models
    for name, model in models:
        result = evaluate_model(name, model, X_train_resampled, y_train_resampled,
                              X_test_scaled, y_test, test_idx, enhanced_df)
        results.append(result)

    # Create a summary DataFrame
    results_df = pd.DataFrame(results)

    # Sort by post-processed F1 score
    results_df = results_df.sort_values(by='post_f1', ascending=False)

    print("\n\nSummary of Model Performance (sorted by post-processed F1 score):")
    print(results_df)

    # Visualize results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    # Plot F1 scores
    plt.subplot(1, 3, 1)
    plt.barh(results_df['name'], results_df['post_f1'], color='blue', alpha=0.7, label='Post-processed')
    plt.barh(results_df['name'], results_df['raw_f1'], color='lightblue', alpha=0.7, label='Raw')
    plt.xlabel('F1 Score')
    plt.title('F1 Score by Model')
    plt.legend()

    # Plot Precision
    plt.subplot(1, 3, 2)
    plt.barh(results_df['name'], results_df['post_precision'], color='green', alpha=0.7, label='Post-processed')
    plt.barh(results_df['name'], results_df['raw_precision'], color='lightgreen', alpha=0.7, label='Raw')
    plt.xlabel('Precision')
    plt.title('Precision by Model')
    plt.legend()

    # Plot Recall
    plt.subplot(1, 3, 3)
    plt.barh(results_df['name'], results_df['post_recall'], color='red', alpha=0.7, label='Post-processed')
    plt.barh(results_df['name'], results_df['raw_recall'], color='lightcoral', alpha=0.7, label='Raw')
    plt.xlabel('Recall')
    plt.title('Recall by Model')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return results_df


