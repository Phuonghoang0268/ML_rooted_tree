import csv
import pandas as pd
import matplotlib as plt
import  numpy as np
import seaborn as sns
import networkx as nx
from scipy.stats import zscore

def centralities(edgelist):
    """
    - edgelist is a list of node pairs e.g. [(7,2),(1,7),(1,9),...]
    - returns a dictionary of vertex -> (centrality values)
    """
    T = nx.from_edgelist(edgelist)
    dc = nx.degree_centrality(T)
    cc = nx.harmonic_centrality(T)
    bc = nx.betweenness_centrality(T)
    pc = nx.pagerank(T)

    # Additional centralities that might be useful
    ec = nx.eigenvector_centrality(T, max_iter=10000)  # Eigenvector centrality
    kc = nx.katz_centrality(T, max_iter=10000)         # Katz centrality

    return {v: (dc[v], cc[v], bc[v], pc[v], ec[v], kc[v]) for v in T}


def df_central(df):
    print('getting centrality')
    import ast
    df['language_code'] = df['language'].astype('category').cat.codes
    all_features = []
    all_targets = []
    all_sentence_ids = []
    for i, row in df.iterrows():

        sentence_id = row['sentence']
        language = row['language']
        language_code = row['language_code']
        n = row['n']

        if isinstance(row['edgelist'], str):
            edgelist = ast.literal_eval(row['edgelist'])
        else:
            edgelist = row['edgelist']

        # Parse root from string if needed
        if isinstance(row['root'], str):
            import ast
            root = ast.literal_eval(row['root'])
        else:
            root = row['root']

        T = nx.from_edgelist(edgelist)
        cent = centralities(edgelist)
        for v in T.nodes():
            is_root = 1 if v == root else 0
            dc, cc, bc, pc, ec, kc = cent[v]

            features = [language, language_code, sentence_id, n, v, dc, cc, bc, pc, ec, kc]
            all_features.append(features)
            all_targets.append(is_root)
            all_sentence_ids.append(sentence_id)

    feature_columns = ['language', 'language_code', 'sentence_id', 'n', 'vertex', 'degree_cent', 'closeness_cent',
                       'betweenness_cent', 'pagerank_cent',
                       'eigenvector_cent', 'katz_cent']
    # Create DataFrame
    X = pd.DataFrame(all_features, columns=feature_columns)
    y = pd.Series(all_targets, name='is_root')
    sentence_ids = pd.Series(all_sentence_ids, name='sentence_id')
    combined_df = pd.concat([X, y], axis=1)
    return combined_df, sentence_ids


def df_central_without_root(df):
    print('getting centrality without root information')
    import ast
    df['language_code'] = df['language'].astype('category').cat.codes
    all_features = []
    all_sentence_ids = []

    for i, row in df.iterrows():
        id = row['id']
        sentence_id = row['sentence']
        language = row['language']
        language_code = row['language_code']
        n = row['n']

        if isinstance(row['edgelist'], str):
            edgelist = ast.literal_eval(row['edgelist'])
        else:
            edgelist = row['edgelist']

        # Create graph from edgelist
        T = nx.from_edgelist(edgelist)
        cent = centralities(edgelist)

        for v in T.nodes():
            # No root information, so we skip the is_root flag
            dc, cc, bc, pc, ec, kc = cent[v]

            features = [id, language, language_code, sentence_id, n, v, dc, cc, bc, pc, ec, kc]
            all_features.append(features)
            all_sentence_ids.append(sentence_id)

    feature_columns = ['id','language', 'language_code', 'sentence_id', 'n', 'vertex', 'degree_cent', 'closeness_cent',
                       'betweenness_cent', 'pagerank_cent',
                       'eigenvector_cent', 'katz_cent']

    # Create DataFrame without the is_root target column
    X = pd.DataFrame(all_features, columns=feature_columns)
    sentence_ids = pd.Series(all_sentence_ids, name='sentence_id')

    # Return just the feature dataframe since we don't have targets
    return X, sentence_ids

def ranking(df, sentence_ids):
    print('ranking')
    ranked_df = df.copy()
    for sentence_id, group in df.groupby('sentence_id'):
        ranked_df.loc[group.index, 'degree_rank'] = group['degree_cent'].rank(ascending=False, method='min').astype(int)
        ranked_df.loc[group.index, 'closeness_rank'] = group['closeness_cent'].rank(ascending=False,
                                                                                    method='min').astype(int)
        ranked_df.loc[group.index, 'betweenness_rank'] = group['betweenness_cent'].rank(ascending=False,
                                                                                        method='min').astype(int)
        ranked_df.loc[group.index, 'pagerank_rank'] = group['pagerank_cent'].rank(ascending=False, method='min').astype(
            int)
        ranked_df.loc[group.index, 'eigenvector_rank'] = group['eigenvector_cent'].rank(ascending=False,
                                                                                        method='min').astype(int)
        ranked_df.loc[group.index, 'katz_rank'] = group['eigenvector_cent'].rank(ascending=False, method='min').astype(
            int)
    return ranked_df



def create_graph(df, sentence_id, root):
    print('create graph')
    """
    Create a NetworkX graph for a specific sentence ID.

    Args:
        df: DataFrame with network data
        sentence_id: ID of the sentence to create a graph for

    Returns:
        NetworkX graph
    """
    # Filter dataframe for the specific sentence
    sentence_df = df[df['sentence_id'] == sentence_id]

    # Create an empty graph
    G = nx.Graph()

    if root:
        # Add nodes to the graph
        for _, row in sentence_df.iterrows():
            G.add_node(row['vertex'], is_root=row['is_root'])
    else:
        # Add nodes without root information
        for _, row in sentence_df.iterrows():
            G.add_node(row['vertex'])


    # Since we don't have edge information, we'll create a simple
    # connected graph based on vertex IDs for demonstration
    # In a real scenario, you would use actual edge data

    # For simplicity, connect each node to the next one
    nodes = list(G.nodes())
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i+1])

    # Add some additional edges to create a more interesting graph
    if len(nodes) > 2:
        # Connect first node to the last node to create a cycle
        G.add_edge(nodes[0], nodes[-1])

        # Add a few more edges
        for i in range(0, len(nodes) - 2, 2):
            G.add_edge(nodes[i], nodes[i+2])

    return G

def engineer_graph_features(df, G=None):
    print('engineer graph features')
    """
    Engineer additional features from the graph and centrality metrics.

    Args:
        df: DataFrame with network data
        G: NetworkX graph (optional)

    Returns:
        DataFrame with additional engineered features
    """
    # Make a copy to avoid modifying the original
    enhanced_df = df.copy()

    # Combine centrality scores
    enhanced_df['closeness_pagerank_ratio'] = enhanced_df['closeness_cent'] / (enhanced_df['pagerank_cent'] + 1e-6)
    enhanced_df['eigen_betweenness_product'] = enhanced_df['eigenvector_cent'] * enhanced_df['betweenness_cent']
    enhanced_df['degree_closeness_ratio'] = enhanced_df['degree_cent'] / (enhanced_df['closeness_cent'] + 1e-6)
    enhanced_df['pagerank_katz_ratio'] = enhanced_df['pagerank_cent'] / (enhanced_df['katz_cent'] + 1e-6)

    # Apply logarithmic transformation to handle skewness
    enhanced_df['log_betweenness'] = np.log1p(enhanced_df['betweenness_cent'])
    enhanced_df['log_pagerank'] = np.log1p(enhanced_df['pagerank_cent'])

    # Boolean features based on medians
    enhanced_df['high_closeness'] = (enhanced_df['closeness_cent'] > enhanced_df['closeness_cent'].median()).astype(int)
    enhanced_df['high_betweenness'] = (enhanced_df['betweenness_cent'] > enhanced_df['betweenness_cent'].median()).astype(int)
    enhanced_df['high_eigenvector'] = (enhanced_df['eigenvector_cent'] > enhanced_df['eigenvector_cent'].median()).astype(int)

    # Z-score normalization
    centrality_cols = ['closeness_cent', 'pagerank_cent', 'betweenness_cent', 'eigenvector_cent', 'degree_cent', 'katz_cent']
    z_score_cols = [f"{col.split('_')[0]}_z" for col in centrality_cols]

    for col, z_col in zip(centrality_cols, z_score_cols):
        enhanced_df[z_col] = enhanced_df.groupby(['language_code', 'sentence_id'])[col].transform(
            lambda x: zscore(x, nan_policy='omit')
        )

    # Combinations of z-scores
    enhanced_df['closeness_pagerank_z_sum'] = enhanced_df['closeness_z'] + enhanced_df['pagerank_z']
    enhanced_df['eigen_betweenness_z_product'] = enhanced_df['eigenvector_z'] * enhanced_df['betweenness_z']

    # Rank-based features
    enhanced_df['min_rank'] = enhanced_df[['degree_rank', 'closeness_rank', 'betweenness_rank',
                                         'pagerank_rank', 'eigenvector_rank', 'katz_rank']].min(axis=1)
    enhanced_df['max_rank'] = enhanced_df[['degree_rank', 'closeness_rank', 'betweenness_rank',
                                         'pagerank_rank', 'eigenvector_rank', 'katz_rank']].max(axis=1)
    enhanced_df['rank_range'] = enhanced_df['max_rank'] - enhanced_df['min_rank']
    enhanced_df['avg_rank'] = enhanced_df[['degree_rank', 'closeness_rank', 'betweenness_rank',
                                         'pagerank_rank', 'eigenvector_rank', 'katz_rank']].mean(axis=1)

    # If we have a graph, we can compute additional graph-based features
    if G is not None:
        # Initialize new columns
        enhanced_df['clustering_coef'] = 0.0
        enhanced_df['eccentricity'] = 0.0
        enhanced_df['num_leaf_neighbors'] = 0

        # Compute clustering coefficients
        clustering_coeffs = nx.clustering(G)

        # Compute eccentricity (maximum path length from node to any other node)
        try:
            eccentricity = nx.eccentricity(G)
        except nx.NetworkXError:
            # If the graph is not connected, we'll assign -1 to eccentricity
            eccentricity = {node: -1 for node in G.nodes()}

        # Find leaf nodes (nodes with degree 1)
        leaf_nodes = [node for node, degree in dict(G.degree()).items() if degree == 1]

        # For each node in our dataframe
        for _, row in enhanced_df.iterrows():
            vertex = row['vertex']
            if vertex in G:
                # Update clustering coefficient
                enhanced_df.loc[enhanced_df['vertex'] == vertex, 'clustering_coef'] = clustering_coeffs.get(vertex, 0)

                # Update eccentricity
                enhanced_df.loc[enhanced_df['vertex'] == vertex, 'eccentricity'] = eccentricity.get(vertex, -1)

                # Count leaf neighbors
                neighbors = list(G.neighbors(vertex))
                leaf_count = sum(1 for n in neighbors if n in leaf_nodes)
                enhanced_df.loc[enhanced_df['vertex'] == vertex, 'num_leaf_neighbors'] = leaf_count

    return enhanced_df

def final_engineering(data, root=True):
    if root==True:
        # Extract features and targets
        combined_df, sentence_ids = df_central(data)
    else:
        # Extract features without root information
        combined_df, sentence_ids = df_central_without_root(data)

    df = ranking(combined_df, sentence_ids)
    # Try to create a graph if we have all needed data
    try:
        # Get a unique sentence ID (for demonstration)
        sample_sentence_id = df['sentence_id'].iloc[0]

        # Create a graph for this sentence
        G = create_graph(df, sample_sentence_id, root)

        # Engineer additional features using the graph
        enhanced_df = engineer_graph_features(df, G)

        print("\nCreated graph and engineered graph-based features")
    except Exception as e:
        print(f"\nCould not create graph: {e}")
        # Engineer features without graph data
        enhanced_df = engineer_graph_features(df)
        print("\nEngineered features without graph data")

    return enhanced_df

if __name__ == '__main__':
    df=pd.read_csv('../data_kaggle/train.csv')
    enhanced_df=final_engineering(df)
    print(enhanced_df)
    enhanced_df.to_csv('../python_data_pre/engineered_features_train_nor_s.csv', index=False)

    df = pd.read_csv('../data_kaggle/test.csv')
    enhanced_df = final_engineering(df, root=False)
    print(enhanced_df)
    enhanced_df.to_csv('../python_data_pre/engineered_features_test_nor_s.csv', index=False)