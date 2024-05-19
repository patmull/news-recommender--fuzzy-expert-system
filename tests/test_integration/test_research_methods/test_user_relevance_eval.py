import pandas as pd

from research.user_based.user_relevance_eval import create_relevance_stats_df
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.user_evaluation_results import \
    get_thumbs_evaluations


def test_get_user_evaluation_results_dataframe():
    recommender_methods = RecommenderMethods()
    results_df = recommender_methods.get_item_evaluation_results_dataframe()  # load_texts posts to dataframe
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df.index) > 0


def test_create_relevance_stats_df():
    sections = get_thumbs_evaluations()['method_section'].unique().tolist()
    results_df = create_relevance_stats_df(sections)  # load_texts posts to dataframe
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df.index) > 0
    expected_column_names = ['precision_score', 'balanced_accuracy_score',
                             'dcg_score', 'dcg_score_at_k', 'f1_score',
                             'jaccard_score', 'ndcg_score', 'ndcg_at_k_score',
                             'precision_score_weighted', 'sections_list', 'N']

    for expected_column_name in expected_column_names:
        assert expected_column_name in results_df.columns.tolist()
