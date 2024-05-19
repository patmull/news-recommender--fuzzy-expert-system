"""
Proposed new structure of the results:

{
    "queried": "article-slug-a",
    "recommended" : [
        {
            "k": 1
            "slug": "article-slug-b",
            "relevant": 1,
        },
        {
            "k": 2
            "slug": "article-slug-c",
            "relevant": 0,
        },
        ...
    ],
    "model": "word2vec",
    "queried_category": "sport",
}
"""
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.user_evaluation_results import \
    get_thumbs_evaluations, get_playground_evaluations


def clean_results_playground():
    df_user_testing_results = get_playground_evaluations()
    return df_user_testing_results


def clean_results_thumbs():
    df_user_testing_results = get_thumbs_evaluations()
    return df_user_testing_results
