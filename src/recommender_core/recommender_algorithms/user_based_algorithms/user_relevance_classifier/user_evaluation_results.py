from src.recommender_core.data_handling.data_queries import RecommenderMethods


def get_playground_evaluations():
    recommender_methods = RecommenderMethods()
    return recommender_methods.get_ranking_evaluation_results_dataframe()  # load_texts posts to dataframe


def get_thumbs_evaluations():
    """
    User thumbs ratings.
    @return:
    """
    recommender_methods = RecommenderMethods()
    return recommender_methods.get_item_evaluation_results_dataframe()  # load_texts posts to dataframe
