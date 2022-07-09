import warnings

from content_based_algorithms.data_queries import RecommenderMethods
from data_connection import Database
from prefilling_all import prepare_and_run, run_prefilling


def try_word2vec_recommendation_prefiller(database, method, full_text, reverse, random):
    prepare_and_run(database, method, full_text, reverse, random)


def try_prefillers():
    database = Database()
    method = "word2vec"
    reverse = True
    random = False
    try_word2vec_recommendation_prefiller(database=database, method=method, full_text=False, reverse=reverse, random=random)


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    recommender_methods = RecommenderMethods()
    recommender_methods.database.insert_posts_dataframe_to_cache()
    run_prefilling()