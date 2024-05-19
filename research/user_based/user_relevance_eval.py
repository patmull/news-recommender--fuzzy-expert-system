import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, balanced_accuracy_score, dcg_score, f1_score, jaccard_score, ndcg_score

from src.constants.naming import Naming
from src.recommender_core.data_handling.data_manipulation import get_redis_connection
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.user_evaluation_results import \
    get_thumbs_evaluations


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)


def create_relevance_stats_df(sections):
    """
    User evaluation based on the front-side thumbs ratings of the users.
    @param sections:
    @return: Pandas Dataframe with the used statistics on the user thumbs ratings.
    """

    precision_score_weighted_list = []
    dcg_score_list = []
    dcg_score_at_k_list = []
    f1_score_list = []
    jaccard_score_list = []
    ndcg_score_list = []
    ndcg_score_at_k_list = []
    balanced_accuracy_score_list = []
    precision_score_list = []
    N_list = []
    sections_list = []

    for section in sections:
        user_eval_df = get_thumbs_evaluations()
        user_eval_df = user_eval_df.loc[user_eval_df['method_section'] == section]
        true_relevance = user_eval_df['value'].tolist()

        if len(true_relevance) > 0:
            pred = np.full((1, len(true_relevance)), 1)[0]

            logging.info("Calculating statistics...")
            precision_score_weighted_list.append(precision_score(true_relevance, pred, average='weighted'))
            dcg_score_list.append(dcg_score([true_relevance], [pred]))
            dcg_score_at_k_list.append(dcg_score([true_relevance], [pred], k=5))
            f1_score_list.append(f1_score(true_relevance, pred, average='weighted'))
            jaccard_score_list.append(jaccard_score(true_relevance, pred))
            ndcg_score_list.append(ndcg_score([true_relevance], [pred]))
            ndcg_score_at_k_list.append(ndcg_score([true_relevance], [pred], k=5))
            balanced_accuracy_score_list.append(balanced_accuracy_score(true_relevance, pred))
            precision_scores = precision_score(true_relevance, pred, average=None)

            precision_score_list_cleaned = [ps for ps in precision_scores if ps != 0]
            logging.debug("precision_score_list_cleaned:")
            logging.debug(precision_score_list_cleaned)
            precision_score_list.append(precision_score_list_cleaned[0])
            sections_list.append(section)
            N_list.append(len(true_relevance))

    df = pd.DataFrame({
        'precision_score': precision_score_list,
        'balanced_accuracy_score': balanced_accuracy_score_list,
        'dcg_score': dcg_score_list,
        'dcg_score_at_k': dcg_score_at_k_list,
        'f1_score': f1_score_list,
        'jaccard_score': jaccard_score_list,
        'ndcg_score': ndcg_score_list,
        'ndcg_at_k_score': ndcg_score_at_k_list,
        'precision_score_weighted': precision_score_weighted_list,
        'sections_list': sections_list,
        'N': N_list
    })

    return df


def user_relevance_asessment(save_to_redis=True):
    """
    Statistics coming from relevance votes (thumbs) of users.
    RUN WITH: run_user_eval.py
    @return:
    """

    sections = get_thumbs_evaluations()['method_section'].unique().tolist()

    df = create_relevance_stats_df(sections)

    output_file = Path("research/relevance/from_front_pages/user_eval_results.csv")
    df.to_csv(output_file.as_posix(), header=True)
    logging.info("Saved user evaluation results to: %s" % output_file.as_posix())

    if save_to_redis:
        REDIS_TOP_FOLDER = 'statistics'
        testing_redis_key = REDIS_TOP_FOLDER + Naming.REDIS_DELIMITER + 'testing-redis-key'
        r = get_redis_connection()
        if "PYTEST_CURRENT_TEST" in os.environ:
            r.set(testing_redis_key, 'testing-redis-value')
        else:
            list_of_columns = df.columns.tolist()
            list_of_columns.remove('sections_list')
            for section in df['sections_list'].tolist():
                for column_name in list_of_columns[1:]:
                    redis_key = REDIS_TOP_FOLDER + Naming.REDIS_DELIMITER + section + Naming.REDIS_DELIMITER \
                                + column_name
                    stat_value = df.loc[df['sections_list'] == section, column_name].iloc[0]
                    r.set(redis_key, float(stat_value))
