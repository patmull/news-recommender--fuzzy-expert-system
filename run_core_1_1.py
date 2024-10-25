from pathlib import Path

from src.news_recommender_core_1_1.user_interactions import init_user_interaction_recommender

import logging
import random

logging.basicConfig(level=logging.INFO)


# init_user_interaction_recommender(num_of_users=40, num_of_interactions=200, limit_num_of_recommendations=20)
# CF still poor. Needs to be increased.
# or we can show it performs better if sample size low, but CF still needs to be reasonable for better results
# init_user_interaction_recommender(use_fuzzy_expert=True)
# init_user_interaction_recommender(use_fuzzy_expert=False)


def hyperparameter_tuning_fuzzy_expert_grid_search():
    beliefs_in_model_cb = range(0, 10, 1)
    beliefs_in_model_cf = range(0, 10, 1)
    beliefs_in_interaction_strength_views_global = range(0, 10, 1)
    beliefs_in_interaction_strength_likes_global = range(0, 10, 1)
    beliefs_in_liked = range(0, 10, 1)
    beliefs_in_viewed = range(0, 10, 1)

    path_to_results_csv = Path("evaluations/1_1/results_grid_search.csv")

    existing_rows = set()  # Set to store existing rows

    # Read existing rows from the CSV file
    with open(path_to_results_csv.as_posix(), "r") as f:
        next(f)  # Skip header
        for line in f:
            existing_rows.add(tuple(int(x) if x else 0 for x in line.strip().split(',')[:6]))

    for belief_in_model_cb in beliefs_in_model_cb:
        for belief_in_model_cf in beliefs_in_model_cf:
            for belief_in_interaction_strength_views_global in beliefs_in_interaction_strength_views_global:
                for belief_in_interaction_strength_likes_global in beliefs_in_interaction_strength_likes_global:
                    for belief_in_liked in beliefs_in_liked:
                        for belief_in_viewed in beliefs_in_viewed:
                            new_row = (
                                belief_in_model_cb, belief_in_model_cf, belief_in_interaction_strength_views_global,
                                belief_in_interaction_strength_likes_global, belief_in_liked, belief_in_viewed)
                            if new_row not in existing_rows:  # Check if new row already exists
                                recall_at_5, recall_at_10 = init_user_interaction_recommender(
                                    belief_in_model_cb=belief_in_model_cb,
                                    belief_in_model_cf=belief_in_model_cf,
                                    belief_in_interaction_strength_views_global=belief_in_interaction_strength_views_global,
                                    belief_in_interaction_strength_likes_global=belief_in_interaction_strength_likes_global,
                                    belief_in_liked=belief_in_liked,
                                    belief_in_viewed=belief_in_viewed,
                                    fuzzy_interactions_global=False,
                                    fuzzy_interactions_user=False,
                                    fuzzy_ensemble=False
                                )

                                with open(path_to_results_csv.as_posix(), "a") as f:
                                    logging.info("Results:")
                                    logging.info("recall_at_5: {}".format(recall_at_5))
                                    logging.info("recall_at_10: {}".format(recall_at_10))

                                    f.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                                        belief_in_model_cb, belief_in_model_cf,
                                        belief_in_interaction_strength_views_global,
                                        belief_in_interaction_strength_likes_global,
                                        belief_in_liked, belief_in_viewed,
                                        recall_at_5, recall_at_10))

                                existing_rows.add(new_row)  # Add the new row to existing rows set

    logging.info("All iterations completed.")


def hyperparameter_tuning_fuzzy_expert_random_search(num_iterations=100000000):
    """

    :param num_iterations: some sources say that regardless of the # of dimensions,
    60 iterations should be enough for reaching at least 1 from top 5% solution
    :return:
    """
    beliefs_in_model_cb = range(0, 10, 1)
    beliefs_in_model_cf = range(0, 10, 1)
    beliefs_in_interaction_strength_views_global = range(0, 10, 1)
    beliefs_in_interaction_strength_likes_global = range(0, 10, 1)
    beliefs_in_liked = range(0, 10, 1)
    beliefs_in_viewed = range(0, 10, 1)

    path_to_results_csv = Path("evaluations/1_1/results_random_search_new.csv")

    existing_rows = set()  # Set to store existing rows

    # Read existing rows from the CSV file
    try:
        with open(path_to_results_csv.as_posix(), "r") as f:
            next(f)  # Skip header
            for line in f:
                # existing_rows.add(tuple(int(x) if x else 0 for x in line.strip().split(',')[:6]))
                existing_rows.add(tuple(int(x) if x else 0 for x in line.strip().split(',')[:2]))
    except FileNotFoundError:
        # If file does not exist, it will be created when writing the first result
        pass

    for _ in range(num_iterations):
        new_row = (
            random.choice(beliefs_in_model_cb),
            random.choice(beliefs_in_model_cf),
            random.choice(beliefs_in_interaction_strength_views_global),
            random.choice(beliefs_in_interaction_strength_likes_global),
            random.choice(beliefs_in_liked),
            random.choice(beliefs_in_viewed)
        )

        if new_row not in existing_rows:  # Check if the combination is unique
            (recall_at_5_hybrid, recall_at_10_hybrid, recall_at_5_hybrid_or_fuzzy_hybrid,
             recall_at_10_hybrid_or_fuzzy_hybrid, recall_at_5_hybrid_bayesian, recall_at_10_hybrid_bayesian,
             recall_at_5_cf, recall_at_10_cf, recall_at_5_cb, recall_at_10_cb, recall_at_5_pop, recall_at_10_pop) \
                = init_user_interaction_recommender(
                belief_in_model_cb=new_row[0],
                belief_in_model_cf=new_row[1],
                belief_in_interaction_strength_views_global=new_row[2],
                belief_in_interaction_strength_likes_global=new_row[3],
                belief_in_liked=new_row[4],
                belief_in_viewed=new_row[5],
                fuzzy_interactions_global=False,
                fuzzy_interactions_user=False,
                fuzzy_ensemble=False,
                limit_num_of_recommendations_recommended=10000000000000
            )

            with open(path_to_results_csv.as_posix(), "a") as f:
                logging.info("Results for iteration with: %s", new_row)
                logging.info("recall_at_5_hybrid: {}".format(recall_at_5_hybrid_or_fuzzy_hybrid))
                logging.info("recall_at_10_hybrid: {}".format(recall_at_10_hybrid_or_fuzzy_hybrid))
                logging.info("recall_at_5_hybrid_or_fuzzy_hybrid: {}".format(recall_at_5_hybrid_or_fuzzy_hybrid))
                logging.info("recall_at_10_hybrid_or_fuzzy_hybrid: {}".format(recall_at_10_hybrid_or_fuzzy_hybrid))
                logging.info("recall_at_5_hybrid_bayesian: {}".format(recall_at_5_hybrid_bayesian))
                logging.info("recall_at_10_hybrid_bayesian: {}".format(recall_at_10_hybrid_bayesian))
                logging.info("recall_at_5_cf: {}".format(recall_at_5_cf))
                logging.info("recall_at_10_cf: {}".format(recall_at_10_cf))
                logging.info("recall_at_5_cb: {}".format(recall_at_5_cb))
                logging.info("recall_at_10_cb: {}".format(recall_at_10_cb))
                logging.info("recall_at_5_pop: {}".format(recall_at_5_pop))
                logging.info("recall_at_10_pop: {}".format(recall_at_10_pop))

                f.write(
                    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                        new_row[0], new_row[1],
                        new_row[2], new_row[3],
                        new_row[4], new_row[5],
                        recall_at_5_hybrid, recall_at_10_hybrid,
                        recall_at_5_hybrid_or_fuzzy_hybrid, recall_at_10_hybrid_or_fuzzy_hybrid,
                        recall_at_5_hybrid_bayesian, recall_at_10_hybrid_bayesian,
                        recall_at_5_cf, recall_at_10_cf, recall_at_5_cb, recall_at_10_cb,
                        recall_at_5_pop, recall_at_10_pop
                    ))

            existing_rows.add(new_row)  # Add the new row to existing rows set

    logging.info("Random search completed.")


def hyperparameter_tuning_ensemble_random_search(num_iterations=100000000):
    beliefs_in_model_cb = range(0, 10, 1)
    beliefs_in_model_cf = range(0, 10, 1)

    path_to_results_csv = Path("evaluations/1_1/results_random_search_new.csv")

    existing_rows = set()  # Set to store existing rows

    # Read existing rows from the CSV file
    try:
        with open(path_to_results_csv.as_posix(), "r") as f:
            next(f)  # Skip header
            for line in f:
                existing_rows.add(tuple(int(x) if x else 0 for x in line.strip().split(',')[:2]))
    except FileNotFoundError:
        # If file does not exist, it will be created when writing the first result
        pass

    for _ in range(num_iterations):
        new_row = (
            random.choice(beliefs_in_model_cb),
            random.choice(beliefs_in_model_cf)
        )

        if new_row not in existing_rows:  # Check if the combination is unique
            recall_at_5, recall_at_10 = init_user_interaction_recommender_ensemble(
                belief_in_model_cb=new_row[0],
                belief_in_model_cf=new_row[1]
            )

            with open(path_to_results_csv.as_posix(), "a") as f:
                logging.info("Results for iteration with: %s", new_row)
                logging.info("recall_at_5: {}".format(recall_at_5))
                logging.info("recall_at_10: {}".format(recall_at_10))

                f.write("%s,%s,%s,%s\n" % (new_row[0], new_row[1], recall_at_5, recall_at_10))

            existing_rows.add(new_row)  # Add the new row to existing rows set

    logging.info("Random search completed.")


# hyperparameter_tuning_fuzzy_expert_grid_search()
hyperparameter_tuning_fuzzy_expert_random_search()
