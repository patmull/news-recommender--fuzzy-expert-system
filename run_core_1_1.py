from pathlib import Path

from src.news_recommender_core_1_1.user_interactions import init_user_interaction_recommender

import logging

logging.basicConfig(level=logging.INFO)


# init_user_interaction_recommender(num_of_users=40, num_of_interactions=200, topn_recommended=20)
# CF still poor. Needs to be increased.
# or we can show it performs better if sample size low, but CF still needs to be reasonable for better results
# init_user_interaction_recommender(use_fuzzy_expert=True)
# init_user_interaction_recommender(use_fuzzy_expert=False)


def hyperparameter_tuning_fuzzy_expert():
    beliefs_in_model_cb = range(0, 10, 1)
    beliefs_in_model_cf = range(0, 10, 1)
    beliefs_in_interaction_strength_views_global = range(0, 10, 1)
    beliefs_in_interaction_strength_likes_global = range(0, 10, 1)
    beliefs_in_liked = range(0, 10, 1)
    beliefs_in_viewed = range(0, 10, 1)

    path_to_results_csv = Path("evaluations/1_1/results.csv")

    existing_rows = set()  # Set to store existing rows

    # Read existing rows from the CSV file
    with open(path_to_results_csv.as_posix(), "r") as f:
        next(f)  # Skip header
        for line in f:
            existing_rows.add(tuple(int(x) if x else 0 for x in line.strip().split(',')[:6]))

    with open(path_to_results_csv.as_posix(), "a") as f:
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
                                        use_fuzzy_expert=True,
                                        belief_in_model_cb=belief_in_model_cb,
                                        belief_in_model_cf=belief_in_model_cf,
                                        belief_in_interaction_strength_views_global=belief_in_interaction_strength_views_global,
                                        belief_in_interaction_strength_likes_global=belief_in_interaction_strength_likes_global,
                                        belief_in_liked=belief_in_liked,
                                        belief_in_viewed=belief_in_viewed)

                                    logging.info("Results:")
                                    logging.info("recall_at_5: {}".format(recall_at_5))
                                    logging.info("recall_at_10: {}".format(recall_at_10))

                                    f.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                                        belief_in_model_cb, belief_in_model_cf,
                                        belief_in_interaction_strength_views_global,
                                        belief_in_interaction_strength_likes_global, belief_in_liked, belief_in_viewed,
                                        recall_at_5, recall_at_10))

                                    existing_rows.add(new_row)  # Add the new row to existing rows set

    logging.info("All iterations completed.")


hyperparameter_tuning_fuzzy_expert()
