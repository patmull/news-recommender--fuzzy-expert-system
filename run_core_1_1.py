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
    belief_in_viewed = range(0, 10, 1)

    path_to_results_csv = Path("evaluations/1_1/results.csv")

    with open(path_to_results_csv.as_posix(), "w+") as f:
        f.write("belief_in_model_cb,belief_in_model_cf,belief_in_interaction_strength_views_global,belief_in_interaction_strength_likes_global,belief_in_liked,belief_in_viewed,recall_at_5,recall_at_10\n")

    logging.info("Starting hyperparameter tuning of the fuzzy expert model...")

    for belief_in_model_cb in beliefs_in_model_cb:
        for belief_in_model_cf in beliefs_in_model_cf:
            for belief_in_interaction_strength_views_global in beliefs_in_interaction_strength_views_global:
                for belief_in_interaction_strength_likes_global in beliefs_in_interaction_strength_likes_global:
                    for belief_in_liked in beliefs_in_liked:
                        for belief_in_viewed in belief_in_viewed:
                            logging.info("Tested parameters:")
                            logging.info("belief_in_model_cb: {}".format(belief_in_model_cb))
                            logging.info("belief_in_model_cf: {}".format(belief_in_model_cf))
                            logging.info("belief_in_interaction_strength_views_global: {}".format(belief_in_interaction_strength_views_global))
                            logging.info("belief_in_interaction_strength_likes_global: {}".format(belief_in_interaction_strength_likes_global))
                            logging.info("belief_in_liked: {}".format(belief_in_liked))
                            logging.info("belief_in_viewed: {}".format(belief_in_viewed))
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

                            with open(path_to_results_csv.as_posix(), "a") as f:
                                f.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                                belief_in_model_cb, belief_in_model_cf, belief_in_interaction_strength_views_global,
                                belief_in_interaction_strength_likes_global, belief_in_liked, belief_in_viewed,
                                recall_at_5, recall_at_10))


hyperparameter_tuning_fuzzy_expert()

