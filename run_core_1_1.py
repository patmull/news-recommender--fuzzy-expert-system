from src.news_recommender_core_1_1.user_interactions import init_user_interaction_recommender

import logging
logging.basicConfig(level=logging.INFO)

# init_user_interaction_recommender(num_of_users=40, num_of_interactions=200, topn_recommended=20)
# CF still poor. Needs to be increased.
# or we can show it performs better if sample size low, but CF still needs to be reasonable for better results
init_user_interaction_recommender(num_of_users=1000000000, num_of_interactions=1000, topn_recommended=1000000000, use_fuzzy_expert=False)
# init_user_interaction_recommender(use_fuzzy_expert=False)
