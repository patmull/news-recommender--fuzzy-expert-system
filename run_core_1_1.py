from src.news_recommender_core_1_1.user_interactions import init_user_interaction_recommender

import logging
logging.basicConfig(level=logging.INFO)

# init_user_interaction_recommender(num_of_users=40, num_of_interactions=200, topn_recommended=20)
# CF still poor. Needs to be increased.
# or does not need to be: we can show it performs better if sample size low
init_user_interaction_recommender(num_of_users=40, num_of_interactions=600, topn_recommended=40)
