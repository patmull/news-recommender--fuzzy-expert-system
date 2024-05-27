import json
from pathlib import Path

from src.news_recommender_core_1_1.database import load_post, load_post_category, load_post_ratings, load_user_thumbs_for_post
from src.news_recommender_core_1_1.evaluation.utils.cleaning_results import save_clean_results_playground, \
    add_post_features
from src.news_recommender_core_1_1.user_interactions import init_user_interaction_recommender

#save_clean_results_playground()
#add_post_features()

# save_clean_results_playground()
# add_post_features()

init_user_interaction_recommender()
