import pandas as pd
from src.news_recommender_core_1_1.database import load_user_thumbs, load_user_view_histories


def get_user_interactions():
    user_thumbs = load_user_thumbs()
    interactions_df_likes = pd.DataFrame.from_dict(user_thumbs, orient='columns')
    interactions_df_likes = interactions_df_likes[interactions_df_likes.value != 0]
    interactions_df_likes['interaction_type'] = 'LIKE'

    user_views = load_user_view_histories()
    interactions_df_views = pd.DataFrame.from_dict(user_views, orient='columns')
    interactions_df_views['interaction_type'] = 'VIEW'

    event_type_strength = {
        'VIEW': 1.0,
        'LIKE': 2.0,
    }

    # NOTE: Originally interaction_strength and _type => event_*
    interactions_df = pd.concat([interactions_df_likes, interactions_df_views])
    interactions_df['interaction_strength'] = interactions_df['interaction_type'].apply(lambda x: event_type_strength[x])

    print("interactions_df:")
    print(interactions_df)

    return interactions_df
