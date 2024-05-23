import math
import random
import pandas as pd
from sklearn.model_selection import train_test_split

from src.news_recommender_core_1_1.database import load_user_thumbs, load_user_view_histories, load_posts_df


def get_user_interactions():
    user_thumbs = load_user_thumbs()
    interactions_df_likes = pd.DataFrame.from_dict(user_thumbs, orient='columns')
    interactions_df_likes = interactions_df_likes[interactions_df_likes.value != 0]
    interactions_df_likes['interaction_type'] = 'LIKE'

    user_views = load_user_view_histories()
    interactions_df_views = pd.DataFrame.from_dict(user_views, orient='columns')
    interactions_df_views['interaction_type'] = 'VIEW'

    # TODO: This will be determined by the fuzzy methods.
    # possible variables:
    # number of interactions by the user
    # level of a belief in a given interaction

    event_type_strength = {
        'VIEW': 1.0,
        'LIKE': 2.0,
    }

    # NOTE: Originally interaction_strength and _type => event_*
    interactions_df = pd.concat([interactions_df_likes, interactions_df_views])
    interactions_df['interaction_strength'] = interactions_df['interaction_type'].apply(lambda x: event_type_strength[x])

    print("interactions_df:")
    print(interactions_df)

    users_interactions_count_df = interactions_df.groupby(['user_id', 'post_id']).size().groupby('user_id').size()
    print('# users: %d' % len(users_interactions_count_df))
    users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['user_id']]
    print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))

    print('# of interactions: %d' % len(interactions_df))
    interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df,
                                                                how='right',
                                                                left_on='user_id',
                                                                right_on='user_id')
    print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))

    def smooth_user_preference(x):
        return math.log(1 + x, 2)

    interactions_full_df = interactions_from_selected_users_df \
        .groupby(['user_id', 'post_id'])['interaction_strength'].sum() \
        .apply(smooth_user_preference).reset_index()
    print('# of unique user/item interactions: %d' % len(interactions_full_df))
    print(interactions_full_df.head(10))

    interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                   stratify=interactions_full_df['user_id'],
                                                                   test_size=0.20,
                                                                   random_state=42)

    print('# interactions on Train set: %d' % len(interactions_train_df))
    print('# interactions on Test set: %d' % len(interactions_test_df))

    interactions_full_indexed_df = interactions_full_df.set_index('user_id')
    interactions_train_indexed_df = interactions_train_df.set_index('user_id')
    interactions_test_indexed_df = interactions_test_df.set_index('user_id')

    print("Loading posts...")
    articles_df = load_posts_df()

    print("articles_df:")
    print(articles_df.head(10))

    print("Creating df of posts.")
    model_evaluator = ModelEvaluator(interactions_full_indexed_df, interactions_train_indexed_df,
                                     interactions_test_indexed_df, articles_df)
    print("created model evaluator")

    # Computes the most popular items
    item_popularity_df = interactions_full_df.groupby('post_id')['interaction_strength'].sum().sort_values(
        ascending=False).reset_index()
    print(item_popularity_df.head(10))

    # Create model
    popularity_model = PopularityRecommender(item_popularity_df, articles_df)
    print("created popularity model")

    return item_popularity_df


# Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100


def get_items_interacted(user_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[user_id]['post_id']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


class ModelEvaluator:

    def __init__(self, interactions_full_indexed_df, interactions_train_indexed_df, interactions_test_indexed_df, articles_df):
        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.interactions_test_indexed_df = interactions_test_indexed_df
        self.articles_df = articles_df

    def get_not_interacted_items_sample(self, user_id, sample_size, seed=42):
        interacted_items = get_items_interacted(user_id, self.interactions_full_indexed_df)
        all_items = set(self.articles_df['post_id'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def evaluate_model_for_user(self, model, user_id):
        # Getting the items in test set
        interacted_values_test_set = self.interactions_test_indexed_df.loc[user_id]
        if type(interacted_values_test_set['post_id']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_test_set['post_id'])
        else:
            person_interacted_items_testset = set([int(interacted_values_test_set['post_id'])])
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(user_id,
                                               items_to_ignore=get_items_interacted(user_id,
                                                                                    self.interactions_train_indexed_df),
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            # Getting a random sample (100) items the user has not interacted
            # (to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(user_id,
                                                                               sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                                               seed=item_id % (2 ** 32))

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['post_id'].isin(items_to_filter_recs)]
            valid_recs = valid_recs_df['post_id'].values
            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
        # when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count': hits_at_5_count,
                          'hits@10_count': hits_at_10_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        # print('Running evaluation for users')
        people_metrics = []
        for idx, user_id in enumerate(list(self.interactions_test_indexed_df.index.unique().values)):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, user_id)
            person_metrics['_user_id'] = user_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
            .sort_values('interacted_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())

        global_metrics = {'model_name': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}
        return global_metrics, detailed_results_df


class PopularityRecommender:
    MODEL_NAME = 'Popularity'

    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['post_id'].isin(items_to_ignore)] \
            .sort_values('interaction_strength', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='post_id',
                                                          right_on='post_id')[
                ['interaction_strength', 'post_id', 'title', 'slug']]

        return recommendations_df


