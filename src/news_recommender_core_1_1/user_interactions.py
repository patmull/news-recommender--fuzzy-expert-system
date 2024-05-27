import logging
import math
import random

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.news_recommender_core_1_1.database import load_user_thumbs, load_user_view_histories, load_posts_df
from src.news_recommender_core_1_1.expert_system import get_interaction_strength
from src.prefillers.preprocessing.stopwords_loading import load_cz_stopwords

from sklearn.metrics.pairwise import cosine_similarity


class TfIdf:

    def __init__(self, articles_df):
        self.articles_df = articles_df

    def create_tfidf_matrix(self):
        logging.debug("Creating TF-IDF matrix...")
        stopwords_list = load_cz_stopwords()

        # Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus,
        # ignoring stopwords
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=0.003,
            max_df=0.5,
            max_features=5000,
            stop_words=stopwords_list
        )

        _item_ids = self.articles_df['post_id'].tolist()
        _tfidf_matrix = vectorizer.fit_transform(self.articles_df['title'] + " " + self.articles_df['body'])
        _tfidf_feature_names = vectorizer.get_feature_names()
        return _tfidf_matrix, _tfidf_feature_names, _item_ids


articles_df = load_posts_df()


class ModelBuilder:

    def __init__(self):
        self.tfidf_matrix = None
        self.feature_names = None
        self.item_ids = None

    def get_item_profile(self, item_id):
        logging.debug("Creating TF-IDF matrix...")
        logging.debug("Finding item profile for item: {}".format(item_id))
        idx = self.item_ids.index(item_id)
        item_profile = self.tfidf_matrix[idx:idx + 1]
        return item_profile

    def get_item_profiles(self, ids):
        item_profiles_list = [self.get_item_profile(x) for x in ids]
        logging.debug("Stacking item profiles row wise...")
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    def build_users_profile(self, person_id, interactions_indexed_df):
        interactions_person_df = interactions_indexed_df.loc[person_id]
        logging.debug("Getting item profiles...")
        user_item_profiles = self.get_item_profiles(interactions_person_df['post_id'])
        logging.info("Reshaping user-item matrix...")
        user_item_strengths = np.array(interactions_person_df['interaction_strength']).reshape(-1, 1)
        # Weighted average of item profiles by the interactions strength
        logging.debug("Calculating weighted average of item profiles by the interactions strength...")
        user_item_strengths_weighted_avg = (np.sum(user_item_profiles.multiply(user_item_strengths), axis=0)
                                            / np.sum(user_item_strengths))
        logging.info("Normalizing user profile...")
        user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
        return user_profile_norm

    def build_users_profiles(self, interactions_train_df, articles_df):
        logging.info("Building indexed iterations...")
        interactions_indexed_df = interactions_train_df[interactions_train_df['post_id'] \
            .isin(articles_df['post_id'])].set_index('user_id')
        logging.info("Building user profiles for each user...")
        user_profiles = {}

        tfidf_recommender = TfIdf(articles_df)
        self.tfidf_matrix, self.feature_names, self.item_ids = tfidf_recommender.create_tfidf_matrix()

        for person_id in interactions_indexed_df.index.unique():
            logging.info("Building user profile for the user: {}".format(person_id))
            user_profiles[person_id] = self.build_users_profile(person_id, interactions_indexed_df)
        return user_profiles

    def get_tfidf_feature_names(self):
        return self.feature_names

    def get_item_ids(self):
        return self.item_ids

    def get_tfidf_matrix(self):
        return self.tfidf_matrix


class ContentBasedRecommender:
    MODEL_NAME = 'Content-Based'

    def __init__(self, items_df, tfidf_matrix, item_ids):
        self.items_df = items_df
        self.tfidf_matrix = tfidf_matrix
        self.item_ids = item_ids

    def get_model_name(self):
        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, person_id, user_profiles, topn=1000):
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[person_id], self.tfidf_matrix)
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(self.item_ids[i], cosine_similarities[0, i]) for i in similar_indices],
                               key=lambda x: -x[1])
        return similar_items

    def recommend_items(self, user_id, user_profiles, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id, user_profiles)
        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['post_id', 'recommendation_strength']) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='post_id',
                                                          right_on='post_id')[
                ['recommendation_strength', 'post_id', 'title', 'slug']]

        return recommendations_df


class HybridRecommender:
    MODEL_NAME = 'Hybrid'

    def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.cb_ensemble_weight = cb_ensemble_weight
        self.cf_ensemble_weight = cf_ensemble_weight
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, user_profiles, items_to_ignore=[], topn=10, verbose=False):
        # Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items(user_id=user_id, user_profiles=user_profiles,
                                                       items_to_ignore=items_to_ignore, verbose=verbose,
                                                       topn=1000).rename(
            columns={'recommendation_strength': 'recommendation_strengthCB'})

        # Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id=user_id,
                                                       items_to_ignore=items_to_ignore, verbose=verbose,
                                                       topn=1000).rename(
            columns={'recommendation_strength': 'recommendation_strengthCF'})

        # Combining the results by post_id
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how='outer',
                                   left_on='post_id',
                                   right_on='post_id').fillna(0.0)

        # Computing a hybrid recommendation score based on CF and CB scores
        # recs_df['recommendation_strengthHybrid'] = recs_df['recommendation_strengthCB'] * recs_df['recommendation_strengthCF']
        recs_df['recommendation_strengthHybrid'] = (recs_df['recommendation_strengthCB'] * self.cb_ensemble_weight) \
                                                   + (recs_df['recommendation_strengthCF'] * self.cf_ensemble_weight)

        # Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recommendation_strengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='post_id',
                                                          right_on='post_id')[
                ['recommendation_strengthHybrid', 'post_id', 'title', 'slug']]

        return recommendations_df


def inspect_interactions(person_id, interactions_test_indexed_df, interactions_train_indexed_df, test_set=True):
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df
    return interactions_df.loc[person_id].merge(articles_df, how='left',
                                                left_on='post_id',
                                                right_on='post_id') \
        .sort_values('interaction_strength', ascending=False)[['interaction_strength',
                                                               'post_id',
                                                               'title', 'slug']]


def init_user_interaction_recommender():
    tested_user_profile_id = 3189

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
    interactions_df['interaction_strength'] = interactions_df['interaction_type'].apply(
        lambda x: event_type_strength[x])

    print("interactions_df:")
    print(interactions_df)

    users_interactions_count_df = interactions_df.groupby(['user_id', 'post_id']).size().groupby('user_id').size()
    print('# users: %d' % len(users_interactions_count_df))
    users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[
        ['user_id']]
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

    logging.info("Building user profiles:")
    model_builder = ModelBuilder()
    user_profiles = model_builder.build_users_profiles(interactions_train_df, articles_df)

    print('Evaluating Popularity recommendation model...')
    pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model, user_profiles)
    print('\nGlobal metrics:\n%s' % pop_global_metrics)
    print(pop_detailed_results_df.head(10).to_string())

    logging.debug(user_profiles)
    logging.debug(len(user_profiles))
    logging.debug("Num. of user profiles:")
    logging.debug(len(user_profiles))

    my_profile = user_profiles[tested_user_profile_id]
    logging.debug("my_profile.shape:")
    logging.debug(my_profile.shape)

    _tfidf_feature_names = model_builder.get_tfidf_feature_names()
    features_names = pd.DataFrame(sorted(zip(_tfidf_feature_names,
                                             user_profiles[tested_user_profile_id].flatten().tolist()),
                                         key=lambda x: -x[1])[:20],
                                  columns=['token', 'relevance'])

    logging.debug("features_names:")
    logging.debug(features_names)

    content_based_recommender_model = ContentBasedRecommender(articles_df,
                                                              model_builder.get_tfidf_matrix(),
                                                              model_builder.get_item_ids())

    print('Evaluating Content-Based Filtering model...')
    cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model,
                                                                               user_profiles)
    print('\nGlobal metrics:\n%s' % cb_global_metrics)
    cb_detailed_results_df.head(10)

    # Creating a sparse pivot table with users in rows and items in columns
    users_items_pivot_matrix_df = interactions_train_df.pivot(index='user_id',
                                                              columns='post_id',
                                                              values='interaction_strength').fillna(0)

    logging.debug(users_items_pivot_matrix_df.head(10))

    users_items_pivot_matrix = users_items_pivot_matrix_df.values
    logging.debug(users_items_pivot_matrix[:10])

    users_ids = list(users_items_pivot_matrix_df.index)
    logging.debug("users_ids[:10]:")
    logging.debug(users_ids[:10])

    users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
    logging.debug("users_items_pivot_sparse_matrix:")
    logging.debug(users_items_pivot_sparse_matrix)

    # The number of factors to factor the user-item matrix.
    NUMBER_OF_FACTORS_MF = 15
    # Performs matrix factorization of the original user item matrix
    # U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
    U, sigma, Vt = svds(users_items_pivot_sparse_matrix.todense(), k=NUMBER_OF_FACTORS_MF)

    logging.debug("SVD shapes:")
    logging.debug(U.shape)
    logging.debug(sigma.shape)
    logging.debug(Vt.shape)

    sigma_matrix = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma_matrix), Vt)
    logging.debug("all_user_predicted_ratings:")
    logging.debug(all_user_predicted_ratings)

    all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (
            all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

    cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns=users_items_pivot_matrix_df.columns,
                               index=users_ids).transpose()
    logging.info("cf_preds_df:")
    logging.info(cf_preds_df.head(10))
    len(cf_preds_df.columns)

    cf_recommender_model = CFRecommender(cf_preds_df, articles_df)
    logging.debug("cf_recommender_model:")
    logging.debug(cf_recommender_model)

    print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
    cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model, user_profiles)
    print('\nGlobal metrics:\n%s' % cf_global_metrics)
    logging.debug(cf_detailed_results_df.head(10))

    # TODO: This will be determined by the fuzzy methods.
    hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, articles_df,
                                                 cb_ensemble_weight=1.0, cf_ensemble_weight=100.0)

    print('Evaluating Hybrid model...')
    hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model,
                                                                                       user_profiles)
    print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
    hybrid_detailed_results_df.head(10)

    global_metrics_df = pd.DataFrame([cb_global_metrics, pop_global_metrics, cf_global_metrics, hybrid_global_metrics]) \
        .set_index('model_name')
    logging.debug(global_metrics_df)

    _inspect_interactions = (inspect_interactions(tested_user_profile_id,
                                                  interactions_test_indexed_df,
                                                  interactions_train_indexed_df,
                                                  test_set=False).head(20))
    logging.debug("inspect_interactions:")
    logging.debug(_inspect_interactions)

    hybrid_recommender_model = hybrid_recommender_model.recommend_items(tested_user_profile_id, user_profiles, topn=20,
                                                                        verbose=True)
    logging.debug("hybrid_recommender_model:")
    logging.debug(hybrid_recommender_model)

    interaction_strength = get_interaction_strength()
    logging.debug("interaction_strength:")
    logging.debug(interaction_strength)

    return item_popularity_df


# Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100


def get_items_interacted(user_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[user_id]['post_id']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
            .reset_index().rename(columns={user_id: 'recommendation_strength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['post_id'].isin(items_to_ignore)] \
            .sort_values('recommendation_strength', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='post_id',
                                                          right_on='post_id')[
                ['recommendation_strength', 'post_id', 'title', 'slug']]

        return recommendations_df


class ModelEvaluator:

    def __init__(self, interactions_full_indexed_df, interactions_train_indexed_df, interactions_test_indexed_df,
                 articles_df):
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

    def evaluate_model_for_user(self, model, user_id, user_profiles):
        # Getting the items in test set
        interacted_values_test_set = self.interactions_test_indexed_df.loc[user_id]
        if type(interacted_values_test_set['post_id']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_test_set['post_id'])
        else:
            person_interacted_items_testset = {int(interacted_values_test_set['post_id'])}
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        if model.get_model_name() == "Collaborative Filtering" or model.get_model_name() == "Popularity":
            person_recs_df = model.recommend_items(user_id=user_id,
                                                   items_to_ignore=get_items_interacted(user_id,
                                                                                        self.interactions_train_indexed_df),
                                                   topn=10000000000)
        else:
            person_recs_df = model.recommend_items(user_id=user_id,
                                                   user_profiles=user_profiles,
                                                   items_to_ignore=get_items_interacted(user_id,
                                                                                        self.interactions_train_indexed_df),
                                                   topn=10000000000)

        logging.debug("Model name: %s" % model.get_model_name())
        logging.debug("recommended df:")
        logging.debug(person_recs_df.head(10))
        logging.debug(person_recs_df.columns)

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
            items_to_filter_recs = non_interacted_items_sample.union({item_id})

            # Filtering only recommendations that are either the interacted item or from a random sample of 100
            # non-interacted items
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

    def evaluate_model(self, model, user_profiles):
        # print('Running evaluation for users')
        people_metrics = []
        for idx, user_id in enumerate(list(self.interactions_test_indexed_df.index.unique().values)):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, user_id, user_profiles)
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

    def recommend_items(self, user_id, items_to_ignore=None, topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        if items_to_ignore is None:
            items_to_ignore = []
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

        logging.debug("recommendations_df:")
        logging.debug(recommendations_df.head(10))

        return recommendations_df
