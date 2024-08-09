import csv
import logging
import math
import os
import pickle
import random
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.news_recommender_core_1_1.database import load_user_thumbs, load_user_view_histories, load_posts_df
from src.news_recommender_core_1_1.expert_system import get_interaction_strength, \
    get_recommendation_strength_hybrid
from src.prefillers.preprocessing.stopwords_loading import load_cz_stopwords

from sklearn.metrics.pairwise import cosine_similarity

import logging

logging.basicConfig(level=logging.DEBUG)

TOP_N = 20

class TfIdf:

    def __init__(self, articles_df):
        self.articles_df = articles_df

    @lru_cache(maxsize=8192)  # Or some appropriate size
    def create_tfidf_matrix(self):
        logging.debug("Creating TF-IDF matrix...")
        stopwords_list = load_cz_stopwords()

        # Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus,
        # ignoring stopwords
        vectorizer = TfidfVectorizer(
            analyzer='word',
            # ngram_range=(1, 2),
            # min_df=0.003,
            # max_df=0.5,
            # max_features=5000,
            stop_words=stopwords_list
        )

        self.articles_df['trigrams_short_text'] = self.articles_df['trigrams_short_text'].mask(
            self.articles_df['trigrams_short_text'].isna(),
            self.articles_df['all_features_preprocessed']
        )

        self.articles_df['trigrams_full_text'] = self.articles_df['trigrams_full_text'].mask(
            self.articles_df['trigrams_full_text'].isna(),
            self.articles_df['all_features_preprocessed']
        )
        # to prevent both columns same
        self.articles_df['trigrams_full_text'] = self.articles_df.apply(
            lambda row: '' if row['trigrams_full_text'] == row['trigrams_short_text'] else row['trigrams_full_text'],
            axis=1)
        _item_ids = self.articles_df['post_id'].tolist()
        _tfidf_matrix = vectorizer.fit_transform(  # self.articles_df['title']
            # + " " +
            # self.articles_df['excerpt']
            # + " " +
            # self.articles_df['body']
            # self.articles_df['all_features_preprocessed']
            self.articles_df['trigrams_short_text'] + " " + self.articles_df['trigrams_full_text']
        )
        _tfidf_feature_names = vectorizer.get_feature_names_out()
        return _tfidf_matrix, _tfidf_feature_names, _item_ids


articles_df = load_posts_df()


class ModelBuilder:

    def __init__(self):
        self.tfidf_matrix = None
        self.feature_names = None
        self.item_ids = None

    @lru_cache(maxsize=8192)  # Or some appropriate size
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
        user_item_strengths = np.array(interactions_person_df['recommendation_strength']).reshape(-1, 1)
        # Weighted average of item profiles by the interactions strength
        logging.debug("Calculating weighted average of item profiles by the interactions strength...")
        is_all_zero = np.isin(user_item_strengths, 0).all()

        logging.debug("user_item_profiles shape:")
        logging.debug(user_item_profiles.shape[1])
        logging.debug("user_item_strengths shape:")
        logging.debug(user_item_strengths.shape)

        if is_all_zero:
            user_item_strengths_weighted_avg = np.zeros((1, user_item_profiles.shape[1]))
        else:
            user_item_strengths_weighted_avg = (
                    np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths))

        logging.debug("user_item_strengths_weighted_avg shape:")
        logging.debug(user_item_strengths_weighted_avg.shape)

        # Convert the matrix to a numpy array
        user_item_strengths_weighted_avg_array = np.asarray(user_item_strengths_weighted_avg)
        user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg_array)
        return user_profile_norm

    def build_users_profiles(self, interactions_train_df, articles_df):
        logging.info("Building indexed iterations...")
        interactions_indexed_df = interactions_train_df[interactions_train_df['post_id'] \
            .isin(articles_df['post_id'])].set_index('user_id')
        logging.info("Building user profiles for each user...")
        user_profiles = {}

        tfidf_recommender = TfIdf(articles_df)
        self.tfidf_matrix, self.feature_names, self.item_ids = tfidf_recommender.create_tfidf_matrix()

        logging.debug("interactions_indexed_df:")
        logging.debug(interactions_indexed_df.head(10))

        for person_id in interactions_indexed_df.index:
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

    def _get_similar_items_to_user_profile(self, person_id, user_profiles, limit_num_of_recommendations=TOP_N):
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[person_id], self.tfidf_matrix)
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-limit_num_of_recommendations:]
        # Sort the similar items by similarity
        similar_items = sorted([(self.item_ids[i], cosine_similarities[0, i]) for i in similar_indices],
                               key=lambda x: -x[1])
        return similar_items

    def recommend_items_cb(self, user_id, user_profiles, items_to_ignore=[], limit_num_of_recommendations=1000000, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id, user_profiles, limit_num_of_recommendations)
        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['post_id', 'recommendation_strength'])

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='post_id',
                                                          right_on='post_id')[
                ['recommendation_strength', 'post_id', 'title', 'slug']]

        return recommendations_df


class HybridRecommender:
    """
    self.cb_ensemble_weight and self.cf_ensemble_weight: serves for the non-fuzzy expert case,
    if fuzzy expert system is used,
    then these coeffs are not used and replaced by the computed value in the fuzzy expert hybrid
    """

    def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0,
                 fuzzy_expert=False):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.cb_ensemble_weight = cb_ensemble_weight
        self.cf_ensemble_weight = cf_ensemble_weight
        self.items_df = items_df
        self.fuzzy_expert_enabled = fuzzy_expert
        if fuzzy_expert:
            self.model_name = 'Fuzzy Hybrid'
        else:
            self.model_name = 'Hybrid'
        self.iteration_counter = 0
        self.recs_df = pd.DataFrame()

    def save_recommendation_strength(self, interactions_from_selected_users_df):
        filename_to_save = 'recommendation_strength_{}.pkl'.format(self.point_to_save)
        file_path = Path('results/{}'.format(filename_to_save))
        logging.info("""Saving the recommendation strength data to: {}""".format(file_path))
        with open(file_path.as_posix(), 'wb') as f:
            pickle.dump(interactions_from_selected_users_df, f)

    def load_recommendation_strength(self):
        filename_to_save = 'recommendation_strength_{}.pkl'.format(self.point_to_save)
        file_path = Path('results/{}'.format(filename_to_save))
        if not file_path.is_file():
            return None
        else:
            logging.info("Loading saved recommendation strengths...")
            with open(file_path.as_posix(), 'rb') as f:
                return pickle.load(f)

    def get_cb_recommendations(self, user_id, user_profiles, items_to_ignore=[], limit_num_of_recommendations=TOP_N):
        return self.cb_rec_model.recommend_items_cb(user_id, user_profiles, items_to_ignore, limit_num_of_recommendations=limit_num_of_recommendations)

    def get_cf_recommendations(self, user_id, items_to_ignore=[], limit_num_of_recommendations=TOP_N):
        return self.cf_rec_model.recommend_items_cf(user_id, items_to_ignore, limit_num_of_recommendations=limit_num_of_recommendations)

    def get_recommendation_strength_hybrid_with_logging(self,
                                                        belief_in_model_cf,
                                                        belief_in_model_cb,
                                                        recommendation_coefficient_cf,
                                                        recommendation_coefficient_cb,
                                                        recommendation_strength_result,
                                                        save_every_n_iterations=1000000):
        # this should save the time radically since the cache is used later
        belief_in_model_cf = round(float(belief_in_model_cf), 2)
        belief_in_model_cb = round(float(belief_in_model_cb), 2)
        recommendation_coefficient_cf = round(float(recommendation_coefficient_cf), 2)
        recommendation_coefficient_cb = round(float(recommendation_coefficient_cb), 2)

        if recommendation_strength_result is not None:
            return recommendation_strength_result
        else:
            # save every N iteration
            """
            if self.iteration_counter % save_every_n_iterations == 0:
                self.save_recommendation_strength(self.recs_df)
            """

            if recommendation_coefficient_cf == 0 or recommendation_coefficient_cb == 0:
                self.iteration_counter += 1
                return 0.0
            else:
                self.iteration_counter += 1
                logging.debug(f"Iteration count: {self.iteration_counter}")
                return get_recommendation_strength_hybrid(belief_in_model_cf=belief_in_model_cf,
                                                          belief_in_model_cb=belief_in_model_cb,
                                                          recommendation_coefficient_cf=recommendation_coefficient_cf,
                                                          recommendation_coefficient_cb=recommendation_coefficient_cb)

    def get_model_name(self):
        return self.model_name

    def load_saved_results(self, belief_in_model_cb, belief_in_model_cf):
        self.point_to_save = '{}_{}'.format(belief_in_model_cb, belief_in_model_cf)
        recs_df = self.load_recommendation_strength()
        return recs_df

    def recommend_items_hybrid(self, user_id, user_profiles, belief_in_model_cb, belief_in_model_cf, items_to_ignore=[],
                               limit_num_of_recommendations=TOP_N, verbose=False):

        # For avoiding unnecessary computations
        # self.recs_df = self.load_saved_results(belief_in_model_cb, belief_in_model_cf)

        # Getting the top N from Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items_cb(user_id=user_id, user_profiles=user_profiles,
                                                          items_to_ignore=items_to_ignore, verbose=verbose,
                                                          limit_num_of_recommendations=1000000).rename(
            columns={'recommendation_strength': 'recommendation_strengthCB'})

        # Getting the top N from Collaborative filtering recommendations
        cf_recs_df = (self.cf_rec_model.recommend_items_cf(user_id=user_id,
                                                           items_to_ignore=items_to_ignore, verbose=verbose,
                                                           limit_num_of_recommendations=1000000).rename(
            columns={'recommendation_strength': 'recommendation_strengthCF'}
        ))

        # Combining the results by post_id
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how='outer',
                                   left_on='post_id',
                                   right_on='post_id').fillna(0.0)

        logging.debug("recs_df:")
        logging.debug(recs_df)

        recommender_inputs = []
        # TODO: Normalize the recommendation_strength

        model_type_cb = 'cb'  # Static parameter for CB
        model_type_cf = 'cf'  # Static parameter for CF

        scaler = MinMaxScaler()

        recs_df[['recommendation_strengthCB_normalized',
                 'recommendation_strengthCF_normalized']] = scaler.fit_transform(
            recs_df[['recommendation_strengthCB', 'recommendation_strengthCF']]
        )

        # NOTICE: rounding also should make computations faster since cache is used to cache the params.
        recs_df['recommendation_strengthCB_normalized'] = recs_df[
            'recommendation_strengthCB_normalized'].round(2)
        recs_df['recommendation_strengthCF_normalized'] = recs_df[
            'recommendation_strengthCF_normalized'].round(2)

        # Initialize a counter column to 0
        # TODO: Normalize the recommendation_strength
        self.iteration_counter = 0

        if self.fuzzy_expert_enabled:

            # Variant 3:
            # Use a hybrid fuzzy system (evaluating both cf and cb at the same time)
            recs_df['recommendation_strengthFuzzy'] = None
            recs_df['recommendation_strengthFuzzy'] = recs_df.apply(
                lambda x: self.get_recommendation_strength_hybrid_with_logging(
                    belief_in_model_cf,
                    belief_in_model_cb,
                    x['recommendation_strengthCF_normalized'],
                    x['recommendation_strengthCB_normalized'],
                    x['recommendation_strengthFuzzy']
                ),
                axis=1
            )
            recs_df['recommendation_strengthHybrid'] = (recs_df['recommendation_strengthCB']
                                                        + recs_df['recommendation_strengthCF']) * recs_df[
                                                           'recommendation_strengthFuzzy']
            # Variant 4: multiply with the cb and cf
            # self.recs_df['recommendation_strengthHybrid'] = self.recs_df['recommendation_strengthCF_normalized'] * self.recs_df['recommendation_strengthCB_normalized'] * self.recs_df['recommendation_strengthHybrid']
        else:
            # Computing a hybrid recommendation score based on CF and CB scores
            # self.recs_df['recommendation_strengthHybrid']
            # = self.recs_df['recommendation_strengthCB'] * self.recs_df['recommendation_strengthCF']
            recs_df['recommendation_strengthHybrid'] = (
                    recs_df['recommendation_strengthCB'] * 1.0
                    + recs_df['recommendation_strengthCF'] * 1.0)

        # Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recommendation_strengthHybrid', ascending=False).head(limit_num_of_recommendations)
        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='post_id',
                                                          right_on='post_id')[
                ['recommendation_strengthHybrid', 'post_id', 'title', 'slug']]

        return recommendations_df


import numpy as np
from scipy.stats import beta


class BayesianHybridRecommender(HybridRecommender):
    def __init__(self, cb_rec_model, cf_rec_model, items_df, fuzzy_expert_enabled=False):
        super().__init__(cb_rec_model, cf_rec_model, items_df)
        self.cb_prior = beta(1, 1)
        self.cf_prior = beta(1, 1)
        self.model_name = 'Bayesian Hybrid'
        self.fuzzy_expert_enabled = fuzzy_expert_enabled
        self.beliefs_file = 'recommender_beliefs.csv'
        self.cb_prior, self.cf_prior = self.load_beliefs()

    def update_belief(self, model_type, success, failure):
        if model_type == 'cb':
            self.cb_prior = beta(self.cb_prior.args[0] + success, self.cb_prior.args[1] + failure)
        elif model_type == 'cf':
            self.cf_prior = beta(self.cf_prior.args[0] + success, self.cf_prior.args[1] + failure)
        self.save_beliefs()  # Save updated beliefs after each update

    def load_beliefs(self):
        if os.path.exists(self.beliefs_file):
            with open(self.beliefs_file, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) > 1:  # Check if file has data
                    cb_a, cb_b, cf_a, cf_b = map(float, rows[1])
                    return beta(cb_a, cb_b), beta(cf_a, cf_b)
        return beta(1, 1), beta(1, 1)  # Default priors if file doesn't exist or is empty

    def save_beliefs(self):
        with open(self.beliefs_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cb_a', 'cb_b', 'cf_a', 'cf_b'])
            writer.writerow([self.cb_prior.args[0], self.cb_prior.args[1],
                             self.cf_prior.args[0], self.cf_prior.args[1]])

    def get_model_weight(self, model_type):
        if model_type == 'cb':
            return self.cb_prior.mean()
        elif model_type == 'cf':
            return self.cf_prior.mean()

    def recommend_items_bayesian_hybrid(self, user_id, user_profiles, items_to_ignore=[], limit_num_of_recommendations=TOP_N, verbose=False):

        cb_weight = self.get_model_weight('cb')
        cf_weight = self.get_model_weight('cf')

        cb_recs_df = self.get_cb_recommendations(user_id, user_profiles, items_to_ignore, limit_num_of_recommendations=limit_num_of_recommendations)
        cf_recs_df = self.get_cf_recommendations(user_id, items_to_ignore, limit_num_of_recommendations=limit_num_of_recommendations)

        cb_recs_df = cb_recs_df.rename(columns={'recommendation_strength': 'recommendation_strengthCB'})
        cf_recs_df = cf_recs_df.rename(columns={'recommendation_strength': 'recommendation_strengthCF'})
        recs_df = cb_recs_df.merge(cf_recs_df, how='outer', on='post_id').fillna(0.0)

        logging.debug("cb_weight:")
        logging.debug(cb_weight)
        logging.debug("cf_weight:")
        logging.debug(cf_weight)

        if self.fuzzy_expert_enabled:
            # Variant 3:
            # Use a hybrid fuzzy system (evaluating both cf and cb at the same time)
            recs_df['recommendation_strengthFuzzy'] = None
            recs_df['recommendation_strengthFuzzy'] = recs_df.apply(
                lambda x: self.get_recommendation_strength_hybrid_with_logging(
                    cb_weight,
                    cf_weight,
                    x['recommendation_strengthCF'],
                    x['recommendation_strengthCB'],
                    x['recommendation_strengthFuzzy']
                ),
                axis=1
            )
            recs_df['recommendation_strengthHybrid'] = ((recs_df['recommendation_strengthCB']
                                                         + recs_df['recommendation_strengthCF']) *
                                                        recs_df['recommendation_strengthFuzzy'])
        else:
            recs_df['recommendation_strengthHybrid'] = (recs_df['recommendation_strengthCB'] * cb_weight +
                                                        recs_df['recommendation_strengthCF'] * cf_weight)

        recs_df['recommendation_strength'] = recs_df['recommendation_strengthHybrid']

        recs_df = recs_df.sort_values('recommendation_strength', ascending=False).head(limit_num_of_recommendations)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')
            recs_df = recs_df.merge(self.items_df, how='left', left_on='post_id', right_on='post_id')[
                ['recommendation_strength', 'post_id', 'title', 'slug']]

        return recs_df

    def get_model_name(self):
        return self.model_name


def inspect_interactions(person_id, interactions_test_indexed_df, interactions_train_indexed_df, test_set=True):
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df

    if person_id not in interactions_df.index:
        logging.error("Invalid user_id: " + str(person_id))
        return None

    logging.debug("interactions_df:")
    logging.debug(interactions_df)
    logging.debug(interactions_df.loc[person_id])

    return interactions_df.loc[person_id].merge(articles_df, how='left',
                                                left_on='post_id',
                                                right_on='post_id') \
        .sort_values('recommendation_strength', ascending=False)[
        ['recommendation_strength', 'post_id', 'title', 'slug']
    ]


def init_user_interaction_recommender(belief_in_model_cb=None,
                                      belief_in_model_cf=None,
                                      belief_in_interaction_strength_views_global=None,
                                      belief_in_interaction_strength_likes_global=None,
                                      belief_in_liked=None, belief_in_viewed=None,
                                      min_num_of_interactions: Optional[int] = 5,
                                      limit_num_of_recommendations_recommended=TOP_N,
                                      num_of_interactions: Optional[int] = None,
                                      num_of_users: Optional[int] = None,
                                      checkpoint_file='recommender_checkpoint.pkl',
                                      fuzzy_interactions_global=False,
                                      fuzzy_interactions_user=False,
                                      fuzzy_ensemble=False):
    interaction_strength_iteration_counter = 0

    user_thumbs = load_user_thumbs()
    interactions_df_likes = pd.DataFrame.from_dict(user_thumbs, orient='columns')
    # interactions_df_likes = interactions_df_likes[interactions_df_likes.value != 0]
    interactions_df_likes['interaction_type'] = 'LIKE'

    user_views = load_user_view_histories()
    interactions_df_views = pd.DataFrame.from_dict(user_views, orient='columns')
    interactions_df_views['interaction_type'] = 'VIEW'

    if fuzzy_interactions_global is False:
        # Original implementation
        event_type_strength = {
            'VIEW': 1.0,
            'LIKE': 2.0,
        }
    else:
        # Our modification using the fuzzy system
        event_type_strength = {
            'VIEW': 0.0,
            'LIKE': 0.0,
        }
        logging.debug("interactions_df_views:")
        logging.debug(interactions_df_views)
        logging.debug("interactions_df_likes:")
        logging.debug(interactions_df_likes)

        # TODO: Replace this with the average num. of interactions
        num_of_interaction_likes = len(interactions_df_likes)
        num_of_interaction_views = len(interactions_df_views)

        average_interactions_likes = \
            interactions_df_likes.groupby('interaction_type')['user_id'].value_counts().groupby(
                'interaction_type').mean().iloc[0]
        average_interactions_views = \
            interactions_df_views.groupby('interaction_type')['user_id'].value_counts().groupby(
                'interaction_type').mean().iloc[0]

        max_interactions_likes = interactions_df_likes.groupby(['user_id', 'interaction_type']).size().groupby(
            'interaction_type').max().iloc[0]
        max_interactions_views = interactions_df_views.groupby(['user_id', 'interaction_type']).size().groupby(
            'interaction_type').max().iloc[0]

        logging.debug("descriptive stats:")
        logging.debug(interactions_df_likes.groupby('interaction_type')['user_id'].value_counts())
        logging.debug(interactions_df_views.groupby('interaction_type')['user_id'].value_counts())

        recommendation_strength = get_interaction_strength('view',
                                                           belief_in_interaction_strength_views_global,
                                                           average_interactions_views, max_interactions_views)
        logging.debug("recommendation_strength:")
        logging.debug(recommendation_strength)
        event_type_strength['LIKE'] = recommendation_strength

        recommendation_strength = get_interaction_strength('like',
                                                           belief_in_interaction_strength_likes_global,
                                                           average_interactions_likes, max_interactions_likes)

        logging.debug("recommendation_strength:")
        logging.debug(recommendation_strength)
        event_type_strength['VIEW'] = recommendation_strength

    tested_user_profile_id = random.choice(interactions_df_likes['user_id'].values.tolist())
    logging.info("""Tested user profile id: {}""".format(tested_user_profile_id))

    if tested_user_profile_id in interactions_df_views['user_id']:
        logging.info("""Tested user profile id: {}""".format(tested_user_profile_id))
    # NOTE: Originally recommendation_strength and _type => event_*
    interactions_df = pd.concat([interactions_df_likes, interactions_df_views])
    interactions_df['recommendation_strength'] = interactions_df['interaction_type'].apply(
        lambda x: 0.0 if x is False else event_type_strength[x]
    )

    print("interactions_df:")
    print(interactions_df)

    users_interactions_count_df = interactions_df.groupby(['user_id', 'post_id']).size().groupby('user_id').size()
    print('# users: %d' % len(users_interactions_count_df))
    users_with_enough_interactions_df = \
        users_interactions_count_df[users_interactions_count_df >= min_num_of_interactions].reset_index()[
            ['user_id']]
    print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))

    print('# of interactions: %d' % len(interactions_df))
    interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df,
                                                                how='right',
                                                                left_on='user_id',
                                                                right_on='user_id')
    print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))

    @lru_cache(maxsize=16384)
    def get_interaction_strength_with_logging(model_type, belief_in_model_views, belief_in_model_likes,
                                              num_of_interactions_for_fuzzy, max_interactions_likes,
                                              max_interactions_views):
        nonlocal interaction_strength_iteration_counter
        interaction_strength_iteration_counter += 1
        model_type = model_type.lower()

        logging.debug(f"Iteration count: {interaction_strength_iteration_counter}")

        if model_type == 'like':
            return get_interaction_strength(model_type, belief_in_model_likes,
                                            num_of_interactions_for_fuzzy, max_interactions_likes)
        elif model_type == 'view':
            return get_interaction_strength(model_type, belief_in_model_views,
                                            num_of_interactions_for_fuzzy, max_interactions_views)
        else:
            raise ValueError("model_type must be 'like' or 'view'")

    # Calculate the count of each 'user_id'
    user_id_counts = interactions_from_selected_users_df['user_id'].value_counts()

    # Map the count to a new column
    interactions_from_selected_users_df['num_of_interactions'] = interactions_from_selected_users_df['user_id'].map(
        user_id_counts)

    logging.debug("interactions_from_selected_users_df after value counting:")
    logging.debug(interactions_from_selected_users_df.head(10))
    logging.debug(interactions_from_selected_users_df.columns.tolist())

    if fuzzy_interactions_user is True:
        users_interactions_count_df = interactions_from_selected_users_df.groupby(
            ['user_id', 'post_id']).size().groupby('user_id').size()
        users_with_enough_interactions_df = \
            users_interactions_count_df[users_interactions_count_df >= min_num_of_interactions].reset_index()[
                ['user_id']]
        interactions_from_selected_users_df = pd.merge(
            interactions_from_selected_users_df,
            users_with_enough_interactions_df[['user_id']],
            on='user_id',
            how='inner'
        )
        # Now apply your function using the lambda expression
        interactions_from_selected_users_df['recommendation_strength'] = interactions_from_selected_users_df.apply(
            lambda x: get_interaction_strength_with_logging(
                x['interaction_type'],
                belief_in_viewed,
                belief_in_liked,
                x['num_of_interactions'],
                max_interactions_likes,
                max_interactions_views
            ),
            axis=1
        )

    def smooth_user_preference(x):
        return math.log(1 + x, 2)

    interactions_full_df = (interactions_from_selected_users_df
                            .groupby(['user_id', 'post_id'])
                            .agg({
        'recommendation_strength': lambda x: smooth_user_preference(x.sum()),
        **{col: 'first' for col in interactions_from_selected_users_df.columns
           if col not in ['user_id', 'post_id', 'recommendation_strength']}
    })
                            .reset_index())

    # fillna because the dataframe is merged with the views, we consider the views as positive too
    interactions_full_df['value'] = interactions_full_df['value'].fillna(1)
    interactions_full_df['interaction_positive'] = interactions_full_df['value'].astype(int)
    print('# of unique user/item interactions: %d' % len(interactions_full_df))
    print(interactions_full_df.head(10))

    if num_of_interactions:
        # Get unique user IDs and sample N of them
        unique_user_ids = interactions_full_df['user_id'].unique()
        selected_user_ids = pd.Series(unique_user_ids).sample(n=num_of_interactions, random_state=42, replace=True)

        # Filter the DataFrame to keep all rows for the selected user IDs
        df_to_keep = interactions_full_df[interactions_full_df['user_id'].isin(selected_user_ids)]

        # remove user with too little interactions:
        interactions_full_df = df_to_keep[df_to_keep.groupby('user_id')['user_id']
                                          .transform('size') >= min_num_of_interactions]

    print('# of unique user/item interactions: %d' % len(interactions_full_df))
    print(interactions_full_df.head(10))
    if num_of_users:
        # Get unique user IDs and sample N of them
        unique_user_ids = interactions_full_df['user_id'].unique()
        selected_user_ids = pd.Series(unique_user_ids).sample(n=num_of_users, random_state=42)

        # Filter the DataFrame to keep all rows for the selected user IDs
        interactions_full_df = interactions_full_df[interactions_full_df['user_id'].isin(selected_user_ids)]

    # remove user with too little interactions:
    print('# of unique user/item interactions: %d' % len(interactions_full_df))
    print(interactions_full_df.head(10))
    print(interactions_full_df['user_id'].value_counts())

    if num_of_interactions:
        interactions_full_df = interactions_full_df.sample(num_of_interactions)

    interactions_full_df = interactions_full_df[
        interactions_full_df.groupby('user_id')['user_id'].transform('size') >= min_num_of_interactions]

    # interactions_full_df = interactions_full_df[interactions_full_df['value'].notna()]
    try:
        interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                       stratify=interactions_full_df['user_id'],
                                                                       test_size=0.20,
                                                                       random_state=42)
        interactions_train_df, interactions_validation_df = train_test_split(interactions_train_df,
                                                                             stratify=interactions_train_df['user_id'],
                                                                             test_size=0.25,
                                                                             random_state=42)
    except ValueError as e:
        logging.error("Value error had occurred when trying to split data. "
                      "Probably caused by too few interactions for some users. Full exception: "
                      + str(traceback.format_exc()))
        raise e

    print('# interactions on Train set: %d' % len(interactions_train_df))
    print('# interactions on Test set: %d' % len(interactions_test_df))

    interactions_full_indexed_df = interactions_full_df.set_index('user_id')
    interactions_train_indexed_df = interactions_train_df.set_index('user_id')
    interactions_test_indexed_df = interactions_test_df.set_index('user_id')
    interactions_validation_indexed_df = interactions_validation_df.set_index('user_id')

    print("Loading posts...")
    articles_df = load_posts_df()

    print("articles_df:")
    print(articles_df.head(10))

    print("Creating df of posts.")
    model_evaluator = ModelEvaluator(interactions_full_indexed_df, interactions_train_indexed_df,
                                     interactions_test_indexed_df, interactions_validation_indexed_df,
                                     articles_df, belief_in_model_cb, belief_in_model_cf)
    print("created model evaluator")

    # Computes the most popular items
    item_popularity_df = (interactions_full_df.groupby('post_id')['recommendation_strength']
                          .sum().sort_values(ascending=False).reset_index())
    print(item_popularity_df.head(10))

    # Create model
    popularity_model = PopularityRecommender(interactions_train_df, articles_df)
    print("created popularity model")

    logging.info("Building user profiles:")
    model_builder = ModelBuilder()
    user_profiles = model_builder.build_users_profiles(interactions_train_df, articles_df)

    print('Evaluating Popularity recommendation model...')
    pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model, user_profiles,
                                                                                 limit_num_of_recommendations_recommended)
    fuzzy_description = ""
    if fuzzy_interactions_global is True:
        fuzzy_description += " with fuzzy interactions global"
    if fuzzy_interactions_user is True:
        fuzzy_description += " with fuzzy interactions user"

    print('\nGlobal metrics:\n%s %s' % (pop_global_metrics, fuzzy_description))
    print(pop_detailed_results_df.head(10).to_string())

    logging.debug(user_profiles)
    logging.debug(len(user_profiles))
    logging.debug("Num. of user profiles:")
    logging.debug(len(user_profiles))

    _tfidf_feature_names = model_builder.get_tfidf_feature_names()

    if tested_user_profile_id in user_profiles:
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
                                                                               user_profiles, limit_num_of_recommendations_recommended)
    print('\nGlobal metrics:\n%s' % cb_global_metrics)
    cb_detailed_results_df.head(10)

    # Creating a sparse pivot table with users in rows and items in columns
    users_items_pivot_matrix_df = interactions_train_df.pivot(index='user_id',
                                                              columns='post_id',
                                                              values='recommendation_strength').fillna(0)

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
    try:
        U, sigma, Vt = svds(users_items_pivot_sparse_matrix.todense(), k=NUMBER_OF_FACTORS_MF)
    except ValueError as e:
        logging.error("Value error had occurred when trying to perform SVD. "
                      "Probably caused by too few interactions for some users. Full exception: "
                      + str(traceback.format_exc()))
        logging.info("users_items_pivot_sparse_matrix.shape:")
        logging.info(users_items_pivot_sparse_matrix.shape)
        logging.info("Trying smaller number of factors.")
        NUMBER_OF_FACTORS_MF = 10
        logging.warning(
            "Smaller number of factors: " + str(NUMBER_OF_FACTORS_MF) + " will be used but can lead to poor results!")
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
    logging.debug(cf_recommender_model)
    logging.debug("cf_recommender_model:")

    print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
    cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model, user_profiles,
                                                                               limit_num_of_recommendations_recommended)
    print('\nGlobal metrics:\n%s' % cf_global_metrics)
    logging.debug(cf_detailed_results_df.head(10))

    hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, articles_df,
                                                 fuzzy_expert=False)
    print('Evaluating the Hybrid model...')
    hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model,
                                                                                       user_profiles, limit_num_of_recommendations_recommended)
    print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
    hybrid_detailed_results_df.head(10)

    # Create and train the DeepHybridRecommender
    # merge the interactions and views

    deep_hybrid_recommender_model = DeepHybridRecommender(interactions_full_indexed_df,
                                                          content_based_recommender_model,
                                                          cf_recommender_model,
                                                          articles_df)
    deep_hybrid_recommender_model.fit(interactions_train_df, user_profiles)

    print('Evaluating the Deep Hybrid model...')
    deep_hybrid_global_metrics, deep_hybrid_detailed_results_df = model_evaluator.evaluate_model(
        deep_hybrid_recommender_model, user_profiles)
    print('\nGlobal metrics:\n%s' % deep_hybrid_global_metrics)
    deep_hybrid_detailed_results_df.head(10)


    print('Evaluating the Bayesian Hybrid model...')
    bayesian_hybrid_recommender_model = BayesianHybridRecommender(content_based_recommender_model, cf_recommender_model,
                                                                  articles_df, fuzzy_expert_enabled=fuzzy_ensemble)
    bayesian_hybrid_global_metrics, bayesian_hybrid_detailed_results_df = model_evaluator.evaluate_model(
        bayesian_hybrid_recommender_model, user_profiles, limit_num_of_recommendations_recommended)
    fuzzy_description = ""
    if fuzzy_ensemble:
        fuzzy_description = " with fuzzy expert"

    print('\nGlobal metrics:\n%s %s' % (bayesian_hybrid_global_metrics, fuzzy_description))
    bayesian_hybrid_detailed_results_df.head(10)

    """
    print('Evaluating the Stacking Hybrid model...')
    stacking_hybrid_recommender_model = StackingHybridRecommender(content_based_recommender_model,
                                                                  cf_recommender_model,
                                                                  articles_df)
    print('Fitting the Stacking Hybrid model...')
    stacking_hybrid_recommender_model.fit_stacking_model(user_profiles, interactions_train_df)
    print('Evaluating the Stacking Hybrid model...')
    stacking_hybrid_global_metrics, stacking_hybrid_detailed_results_df = model_evaluator.evaluate_model(
        stacking_hybrid_recommender_model, user_profiles, limit_num_of_recommendations)
    print('\nGlobal metrics:\n%s' % stacking_hybrid_global_metrics)
    stacking_hybrid_detailed_results_df.head(10)
    """
    print('Evaluating the AdaptiveWeightHybridRecommender...')
    adaptive_weight_hybrid_recommender_model = AdaptiveWeightHybridRecommender(content_based_recommender_model,
                                                                               cf_recommender_model,
                                                                               articles_df,
                                                                               fuzzy_ensemble=fuzzy_ensemble
                                                                               )
    adaptive_hybrid_global_metrics, adaptive_hybrid_detailed_results_df = model_evaluator.evaluate_model(
        adaptive_weight_hybrid_recommender_model, user_profiles, limit_num_of_recommendations_recommended)
    fuzzy_description = ""
    if fuzzy_ensemble:
        fuzzy_description = " with fuzzy expert"

    print('\nGlobal metrics adaptive:\n%s %s' % (adaptive_hybrid_global_metrics, fuzzy_description))
    adaptive_hybrid_detailed_results_df.head(10)



    print('Evaluating the Fuzzy Hybrid model...')
    hybrid_fuzzy_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model,
                                                       articles_df,
                                                       fuzzy_expert=fuzzy_ensemble)
    hybrid_fuzzy_global_metrics, hybrid_fuzzy_detailed_results_df = model_evaluator.evaluate_model(
        hybrid_fuzzy_recommender_model,
        user_profiles, limit_num_of_recommendations_recommended)
    print('\nGlobal metrics:\n%s' % hybrid_fuzzy_global_metrics)
    hybrid_fuzzy_detailed_results_df.head(10)



    global_metrics_df = pd.DataFrame(
        [cb_global_metrics, pop_global_metrics, cf_global_metrics, hybrid_global_metrics,
         # bayesian_hybrid_global_metrics,
         hybrid_fuzzy_global_metrics]) \
        .set_index('model_name')
    logging.debug(global_metrics_df)
    recall_at_5_stacking_hybrid = global_metrics_df.loc['StackingHybridRecommender']['recall@5']
    recall_at_10_stacking_hybrid = global_metrics_df.loc['StackingHybridRecommender']['recall@10']
    recall_at_20_stacking_hybrid = global_metrics_df.loc['StackingHybridRecommender']['recall@20']


    inspect_interactions(tested_user_profile_id,
                         interactions_test_indexed_df,
                         interactions_train_indexed_df,
                         test_set=False)
    try:
        _inspect_interactions = inspect_interactions(tested_user_profile_id,
                                                     interactions_test_indexed_df,
                                                     interactions_train_indexed_df,
                                                     test_set=True)
        if not _inspect_interactions.empty:
            _inspect_interactions = _inspect_interactions.head(20)
            logging.debug("inspect_interactions:")
            logging.debug(_inspect_interactions)

        if tested_user_profile_id in user_profiles:
            _inspect_interactions = inspect_interactions(tested_user_profile_id,
                                                         interactions_test_indexed_df,
                                                         interactions_train_indexed_df,
                                                         test_set=True)

            if not _inspect_interactions.empty:
                _inspect_interactions = _inspect_interactions.head(20)
                logging.debug("inspect_interactions:")
                logging.debug(_inspect_interactions)
        if tested_user_profile_id in user_profiles:
            hybrid_fuzzy_recommender_model = hybrid_fuzzy_recommender_model.recommend_items_hybrid(
                tested_user_profile_id,
                user_profiles,
                belief_in_model_cb,
                belief_in_model_cf,
                limit_num_of_recommendations=20,
                verbose=True)
            logging.debug("Hybrid_recommender_model for the tested user profile {}:".format(tested_user_profile_id))
            logging.debug(hybrid_fuzzy_recommender_model)
    except Exception as e:
        logging.error("Exception when trying to inspect interactions. Full exception: " + str(traceback.format_exc()))

    recall_at_5_hybrid = global_metrics_df.loc['Hybrid']['recall@5']
    recall_at_10_hybrid = global_metrics_df.loc['Hybrid']['recall@10']
    recall_at_20_hybrid = global_metrics_df.loc['Hybrid']['recall@20']

    recall_at_5_hybrid_bayesian = global_metrics_df.loc['Bayesian Hybrid']['recall@5']
    recall_at_10_hybrid_bayesian = global_metrics_df.loc['Bayesian Hybrid']['recall@10']
    recall_at_20_hybrid_bayesian = global_metrics_df.loc['Bayesian Hybrid']['recall@20']

    recall_at_5_hybrid_or_fuzzy_hybrid = global_metrics_df.loc['Fuzzy Hybrid']['recall@5']
    recall_at_10_hybrid_or_fuzzy_hybrid = global_metrics_df.loc['Fuzzy Hybrid']['recall@10']
    recall_at_20_hybrid_or_fuzzy_hybrid = global_metrics_df.loc['Fuzzy Hybrid']['recall@20']

    recall_at_5_cf = global_metrics_df.loc['Collaborative Filtering']['recall@5']
    recall_at_10_cf = global_metrics_df.loc['Collaborative Filtering']['recall@10']
    recall_at_20_cf = global_metrics_df.loc['Collaborative Filtering']['recall@20']

    recall_at_5_cb = global_metrics_df.loc['Content-Based']['recall@5']
    recall_at_10_cb = global_metrics_df.loc['Content-Based']['recall@10']
    recall_at_20_cb = global_metrics_df.loc['Content-Based']['recall@20']

    recall_at_5_pop = global_metrics_df.loc['Popularity']['recall@5']
    recall_at_10_pop = global_metrics_df.loc['Popularity']['recall@10']
    recall_at_20_pop = global_metrics_df.loc['Popularity']['recall@20']

    return (recall_at_5_hybrid, recall_at_10_hybrid, recall_at_5_hybrid_or_fuzzy_hybrid,
            recall_at_10_hybrid_or_fuzzy_hybrid, recall_at_5_hybrid_bayesian, recall_at_10_hybrid_bayesian,
            recall_at_5_cf, recall_at_10_cf, recall_at_5_cb,
            recall_at_10_cb, recall_at_5_pop, recall_at_10_pop)


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

    def recommend_items_cf(self, user_id, items_to_ignore=[], limit_num_of_recommendations=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
            .reset_index().rename(columns={user_id: 'recommendation_strength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['post_id'].isin(items_to_ignore)] \
            .sort_values('recommendation_strength', ascending=False) \
            # .head(limit_num_of_recommendations)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='post_id',
                                                          right_on='post_id')[
                ['recommendation_strength', 'post_id', 'title', 'slug']]

        return recommendations_df


class AdaptiveWeightHybridRecommender(HybridRecommender):
    def __init__(self, cb_rec_model, cf_rec_model, items_df, learning_rate=0.01, fuzzy_ensemble=False):
        super().__init__(cb_rec_model, cf_rec_model, items_df)
        self.cb_weight = 0.5
        self.cf_weight = 0.5
        self.learning_rate = learning_rate
        self.model_name = "AdaptiveWeightHybridRecommender"
        self.fuzzy_expert_enabled = fuzzy_ensemble

    def get_model_name(self):
        return self.model_name

    def update_weights(self, cb_performance, cf_performance):
        total_performance = cb_performance + cf_performance
        if total_performance > 0:
            target_cb_weight = cb_performance / total_performance
            self.cb_weight += self.learning_rate * (target_cb_weight - self.cb_weight)
            self.cf_weight = 1 - self.cb_weight

    def recommend_items_adaptive_hybrid(self, user_id, user_profiles, items_to_ignore=[], limit_num_of_recommendations=TOP_N, verbose=False):
        cb_recs_df = self.cb_rec_model.recommend_items_cb(user_id, user_profiles, items_to_ignore, limit_num_of_recommendations=limit_num_of_recommendations)
        cf_recs_df = self.cf_rec_model.recommend_items_cf(user_id, items_to_ignore, limit_num_of_recommendations=limit_num_of_recommendations)

        cb_recs_df = cb_recs_df.rename(columns={'recommendation_strength': 'recommendation_strengthCB'})
        cf_recs_df = cf_recs_df.rename(columns={'recommendation_strength': 'recommendation_strengthCF'})
        recs_df = cb_recs_df.merge(cf_recs_df, how='outer', on='post_id').fillna(0.0)

        logging.debug("cb_weight:")
        logging.debug(self.cb_weight)
        logging.debug("cf_weight:")
        logging.debug(self.cf_weight)

        recs_df['recommendation_strength'] = None
        if self.fuzzy_expert_enabled:
            recs_df['recommendation_strengthFuzzy'] = None

            recs_df['recommendation_strengthFuzzy'] = recs_df.apply(
                lambda x: self.get_recommendation_strength_hybrid_with_logging(
                    self.cb_weight,
                    self.cf_weight,
                    x['recommendation_strengthCF'],
                    x['recommendation_strengthCB'],
                    x['recommendation_strengthFuzzy']
                ),
                axis=1
            )
            recs_df['recommendation_strengthHybrid'] = ( recs_df['recommendation_strengthCB']
                                                               + recs_df['recommendation_strengthCF']) * recs_df[
                                                           'recommendation_strengthFuzzy']
            recs_df['recommendation_strength'] = recs_df['recommendation_strengthHybrid']
        else:
            recs_df['recommendation_strengthHybrid'] = ( recs_df['recommendation_strengthCB'] * self.cb_weight
                                                               + recs_df['recommendation_strengthCF'] * self.cf_weight)

            recs_df['recommendation_strength'] = recs_df['recommendation_strengthHybrid']

        recommendations_df = recs_df.sort_values('recommendation_strength', ascending=False).head(limit_num_of_recommendations)

        return recommendations_df


class ModelEvaluator:

    def __init__(self, interactions_full_indexed_df, interactions_train_indexed_df, interactions_test_indexed_df,
                 interactions_validation_indexed_df,
                 articles_df, belief_in_model_cb, belief_in_model_cf):
        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.interactions_test_indexed_df = interactions_test_indexed_df
        self.interactions_validation_indexed_df = interactions_validation_indexed_df
        self.belief_in_model_cb = belief_in_model_cb
        self.belief_in_model_cf = belief_in_model_cf
        self.articles_df = articles_df

    def get_not_interacted_items_sample(self, user_id, sample_size, seed=42):
        interacted_items = get_items_interacted(user_id, self.interactions_full_indexed_df)
        all_items = set(self.articles_df['post_id'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(list(non_interacted_items), sample_size)

        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, limit_num_of_recommendations):
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, limit_num_of_recommendations))
        return hit, index

    def evaluate_model_for_user(self, model, user_id, user_profiles,
                                interactions_evaluation_indexed_df,  # TODO: This does not work as expected.
                                limit_num_of_recommendations=1000000):
        # Getting the items in test set
        interacted_values_test_set = self.interactions_test_indexed_df.loc[user_id]
        if type(interacted_values_test_set['post_id']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_test_set['post_id'])
        else:
            person_interacted_items_testset = {int(interacted_values_test_set['post_id'])}
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        if model.get_model_name() == "Collaborative Filtering":
            person_recs_df = model.recommend_items_cf(user_id=user_id,
                                                      items_to_ignore=get_items_interacted(user_id,
                                                                                           self.interactions_train_indexed_df),
                                                      limit_num_of_recommendations=limit_num_of_recommendations)
        elif model.get_model_name() == "Popularity":
            person_recs_df = model.recommend_items_popularity(user_id=user_id,
                                                              items_to_ignore=get_items_interacted(user_id,
                                                                                                   self.interactions_train_indexed_df),
                                                              limit_num_of_recommendations=limit_num_of_recommendations)
        elif model.get_model_name() == "Content-Based":
            person_recs_df = model.recommend_items_cb(user_id=user_id,
                                                      user_profiles=user_profiles,
                                                      items_to_ignore=get_items_interacted(user_id,
                                                                                           self.interactions_train_indexed_df),
                                                      limit_num_of_recommendations=limit_num_of_recommendations)

        elif model.get_model_name() == "Hybrid" or model.get_model_name() == "Fuzzy Hybrid":
            person_recs_df = model.recommend_items_hybrid(user_id=user_id,
                                                          user_profiles=user_profiles,
                                                          belief_in_model_cb=self.belief_in_model_cb,
                                                          belief_in_model_cf=self.belief_in_model_cf,
                                                          items_to_ignore=get_items_interacted(user_id,
                                                                                               self.interactions_train_indexed_df),
                                                          limit_num_of_recommendations=limit_num_of_recommendations)
        elif model.get_model_name() == "Bayesian Hybrid":
            person_recs_df = model.recommend_items_bayesian_hybrid(user_id=user_id,
                                                                   user_profiles=user_profiles,
                                                                   items_to_ignore=get_items_interacted(user_id,
                                                                                                        self.interactions_train_indexed_df),
                                                                   limit_num_of_recommendations=limit_num_of_recommendations)

        elif model.get_model_name() == "AdaptiveWeightHybridRecommender":
            person_recs_df = model.recommend_items_adaptive_hybrid(user_id=user_id,
                                                                   user_profiles=user_profiles,
                                                                   items_to_ignore=get_items_interacted(user_id,
                                                                                                        self.interactions_train_indexed_df),
                                                                   limit_num_of_recommendations=limit_num_of_recommendations)
        elif model.get_model_name() == "StackingHybridRecommender":
            person_recs_df = model.recommend_items_stacking_hybrid(user_id=user_id,
                                                                   user_profiles=user_profiles,
                                                                   items_to_ignore=get_items_interacted(user_id,
                                                                                                        self.interactions_train_indexed_df),
                                                                   limit_num_of_recommendations=limit_num_of_recommendations)
        elif model.get_model_name() == "DeepHybridRecommender":
            person_recs_df = model.recommend_items_deep_hybrid(
                user_id=user_id,
                user_profiles=user_profiles,
                items_to_ignore=get_items_interacted(user_id, self.interactions_train_indexed_df),
                limit_num_of_recommendations=limit_num_of_recommendations
            )

        else:
            raise Exception("Unknown model name: " + model.get_model_name())

        logging.debug("Model name: %s" % model.get_model_name())
        logging.debug("recommended df:")
        logging.debug(person_recs_df.head(10))
        logging.debug(person_recs_df.columns)

        hits_at_5_count = 0
        hits_at_10_count = 0
        hits_at_20_count = 0

        if isinstance(model, BayesianHybridRecommender):
            cb_recs_df = model.get_cb_recommendations(user_id, user_profiles,
                                                      get_items_interacted(user_id, self.interactions_train_indexed_df),
                                                      limit_num_of_recommendations=limit_num_of_recommendations)
            cf_recs_df = model.get_cf_recommendations(user_id,
                                                      get_items_interacted(user_id, self.interactions_train_indexed_df),
                                                      limit_num_of_recommendations=limit_num_of_recommendations)

            cb_hits_at_5_count = 0
            cf_hits_at_5_count = 0

            for item_id in person_interacted_items_testset:
                non_interacted_items_sample = self.get_not_interacted_items_sample(user_id,
                                                                                   sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                                                   seed=item_id % (2 ** 32))
                items_to_filter_recs = non_interacted_items_sample.union({item_id})

                cb_valid_recs = cb_recs_df[cb_recs_df['post_id'].isin(items_to_filter_recs)]['post_id'].values
                cf_valid_recs = cf_recs_df[cf_recs_df['post_id'].isin(items_to_filter_recs)]['post_id'].values

                cb_hit_at_5, _ = self._verify_hit_top_n(item_id, cb_valid_recs, 5)
                cf_hit_at_5, _ = self._verify_hit_top_n(item_id, cf_valid_recs, 5)

                cb_hits_at_5_count += cb_hit_at_5
                cf_hits_at_5_count += cf_hit_at_5

                # Evaluate
                valid_recs_df = person_recs_df[person_recs_df['post_id'].isin(items_to_filter_recs)]
                valid_recs = valid_recs_df['post_id'].values

                hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
                hits_at_5_count += hit_at_5
                hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
                hits_at_10_count += hit_at_10
                hit_at_20, index_at_20 = self._verify_hit_top_n(item_id, valid_recs, 20)
                hits_at_20_count += hit_at_20

            cb_success = cb_hits_at_5_count
            cf_success = cf_hits_at_5_count
            cb_failure = interacted_items_count_testset - cb_success
            cf_failure = interacted_items_count_testset - cf_success
            model.update_belief('cb', cb_success, cb_failure)
            model.update_belief('cf', cf_success, cf_failure)

        elif isinstance(model, AdaptiveWeightHybridRecommender):
            cb_recs_df = model.get_cb_recommendations(user_id, user_profiles,
                                                      get_items_interacted(user_id,
                                                                           self.interactions_train_indexed_df),
                                                      limit_num_of_recommendations=limit_num_of_recommendations)
            cf_recs_df = model.get_cf_recommendations(user_id,
                                                      get_items_interacted(user_id,
                                                                           self.interactions_train_indexed_df),
                                                      limit_num_of_recommendations=limit_num_of_recommendations)

            cb_hits_at_5_count = 0
            cf_hits_at_5_count = 0

            logging.debug("Adaptive weights algorithm.")
            for item_id in person_interacted_items_testset:
                logging.debug("Evaluating for: Item id: " + str(item_id) + " User id: " + str(user_id))
                non_interacted_items_sample = self.get_not_interacted_items_sample(user_id,
                                                                                   sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                                                   seed=item_id % (2 ** 32))
                items_to_filter_recs = non_interacted_items_sample.union({item_id})

                cb_valid_recs = cb_recs_df[cb_recs_df['post_id'].isin(items_to_filter_recs)]['post_id'].values
                cf_valid_recs = cf_recs_df[cf_recs_df['post_id'].isin(items_to_filter_recs)]['post_id'].values

                cb_hit_at_5, _ = self._verify_hit_top_n(item_id, cb_valid_recs, 5)
                cf_hit_at_5, _ = self._verify_hit_top_n(item_id, cf_valid_recs, 5)

                cb_hits_at_5_count += cb_hit_at_5
                cf_hits_at_5_count += cf_hit_at_5

                cb_success = cb_hits_at_5_count
                cf_success = cf_hits_at_5_count
                cb_failure = interacted_items_count_testset - cb_success
                cf_failure = interacted_items_count_testset - cf_success

                cb_performance = cb_success / (cb_success + cb_failure)
                cf_performance = cf_success / (cf_success + cf_failure)
                model.update_weights(cb_performance, cf_performance)

                person_recs_df = model.recommend_items_adaptive_hybrid(user_id=user_id,
                                                                       user_profiles=user_profiles,
                                                                       items_to_ignore=get_items_interacted(user_id,
                                                                                                            self.interactions_train_indexed_df),
                                                                       limit_num_of_recommendations=limit_num_of_recommendations,
                                                                       )
                # Evaluate
                valid_recs_df = person_recs_df[person_recs_df['post_id'].isin(items_to_filter_recs)]
                valid_recs = valid_recs_df['post_id'].values

                hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
                hits_at_5_count += hit_at_5
                hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
                hits_at_10_count += hit_at_10
                hit_at_20, index_at_20 = self._verify_hit_top_n(item_id, valid_recs, 20)
                hits_at_20_count += hit_at_20

        else:
            # For each item the user has interacted in test set
            for item_id in person_interacted_items_testset:
                # Getting a random sample of 100 items the user has not interacted with
                non_interacted_items_sample = self.get_not_interacted_items_sample(user_id,
                                                                                   sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                                                   seed=item_id % (2 ** 32))

                items_to_filter_recs = non_interacted_items_sample.union({item_id})

                # Filtering only recommendations that are either the interacted item or from a random sample
                # non-interacted items
                valid_recs_df = person_recs_df[person_recs_df['post_id'].isin(items_to_filter_recs)]
                valid_recs = valid_recs_df['post_id'].values

                hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
                hits_at_5_count += hit_at_5
                hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
                hits_at_10_count += hit_at_10
                hit_at_20, index_at_20 = self._verify_hit_top_n(item_id, valid_recs, 20)
                hits_at_20_count += hit_at_20

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
        # when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)
        recall_at_20 = hits_at_20_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count': hits_at_5_count,
                          'hits@10_count': hits_at_10_count,
                          'hits@20_count': hits_at_20_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10,
                          'recall@20': recall_at_20}
        return person_metrics

    def evaluate_model(self, model, user_profiles, limit_num_of_recommnedations=1000000, evaluation_set='test'):
        # print('Running evaluation for users')
        people_metrics = []
        evaluation_df = pd.DataFrame()
        if evaluation_set == 'test':
            evaluation_df = self.interactions_test_indexed_df
        elif evaluation_set == 'validation':
            evaluation_df = self.interactions_validation_indexed_df

        for idx, user_id in enumerate(list(evaluation_df.index.unique().values)):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, user_id, user_profiles, evaluation_df,
                                                          limit_num_of_recommnedations)
            person_metrics['_user_id'] = user_id
            people_metrics.append(person_metrics)
            print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
            .sort_values('interacted_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())
        global_recall_at_20 = detailed_results_df['hits@20_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())

        global_metrics = {
            'set': evaluation_set,
            'model_name': model.get_model_name(),
            'recall@5': global_recall_at_5,
            'recall@10': global_recall_at_10,
            'recall@20': global_recall_at_20
        }

        return global_metrics, detailed_results_df

from sklearn.linear_model import LogisticRegression


class StackingHybridRecommender(HybridRecommender):

    def __init__(self, cb_rec_model, cf_rec_model, items_df):
        super().__init__(cb_rec_model, cf_rec_model, items_df)
        self.stacking_model = LogisticRegression()
        self.is_fitted = False
        self.model_name = 'StackingHybridRecommender'

    def fit_stacking_model(self, user_profiles, interactions_train_df):
        cb_preds = []
        cf_preds = []
        labels = []

        for _, row in interactions_train_df.iterrows():
            user_id = row['user_id']
            item_id = row['post_id']

            cb_rec = self.cb_rec_model.recommend_items_cb(user_id, user_profiles, limit_num_of_recommendations=None)
            cf_rec = self.cf_rec_model.recommend_items_cf(user_id, limit_num_of_recommendations=None)

            cb_strength = cb_rec[cb_rec['post_id'] == item_id]['recommendation_strength'].values
            cf_strength = cf_rec[cf_rec['post_id'] == item_id]['recommendation_strength'].values

            if len(cb_strength) > 0 and len(cf_strength) > 0:
                cb_preds.append(cb_strength[0])
                cf_preds.append(cf_strength[0])
                labels.append(1)  # Interacted item

                # Add some negative samples
                for _ in range(3):  # 3 negative samples for each positive
                    random_item = np.random.choice(self.items_df['post_id'])
                    cb_strength = cb_rec[cb_rec['post_id'] == random_item]['recommendation_strength'].values
                    cf_strength = cf_rec[cf_rec['post_id'] == random_item]['recommendation_strength'].values
                    if len(cb_strength) > 0 and len(cf_strength) > 0:
                        cb_preds.append(cb_strength[0])
                        cf_preds.append(cf_strength[0])
                        labels.append(0)  # Non-interacted item

        X = np.column_stack([cb_preds, cf_preds])
        y = np.array(labels)
        self.stacking_model.fit(X, y)
        self.is_fitted = True

    def recommend_items_stacking_hybrid(self, user_id, user_profiles, items_to_ignore=[], limit_num_of_recommendations=TOP_N, verbose=False):
        if not self.is_fitted:
            raise Exception("Stacking model is not fitted. Call fit_stacking_model first.")

        cb_recs_df = self.cb_rec_model.recommend_items_cb(user_id, user_profiles, items_to_ignore, limit_num_of_recommendations=None)
        cf_recs_df = self.cf_rec_model.recommend_items_cf(user_id, items_to_ignore, limit_num_of_recommendations=None)

        combined_recs_df = cb_recs_df.merge(cf_recs_df, on='post_id', suffixes=('_cb', '_cf'))
        X_pred = combined_recs_df[['recommendation_strength_cb', 'recommendation_strength_cf']].values
        combined_recs_df['recommendation_strength'] = self.stacking_model.predict_proba(X_pred)[:, 1]

        recommendations_df = combined_recs_df.sort_values('recommendation_strength', ascending=False).head(limit_num_of_recommendations)

        return recommendations_df
    
    def get_model_name(self):
        return self.model_name


class PopularityRecommender:
    MODEL_NAME = 'Popularity'

    def __init__(self, interactions_df, items_df=None):
        # Calculate item popularity based only on training data
        self.popularity_df = (interactions_df.groupby('post_id')['recommendation_strength']
                              .sum().sort_values(ascending=False).reset_index())
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items_popularity(self, user_id, items_to_ignore=[], limit_num_of_recommendations=TOP_N, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['post_id'].isin(items_to_ignore)] \
            .sort_values('recommendation_strength', ascending=False) \
            .head(limit_num_of_recommendations)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='post_id',
                                                          right_on='post_id')[
                ['recommendation_strength', 'post_id', 'title', 'slug']]

        return recommendations_df

from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout

class DeepHybridRecommender(HybridRecommender):
    def __init__(self, interactions_full_indexed_df, cb_rec_model, cf_rec_model, items_df, n_factors=300,
                 max_features=300):
        super().__init__(cb_rec_model, cf_rec_model, items_df)
        self.n_factors = n_factors
        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.items_df = items_df
        self.max_features = max_features

        # Create item ID mapping
        unique_item_ids = self.items_df['post_id'].unique()
        self.item_id_map = {id: i for i, id in enumerate(unique_item_ids)}
        self.reverse_item_id_map = {i: id for id, i in self.item_id_map.items()}

        # Create user ID mapping
        unique_user_ids = self.interactions_full_indexed_df.index.unique()
        self.user_id_map = {id: i for i, id in enumerate(unique_user_ids)}
        self.reverse_user_id_map = {i: id for id, i in self.user_id_map.items()}

        self.model = self._build_model()
        self.model_name = 'DeepHybridRecommender'

        # Create and fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.tfidf_vectorizer.fit(self.items_df['trigrams_full_text'])
        # Transform all item content features
        self.content_features_matrix = self.tfidf_vectorizer.transform(self.items_df['trigrams_full_text']).toarray()

    def _build_model(self):
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)

        user_input = Input(shape=(1,))
        item_input = Input(shape=(1,))

        user_embedding = Embedding(n_users, self.n_factors)(user_input)
        item_embedding = Embedding(n_items, self.n_factors)(item_input)

        user_vecs = Flatten()(user_embedding)
        item_vecs = Flatten()(item_embedding)

        # Add content-based features
        content_features = Input(shape=(self.max_features,))

        x = Concatenate()([user_vecs, item_vecs, content_features])
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[user_input, item_input, content_features], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def fit(self, interactions_df, user_profiles, epochs=10, batch_size=64):
        # Prepare data for training

        labels = interactions_df['interaction_positive'].values

        user_ids = np.array([self.user_id_map[id] for id in interactions_df['user_id'].values])
        item_ids = np.array([self.item_id_map[id] for id in interactions_df['post_id'].values])

        # Get content features for each item
        content_features = np.array(
            [self._get_content_features(item_id) for item_id in interactions_df['post_id'].values])

        self.model.fit(
            [user_ids, item_ids, content_features],
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )

    def _get_content_features(self, item_id):
        item_index = self.items_df[self.items_df['post_id'] == item_id].index[0]
        return self.content_features_matrix[item_index]

    def recommend_items_deep_hybrid(self, user_id, user_profiles, items_to_ignore=[], limit_num_of_recommendations=TOP_N):
        # Generate recommendations using the trained model
        all_items = list(self.item_id_map.values())
        user_id = self.user_id_map[user_id]
        user_ids = np.full(len(all_items), user_id)

        # Reshape inputs to match the expected format
        user_ids = user_ids.reshape(-1, 1)
        all_items = np.array(all_items).reshape(-1, 1)
        content_features = self.content_features_matrix

        # Make predictions
        predictions = self.model.predict([user_ids, all_items, content_features])

        # Create DataFrame with predictions
        recs_df = pd.DataFrame({
            'post_id': [self.reverse_item_id_map[i[0]] for i in all_items],
            'prediction': predictions.flatten()
        })
        recs_df = recs_df[~recs_df['post_id'].isin(items_to_ignore)]

        return recs_df.sort_values('prediction', ascending=False).head(limit_num_of_recommendations)
