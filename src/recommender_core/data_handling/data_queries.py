import gc
import json
import random
import traceback
from pathlib import Path
from threading import Thread

# import dropbox
import gensim
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender_core.data_handling.data_handlers import flatten
from src.prefillers.preprocessing.cz_preprocessing import preprocess
from src.recommender_core.recommender_algorithms.content_based_algorithms.similarities import CosineTransformer
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
import os

from typing import Dict, List
import logging

CACHED_FILE_PATH = "db_cache/cached_posts_dataframe.pkl"

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)


def convert_df_to_json(dataframe):
    result = dataframe[["title", "excerpt", "body"]].to_json(orient="records", lines=True)
    parsed = json.loads(result)
    return parsed


def convert_to_json_keyword_based(post_recommendations):
    list_of_article_slugs = []
    post_recommendation_dictionary = post_recommendations.to_dict('records')
    list_of_article_slugs.append(post_recommendation_dictionary.copy())
    # print("------------------------------------")
    # print("JSON:")
    # print("------------------------------------")
    # print(list_of_article_slugs[0])
    return list_of_article_slugs[0]


def convert_dataframe_posts_to_json(post_recommendations, slug, cosine_sim_df):
    list_of_article_slugs = []
    list_of_coefficients = []

    # finding coefficient belonging to recommended posts compared to original post (for which we want to find
    # recommendations)
    if 'post_slug' in post_recommendations:
        post_recommendations = post_recommendations.rename(columns={'post_slug': 'slug'})

    for index, row in post_recommendations.iterrows():
        list_of_coefficients.append(cosine_sim_df.at[row['slug'], slug])

    post_recommendations['coefficient'] = list_of_coefficients
    posts_recommendations_dictionary = post_recommendations.to_dict('records')
    list_of_article_slugs.append(posts_recommendations_dictionary.copy())
    return list_of_article_slugs[0]


def dropbox_file_download(dropbox_file_path, local_folder_name):
    try:
        access_token = os.environ.get('DROPBOX_ACCESS_TOKEN')
        dbx = dropbox.Dropbox(access_token)
        # Files can be displayed here. See this method in older commits.
        dbx.files_download_to_file(dropbox_file_path, local_folder_name)

    except Exception as e:
        print(e)
        return False


def preprocess_single_post_find_by_slug(slug, supplied_json=False):
    recommender_methods = RecommenderMethods()
    post_dataframe = recommender_methods.find_post_by_slug(slug)
    post_dataframe["title"] = post_dataframe["title"].map(lambda s: preprocess(s))
    post_dataframe["excerpt"] = post_dataframe["excerpt"].map(lambda s: preprocess(s))
    if supplied_json is False:
        return post_dataframe
    else:
        # evaluate if this ok
        return post_dataframe.to_json()


def random_hyperparameter_choice(model_variants, vector_size_range, window_range, min_count_range,
                                 epochs_range, sample_range, negative_sampling_variants):
    model_variant = random.choice(model_variants)
    vector_size = random.choice(vector_size_range)
    window = random.choice(window_range)
    min_count = random.choice(min_count_range)
    epochs = random.choice(epochs_range)
    sample = random.choice(sample_range)
    negative_sampling_variant = random.choice(negative_sampling_variants)
    return model_variant, vector_size, window, min_count, epochs, sample, negative_sampling_variant


def get_eval_results_header():
    corpus_title = ['100% Corpus']
    model_results = {'Validation_Set': [],  # type: ignore
                     'Model_Variant': [],
                     'Negative': [],
                     'Vector_size': [],
                     'Window': [],
                     'Min_count': [],
                     'Epochs': [],
                     'Sample': [],
                     'Softmax': [],
                     'Word_pairs_test_Pearson_coeff': [],
                     'Word_pairs_test_Pearson_p-val': [],
                     'Word_pairs_test_Spearman_coeff': [],
                     'Word_pairs_test_Spearman_p-val': [],
                     'Word_pairs_test_Out-of-vocab_ratio': [],
                     'Analogies_test': []
                     }  # type: Dict[str, List]
    return corpus_title, model_results


def save_wordsim(path_to_cropped_wordsim_file):
    df = pd.read_csv('research/word2vec/similarities/WordSim353-cs.csv',
                     usecols=['cs_word_1', 'cs_word_2', 'cs mean'])
    df['cs_word_1'] = df['cs_word_1'].apply(lambda x: gensim.utils.deaccent(preprocess(x)))
    df['cs_word_2'] = df['cs_word_2'].apply(lambda x: gensim.utils.deaccent(preprocess(x)))

    df.to_csv(path_to_cropped_wordsim_file, sep='\t', encoding='utf-8', index=False)


def append_training_results(source, corpus_title, model_variant, negative_sampling_variant, vector_size,
                            window,
                            min_count, epochs, sample, hs_softmax, pearson_coeff_word_pairs_eval,
                            pearson_p_val_word_pairs_eval, spearman_p_val_word_pairs_eval,
                            spearman_coeff_word_pairs_eval, out_of_vocab_ratio, analogies_eval, model_results):
    model_results['Validation_Set'].append(source + " " + corpus_title)
    model_results['Model_Variant'].append(model_variant)
    model_results['Negative'].append(negative_sampling_variant)
    model_results['Vector_size'].append(vector_size)
    model_results['Window'].append(window)
    model_results['Min_count'].append(min_count)
    model_results['Epochs'].append(epochs)
    model_results['Sample'].append(sample)
    model_results['Softmax'].append(hs_softmax)
    model_results['Word_pairs_test_Pearson_coeff'].append(pearson_coeff_word_pairs_eval)
    model_results['Word_pairs_test_Pearson_p-val'].append(pearson_p_val_word_pairs_eval)
    model_results['Word_pairs_test_Spearman_coeff'].append(spearman_coeff_word_pairs_eval)
    model_results['Word_pairs_test_Spearman_p-val'].append(spearman_p_val_word_pairs_eval)
    model_results['Word_pairs_test_Out-of-vocab_ratio'].append(out_of_vocab_ratio)
    model_results['Analogies_test'].append(analogies_eval)
    return model_results


def prepare_hyperparameters_grid():
    negative_sampling_variants = range(5, 20, 5)  # 0 = no negative sampling
    no_negative_sampling = 0  # use with hs_soft_max
    vector_size_range = [50, 100, 158, 200, 250, 300, 450]
    window_range = [1, 2, 4, 5, 8, 12, 16, 20]
    min_count_range = [0, 1, 2, 3, 5, 8, 12]
    epochs_range = [20, 25, 30]
    sample_range = [0.0, 1.0 * (10.0 ** -1.0), 1.0 * (10.0 ** -2.0), 1.0 * (10.0 ** -3.0), 1.0 * (10.0 ** -4.0),
                    1.0 * (10.0 ** -5.0)]

    corpus_title, model_results = get_eval_results_header()
    # noinspection PyPep8
    return negative_sampling_variants, no_negative_sampling, vector_size_range, window_range, \
           min_count_range, epochs_range, sample_range, corpus_title, model_results


def combine_features_from_single_df_row(single_row_df, list_of_features):
    logging.debug("single_row_df")
    logging.debug(single_row_df)
    logging.debug("list_of_features")
    logging.debug(list_of_features)
    return (single_row_df[list_of_features] + ' ').sum(axis=1).str.strip()


class RecommenderMethods:

    def __init__(self):
        self.database = DatabaseMethods()
        self.cached_file_path = Path(CACHED_FILE_PATH)
        self.posts_df = pd.DataFrame()
        self.categories_df = pd.DataFrame()
        self.df = pd.DataFrame()

    def get_posts_dataframe(self, force_update=False, from_cache=True):
        if force_update is True:
            self.database.connect()
            self.posts_df = self.database.insert_posts_dataframe_to_cache(self.cached_file_path)
            self.database.disconnect()
        else:
            logging.debug("Trying reading from cache as default...")
            if os.path.isfile(self.cached_file_path):
                try:
                    logging.debug("Reading from cache...")
                    self.posts_df = self.database.get_posts_dataframe(from_cache=from_cache)
                except Exception as e:
                    logging.warning("Exception occurred when trying to read post from cache:")
                    logging.warning(e)
                    logging.warning(traceback.format_exc())
                    self.posts_df = self.get_df_from_sql_meanwhile_insert_to_cache()
            else:
                self.posts_df = self.get_df_from_sql_meanwhile_insert_to_cache()

        # ** HERE WAS DROP DUPLICATION. ABANDONED TO UNNECESSARZ DUPLICATED OPERATION AND MOVED TO
        # ABOVE CHILDREN METHODS **
        return self.posts_df

    def update_cache_of_posts_df(self):
        logging.info("Updating posts cache...")
        self.database.connect()
        self.posts_df = self.database.insert_posts_dataframe_to_cache(self.cached_file_path)
        self.database.disconnect()

    def get_posts_dataframe_only_with_bert(self):
        self.database.connect()
        self.posts_df = self.database.get_posts_dataframe_only_with_bert_vectors()
        self.database.disconnect()

        return self.posts_df

    # noinspection PyShadowingNames
    def update_cache(self):
        logging.debug("Inserting file to cache...")
        self.database.insert_posts_dataframe_to_cache()

    def get_df_from_sql_meanwhile_insert_to_cache(self):
        logging.debug("Posts not found on cache. Will use PgSQL command.")
        posts_df = self.database.get_posts_dataframe_from_sql()
        # ** HERE WAS A THREADING OF INSERTING SQL TO DB. ABANDONED DUE TO POSSIBLE DB CONNECTION LEAK *
        self.update_cache()
        return posts_df

    def get_df_from_sql(self):
        logging.debug("It was chosen to use PgSQL command.")
        posts_df = self.database.get_posts_dataframe_from_sql()
        return posts_df

    def get_ratings_dataframe(self):
        self.database.connect()
        self.posts_df = self.database.get_ratings_dataframe()
        self.database.disconnect()
        return self.posts_df

    def get_categories_dataframe(self):
        # rename_title (defaul=False): for ensuring that category title does not collide with post title
        self.database.connect()
        self.categories_df = self.database.get_categories_dataframe()
        self.database.disconnect()
        if 'slug_y' in self.categories_df.columns:
            self.categories_df = self.categories_df.rename(columns={'slug_y': 'category_slug'})
        elif 'slug' in self.categories_df.columns:
            self.categories_df = self.categories_df.rename(columns={'slug': 'category_slug'})
        return self.categories_df

    @DeprecationWarning
    def get_user_posts_ratings(self):
        database = DatabaseMethods()

        sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug, u.id AS user_id, u.name, r.value
        AS ratings_values
        FROM posts p
        JOIN ratings r ON r.post_id = p.id
        JOIN users u ON r.user_id = u.id;"""
        # LOAD INTO A DATAFRAME
        df_ratings = pd.read_sql_query(sql_rating, database.get_cnx())
        return df_ratings

    def get_ranking_evaluation_results_dataframe(self):
        self.database.connect()
        results_df = self.database.get_relevance_testing_dataframe()
        self.database.disconnect()
        results_df.reset_index(inplace=True)
        print("self.results_df:")
        print(results_df)
        results_df_ = results_df[
            ['id', 'query_slug', 'results_part_1', 'results_part_2', 'user_id', 'model_name', 'model_variant',
             'created_at']]
        return results_df_

    def get_item_evaluation_results_dataframe(self):
        self.database.connect()
        results_df = self.database.get_thumbs_dataframe()
        self.database.disconnect()
        results_df.reset_index(inplace=True)
        print("self.results_df:")
        print(results_df)
        results_df_ = results_df[
            ['id', 'value', 'user_id',
             'post_id', 'method_section',
             'created_at']]
        return results_df_

    def find_post_by_slug(self, searched_slug, from_cache=True):
        if type(searched_slug) is not str:
            raise ValueError("Entered slug must be a input_string.")
        else:
            if searched_slug == "":
                raise ValueError("Entered input_string is empty.")
            else:
                pass
        posts_df = self.get_posts_dataframe(from_cache=from_cache)
        return posts_df.loc[posts_df['slug'] == searched_slug]

    def get_posts_categories_dataframe(self, only_with_bert_vectors=False, from_cache=True):
        if only_with_bert_vectors is False:
            # Standard way. Does not support BERT vector loading from cached file.
            posts_df = self.get_posts_dataframe(from_cache=from_cache)
        else:
            posts_df = self.get_posts_dataframe_only_with_bert()
        categories_df = self.get_categories_dataframe()

        posts_df = posts_df.rename(columns={'title': 'post_title'})
        categories_df = categories_df.rename(columns={'title': 'category_title'})
        if 'slug_y' in categories_df.columns:
            categories_df = categories_df.rename(columns={'slug_y': 'category_slug'})
        elif 'slug' in categories_df.columns:
            categories_df = categories_df.rename(columns={'slug': 'category_slug'})
        logging.debug("posts_df")
        logging.debug(posts_df.columns)
        logging.debug("categories_df")
        logging.debug(categories_df.columns)

        # To make sure. If database contains by a mistake duplicated rows, this will cause a doubling of a final df rows
        categories_df = categories_df.drop_duplicates()

        self.df = posts_df.merge(categories_df, how='left', left_on='category_id', right_on='id')
        if 'id_x' in self.df.columns:
            self.df = self.df.rename(columns={'id_x': 'post_id'})
        return self.df

    def get_posts_categories_full_text(self):
        posts_df = self.get_posts_dataframe()
        posts_df = posts_df.rename(columns={'title': 'post_title'})
        print("self.posts_df.columns")
        print(posts_df.columns)
        posts_df = posts_df.rename(columns={'slug': 'post_slug'})
        print("self.posts_df.columns")
        print(posts_df.columns)
        categories_df = self.get_categories_dataframe()
        categories_df = categories_df.rename(columns={'title': 'category_title'})

        self.df = self.posts_df.merge(categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        print("df.columns")
        print(self.df.columns)
        if 'post_title' in self.df.columns:
            self.df = self.df.rename({'title': 'post_title', 'slug': 'post_slug'})

        if 'id_x' in self.df.columns:
            self.df = self.df.rename({'id_x': 'post_id'})
        self.df = self.df[
            ['post_id', 'post_title', 'post_slug', 'excerpt', 'body', 'views', 'keywords', 'category_title',
             'description', 'all_features_preprocessed', 'body_preprocessed', 'full_text', 'category_id']]
        return self.df

    # NOTICE: This does not return Dataframe!
    def get_all_posts(self):
        self.database.connect()
        all_posts_df = self.database.get_all_posts()
        self.database.disconnect()
        return all_posts_df

    def get_posts_users_categories_ratings_df(self, only_with_bert_vectors, user_id=None):
        self.database = DatabaseMethods()
        self.database.connect()
        posts_users_categories_ratings_df = self.database \
            .get_posts_users_categories_ratings(user_id=user_id,
                                                get_only_posts_with_prefilled_bert_vectors=only_with_bert_vectors)
        self.database.disconnect()
        return posts_users_categories_ratings_df

    def get_posts_users_categories_thumbs_df(self, only_with_bert_vectors, user_id=None):
        try:
            self.database.connect()
            posts_users_categories_ratings_df = self \
                .database \
                .get_posts_users_categories_thumbs(user_id=user_id,
                                                   get_only_posts_with_prefilled_bert_vectors=only_with_bert_vectors)
            self.database.disconnect()
        except ValueError as e:
            self.database.disconnect()
            raise ValueError("Value error had occurred when trying to get posts for user." + str(e))
        return posts_users_categories_ratings_df

    def get_sql_columns(self):
        """
        Init method from app.py. Need for post's cache sanity check
        @return: list of post's columns from DB
        """
        self.database.connect()
        df_columns = self.database.get_sql_columns()
        self.database.disconnect()
        return df_columns

    def get_sql_num_of_rows(self):
        """
       Init method from app.py. Need for post's cache sanity check
       @return: number of post's rows from DB
       """
        self.database.connect()
        df = self.database.get_posts_dataframe(from_cache=False)
        self.database.disconnect()
        return len(df.index)

    def get_relevance_results_dataframe(self):
        self.database.connect()
        results_df = self.database.get_results_dataframe()
        results_df.reset_index(inplace=True)
        self.database.disconnect()
        print("self.results_df:")
        print(results_df)
        results_df_ = results_df[['id', 'query_slug', 'results_part_1', 'results_part_2', 'results_part_3', 'user_id',
                                  'model_name']]
        return results_df_

    def tokenize_text(self):

        self.df['tokenized_keywords'] = self.df['keywords'] \
            .apply(lambda x: x.split(', '))
        self.df['tokenized'] = self.df.apply(
            lambda row: row['all_features_preprocessed'].replace(str(row['tokenized_keywords']), ''),
            axis=1)
        self.df['tokenized_full_text'] = self.df.apply(
            lambda row: row['body_preprocessed'].replace(str(row['tokenized']), ''),
            axis=1)

        gc.collect()

        self.df[
            'tokenized_all_features_preprocessed'] = self.df.all_features_preprocessed.apply(
            lambda x: x.split(' '))
        gc.collect()
        self.df['tokenized_full_text'] = self.df.tokenized_full_text.apply(
            lambda x: x.split(' '))
        return self.df['tokenized_keywords'] + self.df['tokenized_all_features_preprocessed'] + self.df[
            'tokenized_full_text']

    # TODO: get into common method (possibly model_methods)
    def get_prefilled_full_text(self, slug, variant):
        self.get_posts_dataframe(force_update=False)  # load posts to dataframe
        self.get_categories_dataframe()  # load categories to dataframe
        self.join_posts_ratings_categories()  # joining posts and categories into one table

        found_post = self.find_post_by_slug(slug)
        column_name = None
        if variant == "idnes_short_text":
            column_name = 'recommended_doc2vec'
        elif variant == "idnes_full_text":
            column_name = 'recommended_doc2vec_full_text'
        elif variant == "wiki_eval_1":
            column_name = 'recommended_doc2vec_wiki_eval_1'

        returned_post = found_post[column_name].iloc[0]
        return returned_post

    def get_all_users(self, only_with_id_and_column_named=None):
        self.database.connect()
        df_users = self.database.get_all_users(column_name=only_with_id_and_column_named)
        self.database.disconnect()
        return df_users

    def insert_recommended_json_user_based(self, recommended_json, user_id, db, method, heroku_testing_db=False):

        if db == "pgsql":
            self.database.connect()
            self.database.insert_recommended_json_user_based(recommended_json=recommended_json,
                                                             user_id=user_id, db=db, method=method)
            self.database.disconnect()
        elif db == "pgsql_heroku_testing":
            database_heroku_testing = DatabaseMethods(db="pgsql_heroku_testing")
            database_heroku_testing.connect()
            database_heroku_testing.insert_recommended_json_user_based(recommended_json=recommended_json,
                                                                       user_id=user_id, db=db, method=method)
            database_heroku_testing.disconnect()

        elif db == "redis":
            self.database.insert_recommended_json_user_based(recommended_json=recommended_json,
                                                             user_id=user_id, db=db, method=method)
        else:
            NotImplementedError("Given method not implemented for storing user methods.")

    def remove_test_user_prefilled_records(self, user_id, db_columns):
        self.database = DatabaseMethods()
        self.database.connect()
        self.database.null_test_user_prefilled_records(user_id, db_columns=db_columns)
        self.database.disconnect()

    def get_posts_with_not_prefilled_ngrams_text(self, full_text=True):
        self.database = DatabaseMethods()
        self.database.connect()
        posts = self.database.get_posts_with_not_prefilled_ngrams_text(full_text)
        self.database.disconnect()
        return posts

    def get_not_preprocessed_posts_all_features_column_and_body_preprocessed(self):
        self.database = DatabaseMethods()
        self.database.connect()
        posts_without_all_features_preprocessed = self.database.get_posts_with_no_features_preprocessed(
            method='all_features_preprocessed')
        posts_without_body_preprocessed = self.database.get_posts_with_no_features_preprocessed(
            method='body_preprocessed')
        self.database.disconnect()
        posts = list(set(posts_without_all_features_preprocessed + posts_without_body_preprocessed))
        return posts

    def insert_preprocessed_body(self, preprocessed_body, article_id):
        self.database = DatabaseMethods()
        self.database.connect()
        self.database.insert_preprocessed_body(preprocessed_body, article_id)
        self.database.disconnect()

    def get_posts_with_no_features_preprocessed(self, method):
        self.database = DatabaseMethods()
        self.database.connect()
        posts = self.database.get_posts_with_no_features_preprocessed(method=method)
        self.database.disconnect()
        return posts

    def insert_keywords(self, keyword_all_types_splitted, article_id):
        self.database = DatabaseMethods()
        self.database.connect()
        self.database.insert_keywords(keyword_all_types_splitted=keyword_all_types_splitted,
                                      article_id=article_id)
        self.database.disconnect()

    def insert_all_features_preprocessed_combined(self, preprocessed_text, post_id):
        self.database = DatabaseMethods()
        self.database.connect()
        self.database.insert_all_features_preprocessed(preprocessed_all_features=preprocessed_text,
                                                       post_id=post_id)
        self.database.disconnect()

    def insert_phrases_text(self, bigram_text, article_id, full_text):
        self.database = DatabaseMethods()
        self.database.connect()
        self.database.insert_phrases_text(bigram_text, article_id, full_text)
        self.database.disconnect()


def get_cleaned_text(row):
    return row


class TfIdfDataHandlers:

    def __init__(self, df=None):
        self.df = df
        self.tfidf_vectorizer = TfidfVectorizer()
        self.cosine_sim_df = None
        self.tfidf_tuples = None

    def set_df(self, df):
        self.df = df

    def get_fit_by_feature_(self, feature_name, second_feature=None):
        fit_by_feature = self.get_tfidf_vectorizer(feature_name, second_feature)
        return fit_by_feature

    def most_similar_by_keywords(self, keywords, tupple_of_fitted_matrices, number_of_recommended_posts=20):
        # combining results of all feature types to sparse matrix
        try:
            combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices, dtype=np.float16)
        except ValueError as e:
            raise e

        # computing cosine similarity using matrix with combined features
        print("Computing cosine similarity using matrix with combined features...")
        cosine_transform = CosineTransformer()
        self.cosine_sim_df = cosine_transform.get_cosine_sim_use_own_matrix(combined_matrix1, self.df)
        combined_all = self.get_recommended_posts_for_keywords(keywords=keywords,
                                                               data_frame=self.df,
                                                               k=number_of_recommended_posts)

        df_renamed = combined_all.rename(columns={'post_slug': 'slug'})
        recommended_posts_in_json = convert_to_json_keyword_based(df_renamed)

        return recommended_posts_in_json

    def get_recommended_posts_for_keywords(self, keywords, data_frame, k=10):

        keywords_list = [keywords]
        txt_cleaned = get_cleaned_text(self.df['post_title'] + self.df['category_title'] + self.df['keywords'] +
                                       self.df['excerpt'])
        tfidf = self.tfidf_vectorizer.fit_transform(txt_cleaned)
        tfidf_keywords_input = self.tfidf_vectorizer.transform(keywords_list)
        cosine_similarities = flatten(cosine_similarity(tfidf_keywords_input, tfidf))
        # cosine_similarities = linear_kernel(tfidf_keywords_input, tfidf).flatten()

        data_frame['coefficient'] = cosine_similarities

        closest = data_frame.sort_values('coefficient', ascending=False)[:k]

        closest.reset_index(inplace=True)
        # closest = closest.set_index('index1')
        closest['index1'] = closest.index
        # closest.index.name = 'index1'
        closest.columns.name = 'index'

        print("closest")
        print(closest)
        print(closest.columns)

        return closest[["slug", "coefficient"]]
        # return pd.DataFrame(closest).merge(items).head(k)

    def get_tfidf_vectorizer(self, fit_by, fit_by_2=None):
        """
        Metoda fit: výpočet průměru a rozptylu jednotlivých sloupců z dat.
        Metoda transformace: # transformuje všechny prvky pomocí příslušného průměru a rozptylu.
        """

        self.set_tfid_vectorizer()
        if fit_by_2 is None:
            logging.debug("self.df")
            logging.debug(self.df)
            self.tfidf_tuples = self.tfidf_vectorizer.fit_transform(self.df[fit_by])
        else:
            if self.df[fit_by] is None and self.df[fit_by_2] is None:
                raise ValueError("Both columns %s and %s cannot be None." % (fit_by, fit_by_2))

            if self.df[fit_by] is None and self.df[fit_by_2] is not None:
                logging.warning("Dataframe has missing data in column %s. Consider to run prefilling of this column "
                                "first." % fit_by)
                self.df[fit_by] = self.df[fit_by_2]
            elif self.df[fit_by] is not None and self.df[fit_by_2] is None:
                logging.warning("Dataframe has missing data in column %s. Consider to run prefilling of this column "
                                "first." % fit_by)
                self.df[fit_by] = self.df[fit_by]
            else:
                # Standard way of gettign both columns
                self.df[fit_by] = self.df[fit_by_2] + " " + self.df[fit_by]
            self.tfidf_tuples = self.tfidf_vectorizer.fit_transform(self.df[fit_by])

        return self.tfidf_tuples  # tuples of (document_id, token_id) and tf-idf score for it

    def set_tfid_vectorizer(self):
        # load_texts czech stopwords from file
        filename = Path("src/prefillers/preprocessing/stopwords/czech_stopwords.txt")
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]

        filename = Path("src/prefillers/preprocessing/stopwords/general_stopwords.txt")
        with open(filename, encoding="utf-8") as file:
            general_stopwords = file.readlines()
            general_stopwords = [line.rstrip() for line in general_stopwords]
        # print(cz_stopwords)
        stopwords = cz_stopwords + general_stopwords

        # transforms text to feature vectors that can be used as input to estimator
        self.tfidf_vectorizer = TfidfVectorizer(dtype=np.float32,
                                                stop_words=stopwords)

    def calculate_cosine_sim_matrix(self, tupple_of_fitted_matrices):
        logging.debug("tupple_of_fitted_matrices:")
        logging.debug(tupple_of_fitted_matrices)
        combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices)
        logging.debug("combined_matrix1:")
        logging.debug(combined_matrix1)

        cosine_transform = CosineTransformer()
        self.cosine_sim_df = cosine_transform.get_cosine_sim_use_own_matrix(combined_matrix1, self.df)
        return self.cosine_sim_df

    # # @profile
    def recommend_by_more_features(self, slug, tupple_of_fitted_matrices, num_of_recommendations=20):
        """
        # combining results of all feature types
        # combined_matrix1 = sparse.hstack(tupple_of_fitted_matrices)
        # creating sparse matrix containing mostly zeroes from combined feature tupples

        Example 1: solving linear system A*x=b where A is 5000x5000 but is block diagonal matrix constructed of 500 5x5
        blocks. Setup code:

        As = sparse(rand(5, 5));
        for(i=1:999)
           As = blkdiag(As, sparse(rand(5,5)));
        end;                         %As is made up of 500 5x5 blocks along diagonal
        Af = full(As); b = rand(5000, 1);

        Then you can tests speed difference:

        As operation on sparse As takes .0012 seconds
        Af solving with full Af takes about 2.3 seconds
        """
        self.cosine_sim_df = self.calculate_cosine_sim_matrix(tupple_of_fitted_matrices)

        # getting posts with the highest similarity
        combined_all = self.get_closest_posts(slug, self.cosine_sim_df,
                                              self.df[['slug']], k=num_of_recommendations)

        recommended_posts_in_json = convert_dataframe_posts_to_json(combined_all, slug, self.cosine_sim_df)
        return recommended_posts_in_json

    def get_closest_posts(self, find_by_string, data_frame, items, k=20):
        logging.debug("self.cosine_sim_df:")
        logging.debug(self.cosine_sim_df)
        ix = data_frame.loc[:, find_by_string].to_numpy().argpartition(range(-1, -k, -1))
        closest = data_frame.columns[ix[-1:-(k + 2):-1]]

        # drop post itself
        closest = closest.drop(find_by_string, errors='ignore')

        return pd.DataFrame(closest).merge(items).head(k)

    def get_tupple_of_fitted_matrices(self, fit_by_post_title_matrix):
        print("fit_by_post_title_matrix")
        print(fit_by_post_title_matrix)
        # fit_by_category_matrix = recommender_methods.get_fit_by_feature_('category_title')
        fit_by_excerpt_matrix = self.get_fit_by_feature_('excerpt')
        print("fit_by_excerpt_matrix")
        print(fit_by_excerpt_matrix)
        fit_by_keywords_matrix = self.get_fit_by_feature_('keywords')
        print("fit_by_keywords_matrix")
        print(fit_by_keywords_matrix)

        # join feature tuples into one matrix
        tuple_of_fitted_matrices = (fit_by_post_title_matrix, fit_by_excerpt_matrix, fit_by_keywords_matrix)
        return tuple_of_fitted_matrices


def unique_list(items):
    """
    Getting unique values from list supplied. This is one of the fastest implementation,
    see: https://stackoverflow.com/a/90225/4183655
    @return:
    """
    seen = set()
    for i in range(len(items) - 1, -1, -1):
        it = items[i]
        if it in seen:
            del items[i]
        else:
            seen.add(it)
    return list(seen)


if __name__ == '__main__':
    logging.debug("Testing logging from data_queries module.")
