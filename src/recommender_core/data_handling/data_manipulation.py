# import psycopg2.connector
import logging
import os
import random
import traceback
from pathlib import Path

import psycopg2
import pandas as pd

import redis
from pandas.io.sql import DatabaseError
from typing import List

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from data?manipulation.")


def print_exception_not_inserted(e):
    logging.debug(e)


class DatabaseMethods(object):
    """
    The main class of the database methods.

    """

    def __init__(self, db="pgsql"):
        """

        :param db:
        """
        self.categories_df = None
        self.posts_df = pd.DataFrame()
        self.df = None
        self.cnx = None
        self.cursor = None

        if db == "pgsql":
            if "PYTEST_CURRENT_TEST" in os.environ:
                self.DB_USER = os.environ['DB_RECOMMENDER_TESTING_USER']
                self.DB_PASSWORD = os.environ['DB_RECOMMENDER_TESTING_PASSWORD']
                self.DB_HOST = os.environ['DB_RECOMMENDER_TESTING_HOST']
                self.DB_NAME = os.environ['DB_RECOMMENDER_TESTING_NAME']
            else:

                self.DB_USER = os.environ['DB_RECOMMENDER_USER']
                self.DB_PASSWORD = os.environ['DB_RECOMMENDER_PASSWORD']
                self.DB_HOST = os.environ['DB_RECOMMENDER_HOST']
                self.DB_NAME = os.environ['DB_RECOMMENDER_NAME']
                # TODO: Leave this back on production test

                """
                self.DB_USER = os.environ['DB_MC_PRODUCTION_COPY_USER']
                self.DB_PASSWORD = os.environ['DB_MC_PRODUCTION_COPY_PASSWORD']
                self.DB_HOST = os.environ['DB_MC_PRODUCTION_COPY_HOST']
                self.DB_NAME = os.environ['DB_MC_PRODUCTION_COPY_NAME']
                """
                # Debugging prefillers on local production DB copy

        elif db == "pgsql_heroku_testing":
            self.DB_USER = os.environ['DB_RECOMMENDER_HEROKU_TESTING_USER']
            self.DB_PASSWORD = os.environ['DB_RECOMMENDER_HEROKU_TESTING_PASSWORD']
            self.DB_HOST = os.environ['DB_RECOMMENDER_HEROKU_TESTING_HOST']
            self.DB_NAME = os.environ['DB_RECOMMENDER_HEROKU_TESTING_NAME']
        else:
            raise ValueError("No from selected databases are implemented.")

    def connect(self):
        """

        """
        keepalive_kwargs = {
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 5,
            "keepalives_count": 5,
        }
        self.cnx = psycopg2.connect(user=self.DB_USER,
                                    password=self.DB_PASSWORD,
                                    host=self.DB_HOST,
                                    dbname=self.DB_NAME, **keepalive_kwargs)

        self.cursor = self.cnx.cursor()

    def disconnect(self):
        """

        """
        if self.cursor is not None and self.cnx is not None:
            self.cursor.close()
            self.cnx.close()
        else:
            raise ValueError("Cursor is set to None. Cannot continue with next operation.")

    def get_cnx(self):
        """

        :return:
        """
        return self.cnx

    def set_row_var(self):
        """

        """
        sql_set_var = """SET @row_number = 0;"""
        if self.cursor is not None:
            self.cursor.execute(sql_set_var)
        else:
            raise ValueError("Cursor is set to None. Cannot continue with next operation.")

    # NOTICE: This does not return DataFrame but list of columns!
    def get_all_posts(self):
        """

        :return:
        """
        sql = """SELECT * FROM posts ORDER BY id;"""

        query = sql
        if self.cursor is not None:
            self.cursor.execute(query)
            rs = self.cursor.fetchall()
        else:
            raise ValueError("Cursor is set to None. Cannot continue with next operation.")
        return rs

    def get_all_categories(self):
        """

        :return:
        """
        sql = """SELECT * FROM categories ORDER BY id;"""

        query = sql
        if self.cursor is not None:
            self.cursor.execute(query)
            rs = self.cursor.fetchall()
        else:
            raise ValueError("Cursor is set to None. Cannot continue with next operation.")
        return rs

    def get_all_posts_and_categories(self):
        """

        :return:
        """
        sql = """SELECT * FROM categories ORDER BY id;"""

        query = sql
        if self.cursor is not None:
            self.cursor.execute(query)
            rs = self.cursor.fetchall()
        else:
            raise ValueError("Cursor is set to None. Cannot continue with next operation.")
        return rs

    # TODO: This should be handled in RecommenderMethods. Priority: MEDIUM
    def join_posts_ratings_categories(self):
        """

        """
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['post_id', 'post_title', 'slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
             'all_features_preprocessed']]

    def get_posts_join_categories(self):
        """

        :return:
        """
        sql = """SELECT posts.slug, posts.title, categories.title, posts.excerpt, body,
        keywords, all_features_preprocessed, full_text, body_preprocessed, posts.recommended_tfidf,
        posts.recommended_word2vec, posts.recommended_doc2vec, posts.recommended_lda, posts.recommended_tfidf_full_text,
        posts.recommended_word2vec_full_text, posts.recommended_doc2vec_full_text, posts.recommended_lda_full_text
        FROM posts JOIN categories ON posts.category_id = categories.id;"""

        query = sql
        if self.cursor is not None:
            self.cursor.execute(query)
            rs = self.cursor.fetchall()
        else:
            raise ValueError("Cursor is set to None. Cannot continue with next operation.")
        return rs

    def get_all_users(self, column_name: object = None) -> object:
        """

        :param column_name:
        :return:
        """
        logging.debug("type(column_name)")
        logging.debug(type(column_name))
        if column_name is None:
            sql_query = """SELECT * FROM users ORDER BY id;"""
        else:
            if not isinstance(column_name, str):
                raise TypeError('column_name is not string')
            else:
                # noinspection
                sql_query = 'SELECT {} FROM users ORDER BY id;'.format("id, " + column_name)
        logging.debug("sql_query:")
        logging.debug(sql_query)
        try:
            df = pd.read_sql_query(sql_query, self.get_cnx())
        except DatabaseError as e:
            logging.debug(e)
            logging.debug("Check if name of column in this table.")
            raise e
        return df

    def get_post_by_id(self, post_id: object) -> object:
        """

        :return:
        :param post_id:
        :return:
        """
        query = ("SELECT * FROM posts WHERE id = '%s'" % post_id)
        if self.cursor is not None:
            self.cursor.execute(query)
            rs = self.cursor.fetchall()
        else:
            raise ValueError("Cursor is set to None. Cannot continue with next operation.")
        return rs
    def get_posts_dataframe_from_sql(self) -> object:
        """
        Slower, does load the database with query, but supports BERT vectors loading.
        :return:
        """
        logging.info("Getting posts from SQL...")
        sql = """SELECT * FROM posts ORDER BY id;"""
        # NOTICE: Connection is ok here. Need to stay here due to calling from function that's executing thread
        # operation
        self.connect()
        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        self.disconnect()
        df = df.drop_duplicates(subset=['title'])
        return df

    def get_posts_dataframe_only_with_bert_vectors(self):
        logging.info("Getting posts from SQL...")
        sql = """SELECT * FROM posts WHERE bert_vector_representation IS NOT NULL ORDER BY id;"""
        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    def get_posts_dataframe(self, from_cache=True):
        if from_cache is True:
            # TODO: This seems to not work.
            self.posts_df = self.get_posts_dataframe_from_cache()
        else:
            # TODO: This is slow
            self.posts_df = self.get_posts_dataframe_from_sql()

        self.posts_df = self.posts_df.drop_duplicates(subset=['title'])
        return self.posts_df

    def insert_posts_dataframe_to_cache(self, cached_file_path=None):

        if cached_file_path is None:
            if "PYTEST_CURRENT_TEST" in os.environ:
                cached_file_path = "tests/db_cache/cached_posts_dataframe.pkl"
            else:
                logging.debug("Cached file path is None. Using default model_save_location.")
                cached_file_path = "db_cache/cached_posts_dataframe.pkl"

        logging.debug("Loading posts from SQL for cached file load...")
        self.connect()
        sql = """SELECT * FROM posts ORDER BY id;"""
        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        self.disconnect()
        path_to_save_cache = Path(cached_file_path)
        path_to_save_cache.parent.mkdir(parents=True, exist_ok=True)

        logging.debug("Saving cache to:")
        logging.debug(path_to_save_cache.as_posix())
        # TODO: Some workaround for this? (Convert bytearray to input_string?)
        # Removing bert_vector_representation for not supported column type of pickle
        df_for_save = df.drop(columns=['bert_vector_representation'])

        path = Path(cached_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Saving DB cache to pickle file...")
        df_for_save.to_pickle(path.as_posix())  # dataframe of posts will be stored in selected directory
        return df

    def get_posts_dataframe_from_cache(self):
        logging.debug("Reading cache file...")
        try:
            if "PYTEST_CURRENT_TEST" in os.environ:
                path_to_df = Path("tests/db_cache/cached_posts_dataframe.pkl")
            else:
                path_to_df = Path('db_cache/cached_posts_dataframe.pkl')
            df = pd.read_pickle(path_to_df)
            # read from current directory
        except Exception as e:
            logging.warning("Exception occurred when reading cached file:")
            logging.warning(e)
            logging.warning("Full error:")
            logging.warning((traceback.format_exc()))

            if str(e) == "pickle data was truncated":
                logging.warning("Catched the truncated pickle error, dealing with this by removing the current"
                                "cached file...")

                if "PYTEST_CURRENT_TEST" in os.environ:
                    path_to_df = Path("tests/db_cache/cached_posts_dataframe.pkl")
                else:
                    path_to_df = Path('db_cache/cached_posts_dataframe.pkl')

                os.remove(path_to_df)

            logging.info("Getting posts from SQL.")
            df = self.get_posts_dataframe_from_sql()
        return df

    def get_categories_dataframe(self):
        sql = """SELECT * FROM categories ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        self.categories_df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        return self.categories_df

    def get_ratings_dataframe(self):
        sql = """SELECT * FROM ratings ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        # df = pd.read_sql_query(results, database.get_cnx())
        return df

    def get_user_dataframe(self, user_id):
        sql = """SELECT * FROM users WHERE id = {};""".format(user_id)
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    def get_users_dataframe(self):
        sql = """SELECT * FROM users ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    def get_user_history(self, user_id):
        sql = """SELECT * FROM user_histories WHERE user_id = %(user_id)s ORDER BY created_at DESC;"""
        df = pd.read_sql_query(sql, self.get_cnx(), params={'user_id': user_id})
        return df

    def get_posts_df_users_df_ratings_df(self):
        # EXTRACT RESULTS FROM CURSOR

        sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug, u.id
        AS user_id, u.name,
        r.value AS ratings_values
                    FROM posts p
                    JOIN ratings r ON r.post_id = p.id
                    JOIN users u ON r.user_id = u.id;"""
        # LOAD INTO A DATAFRAME
        df_ratings = pd.read_sql_query(sql_rating, self.get_cnx())

        sql_select_all_users = """SELECT u.id AS user_id, u.name FROM users u;"""
        # LOAD INTO A DATAFRAME
        df_users = pd.read_sql_query(sql_select_all_users, self.get_cnx())

        sql_select_all_posts = """SELECT p.id AS post_id, p.slug FROM posts p;"""
        # LOAD INTO A DATAFRAME
        df_posts = pd.read_sql_query(sql_select_all_posts, self.get_cnx())

        return df_posts, df_users, df_ratings

    def get_user_categories(self, user_id=None):
        if user_id is None:
            sql = """SELECT * FROM user_categories ORDER BY id;"""

            # LOAD INTO A DATAFRAME
            df_user_categories = pd.read_sql_query(sql, self.get_cnx())
            # df = pd.read_sql_query(results, database.get_cnx())
        else:
            sql_user_categories = """SELECT c.slug AS "category_slug" FROM user_categories uc
            JOIN categories c ON c.id = uc.category_id WHERE uc.user_id = (%(user_id)s);"""
            query_params = {'user_id': user_id}
            df_user_categories = pd.read_sql_query(sql_user_categories, self.get_cnx(),
                                                   params=query_params)

            logging.debug("df_user_categories:")
            logging.debug(df_user_categories)
            return df_user_categories
        return df_user_categories

    def insert_keywords(self, keyword_all_types_splitted, article_id):
        # PREPROCESSING
        try:
            query = """UPDATE posts SET keywords = %s WHERE id = %s;"""
            inserted_values = (keyword_all_types_splitted, article_id)
            if self.cursor is not None and self.cnx is not None:
                self.cursor.execute(query, inserted_values)
                self.cnx.commit()
            else:
                raise ValueError("Cursor is set to None. Cannot continue with next operation.")

        except psycopg2.OperationalError as e:
            logging.debug("NOT INSERTED")
            logging.debug("Error:", e)  # errno, sqlstate, msg values
            s = str(e)
            logging.debug("Error:", s)  # errno, sqlstate, msg values
            if self.cnx is not None:
                self.cnx.rollback()

    def get_user_rating_categories(self):

        # EXTRACT RESULTS FROM CURSOR

        sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug AS post_slug, r.value AS ratings_values,
        c.title AS category_title, c.slug AS category_slug, p.created_at AS post_created_at
        FROM posts p
        JOIN ratings r ON r.post_id = p.id
        JOIN users u ON r.user_id = u.id
        JOIN categories c ON c.id = p.category_id
        LEFT JOIN user_categories uc ON uc.category_id = c.id;"""

        df_ratings = pd.read_sql_query(sql_rating, self.get_cnx())
        logging.debug("Loaded ratings from DB.")
        logging.debug(df_ratings)
        logging.debug(df_ratings.columns)

        if 'slug_y' in df_ratings.columns:
            df_ratings = df_ratings.rename(columns={'slug': 'category_slug'})

        return df_ratings

    def get_user_keywords(self, user_id):
        sql_user_keywords = """SELECT tags.name AS "keyword_name" FROM tag_user tu JOIN tags
        ON tags.id = tu.tag_id WHERE tu.user_id = (%(user_id)s); """
        query_params = {'user_id': user_id}
        df_user_categories = pd.read_sql_query(sql_user_keywords, self.get_cnx(), params=query_params)
        logging.debug("df_user_categories:")
        logging.debug(df_user_categories)
        return df_user_categories

    @DeprecationWarning
    def insert_recommended_tfidf_json(self, articles_recommended_json, article_id, db):
        if db == "pgsql":
            try:
                query = """UPDATE posts SET recommended_tfidf = %s WHERE id = %s;"""
                inserted_values = (articles_recommended_json, article_id)
                if self.cursor is not None and self.cnx is not None:
                    self.cursor.execute(query, inserted_values)
                    self.cnx.commit()
                else:
                    raise ValueError("Cursor is set to None. Cannot continue with next operation.")
                logging.debug("Inserted")
            except psycopg2.Error as e:
                logging.debug("NOT INSERTED")
                logging.debug(e.pgcode)
                logging.debug(e.pgerror)
                logging.debug("Full error: ", e)  # errno, sqlstate, msg values
                if self.cnx is not None:
                    self.cnx.rollback()
                pass
        elif db == "redis":
            raise Exception("Redis is not implemented yet.")
        else:
            raise ValueError("Not allowed DB model_variant passed.")

    def insert_recommended_json_content_based(self, method, full_text, articles_recommended_json, article_id, db):
        if db == "pgsql":
            if "PYTEST_CURRENT_TEST" in os.environ:
                if method == "test_prefilled_all" and full_text is False:
                    query = """UPDATE posts SET recommended_test_prefilled_all = %s WHERE id = %s;"""
                else:
                    raise ValueError("Methods %s not implemented" % method)
            else:
                if method == "tfidf" and full_text is False:
                    query = """UPDATE posts SET recommended_tfidf = %s WHERE id = %s;"""
                elif method == "tfidf" and full_text is True:
                    query = """UPDATE posts SET recommended_tfidf_full_text = %s WHERE id = %s;"""
                elif method == "word2vec" and full_text is False:
                    query = """UPDATE posts SET recommended_word2vec = %s WHERE id = %s;"""
                elif method == "word2vec" and full_text is True:
                    query = """UPDATE posts SET recommended_word2vec_full_text = %s WHERE id = %s;"""
                elif method == "doc2vec" and full_text is False:
                    query = """UPDATE posts SET recommended_doc2vec = %s WHERE id = %s;"""
                elif method == "doc2vec" and full_text is True:
                    query = """UPDATE posts SET recommended_doc2vec_full_text = %s WHERE id = %s;"""
                elif method == "lda" and full_text is False:
                    query = """UPDATE posts SET recommended_lda = %s WHERE id = %s;"""
                elif method == "lda" and full_text is True:
                    query = """UPDATE posts SET recommended_lda_full_text = %s WHERE id = %s;"""
                elif method == "word2vec_eval_idnes_1" and full_text is True:
                    query = """UPDATE posts SET recommended_word2vec_eval_1 = %s WHERE id = %s;"""
                elif method == "word2vec_eval_idnes_2" and full_text is True:
                    query = """UPDATE posts SET recommended_word2vec_eval_2 = %s WHERE id = %s;"""
                elif method == "word2vec_eval_idnes_3" and full_text is True:
                    query = """UPDATE posts SET recommended_word2vec_eval_3 = %s WHERE id = %s;"""
                elif method == "word2vec_eval_idnes_4" and full_text is True:
                    query = """UPDATE posts SET recommended_word2vec_eval_4 = %s WHERE id = %s;"""
                elif method == "word2vec_eval_cswiki_1" and full_text is True:
                    query = """UPDATE posts SET recommended_word2vec_eval_cswiki_1 = %s WHERE id = %s;"""
                elif method == "doc2vec_eval_cswiki_1" and full_text is True:
                    query = """UPDATE posts SET recommended_doc2vec_eval_cswiki_1 = %s WHERE id = %s;"""
                else:
                    raise Exception("Method not implemented.")
            try:
                inserted_values = (articles_recommended_json, article_id)
                if self.cursor is not None and self.cnx is not None:
                    self.cursor.execute(query, inserted_values)
                    self.cnx.commit()
                else:
                    raise ValueError("Cursor is set to None. Cannot continue with next operation.")
                logging.debug("Inserted")
            except psycopg2.Error as e:
                logging.debug("NOT INSERTED")
                logging.debug(e.pgcode)
                logging.debug(e.pgerror)
                s = str(e)
                logging.debug("Full Error: ", s)  # errno, sqlstate, msg values
                if self.cnx is not None:
                    self.cnx.rollback()
                pass
        elif db == "redis":
            raise Exception("Redis is not implemented yet.")
        else:
            raise ValueError("Not allowed DB method passed.")

    def insert_all_features_preprocessed(self, preprocessed_all_features, post_id):
        try:
            query = """UPDATE posts SET all_features_preprocessed = %s WHERE id = %s;"""
            inserted_values = (preprocessed_all_features, post_id)
            if self.cursor is not None and self.cnx is not None:
                self.cursor.execute(query, inserted_values)
                self.cnx.commit()
            else:
                raise ValueError("Cursor is set to None. Cannot continue with next operation.")

        except psycopg2.Error as e:
            print_exception_not_inserted(e)

    def insert_preprocessed_body(self, preprocessed_body, article_id):
        try:
            query = """UPDATE posts SET body_preprocessed = %s WHERE id = %s;"""
            inserted_values = (preprocessed_body, article_id)
            if self.cursor is not None and self.cnx is not None:
                self.cursor.execute(query, inserted_values)
                self.cnx.commit()
            else:
                raise ValueError("Cursor is set to None. Cannot continue with next operation.")

        except psycopg2.Error as e:
            print_exception_not_inserted(e)

    def insert_phrases_text(self, bigram_text, article_id, full_text):
        try:
            if full_text is False:
                query = """UPDATE posts SET trigrams_short_text = %s WHERE id = %s;"""
            else:
                query = """UPDATE posts SET trigrams_full_text = %s WHERE id = %s;"""
            inserted_values = (bigram_text, article_id)
            if self.cursor is not None and self.cnx is not None:
                self.cursor.execute(query, inserted_values)
                self.cnx.commit()
            else:
                raise ValueError("Cursor is set to None. Cannot continue with next operation.")

        except psycopg2.Error as e:
            print_exception_not_inserted(e)

    def get_not_prefilled_posts(self, full_text, method):
        if full_text is False:
            if "PYTEST_CURRENT_TEST" in os.environ:
                logging.debug("Getting records from testing DB.")
                if method == "test_prefilled_all":
                    sql = """SELECT * FROM posts WHERE recommended_test_prefilled_all IS NULL ORDER BY id;"""
                elif method == "tfidf":
                    sql = """SELECT * FROM posts WHERE recommended_tfidf IS NULL ORDER BY id;"""
                elif method == "word2vec":
                    sql = """SELECT * FROM posts WHERE recommended_word2vec IS NULL ORDER BY id;"""
                elif method == "doc2vec":
                    sql = """SELECT * FROM posts WHERE recommended_doc2vec IS NULL ORDER BY id;"""
                elif method == "lda":
                    sql = """SELECT * FROM posts WHERE recommended_lda IS NULL ORDER BY id;"""
                else:
                    raise ValueError("Selected method " + method + " not implemented.")
            else:
                if method == "tfidf":
                    sql = """SELECT * FROM posts WHERE recommended_tfidf IS NULL ORDER BY id;"""
                elif method == "word2vec":
                    sql = """SELECT * FROM posts WHERE recommended_word2vec IS NULL ORDER BY id;"""
                elif method == "doc2vec":
                    sql = """SELECT * FROM posts WHERE recommended_doc2vec IS NULL ORDER BY id;"""
                elif method == "lda":
                    sql = """SELECT * FROM posts WHERE recommended_lda IS NULL ORDER BY id;"""
                else:
                    raise ValueError("Selected method " + method + " not implemented.")
        else:
            if method == "tfidf":
                sql = """SELECT * FROM posts WHERE recommended_tfidf_full_text IS NULL ORDER BY id;"""
            elif method == "word2vec":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_full_text IS NULL ORDER BY id;"""
            elif method == "doc2vec":
                sql = """SELECT * FROM posts WHERE recommended_doc2vec_full_text IS NULL ORDER BY id;"""
            elif method == "lda":
                sql = """SELECT * FROM posts WHERE recommended_lda_full_text IS NULL ORDER BY id;"""
            elif method == "word2vec_eval_idnes_1":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_1 IS NULL ORDER BY id;"""
            elif method == "word2vec_eval_idnes_2":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_2 IS NULL ORDER BY id;"""
            elif method == "word2vec_eval_idnes_3":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_3 IS NULL ORDER BY id;"""
            elif method == "word2vec_eval_idnes_4":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_4 IS NULL ORDER BY id;"""
            elif method == "word2vec_eval_cswiki_1":
                sql = """SELECT * FROM posts WHERE recommended_word2vec_eval_cswiki_1 IS NULL ORDER BY id;"""
            elif method == "doc2vec_eval_cswiki_1":
                sql = """SELECT * FROM posts WHERE recommended_doc2vec_eval_cswiki_1 IS NULL ORDER BY id;"""
            else:
                raise ValueError("Selected method " + method + " not implemented.")
        query = sql
        if self.cursor is not None:
            self.cursor.execute(query)
            rs = self.cursor.fetchall()
        else:
            raise ValueError("Cursor is set to None. Cannot continue with next operation.")
        return rs

    def get_not_bert_vectors_filled_posts(self):
        sql = """SELECT * FROM posts WHERE bert_vector_representation IS NULL ORDER BY id;"""
        query = sql
        if self.cursor is not None:
            self.cursor.execute(query)
            rs = self.cursor.fetchall()
        else:
            raise ValueError("Cursor is set to None. Cannot continue with next operation.")
        return rs

    def get_posts_dataframe_from_database(self):
        sql = """SELECT * FROM posts ORDER BY id;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    def get_relevance_testing_dataframe(self):
        sql = """SELECT * FROM relevance_testings ORDER BY id;"""
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    def get_thumbs_dataframe(self):
        sql = """SELECT * FROM thumbs ORDER BY id;"""
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    # TODO: Test this. Priority: LOW-MEDIUM
    def get_posts_with_no_body_preprocessed(self):
        sql = """SELECT * FROM posts WHERE body_preprocessed IS NULL ORDER BY id;"""
        # TODO: can be added also keywords, all_features_preprocessed to make sure they were already added in
        #  Parser module
        query = sql
        if self.cursor is not None:
            self.cursor.execute(query)
            rs = self.cursor.fetchall()
        else:
            raise ValueError("Cursor is set to None. Cannot continue with next operation.")
        return rs

    # TODO: Test this. Priority: LOW-MEDIUM
    @PendingDeprecationWarning
    def get_posts_with_no_keywords(self):
        sql = """SELECT * FROM posts WHERE posts.keywords IS NULL ORDER BY id;"""
        # TODO: can be added also keywords, all_features_preprocessed to make sure they were already added in
        #  Parser module
        query = sql
        if self.cursor is not None:
            self.cursor.execute(query)
            rs = self.cursor.fetchall()
        else:
            raise ValueError("Cursor is set to None. Cannot continue with next operation.")
        return rs

    # TODO: Test this. Priority: LOW-MEDIUM
    def get_posts_with_no_features_preprocessed(self, method):

        supported_columns = ['body_preprocessed', 'all_features_preprocessed', 'keywords',
                             'trigrams_full_text']
        if method in supported_columns:
            sql = """SELECT * FROM posts WHERE {} IS NULL ORDER BY id;""".format(method)
            query = sql
            if self.cursor is not None:
                self.cursor.execute(query)
                rs = self.cursor.fetchall()
            else:
                raise ValueError("Cursor is set to None. Cannot continue with next operation.")
            return rs
        else:
            raise NotImplementedError("Selected column not implemented")

    def get_posts_with_not_prefilled_ngrams_text(self, full_text=True):
        if full_text is False:
            sql = """SELECT * FROM posts WHERE trigrams_short_text IS NULL ORDER BY id;"""
        else:
            sql = """SELECT * FROM posts WHERE trigrams_full_text IS NULL ORDER BY id;"""

        # TODO: can be added also: keywords, all_features_preprocessed to make sure they were already added in
        #  Parser module
        query = sql
        if self.cursor is not None:
            self.cursor.execute(query)
            rs = self.cursor.fetchall()
        else:
            raise ValueError("Cursor is set to None. Cannot continue with next operation.")
        return rs

    def get_posts_users_categories_ratings(self, get_only_posts_with_prefilled_bert_vectors=False, user_id=None):
        if get_only_posts_with_prefilled_bert_vectors is False:
            sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, u.id AS user_id,
            p.slug AS post_slug, r.value AS ratings_values, r.created_at AS ratings_created_at,
            c.title AS category_title, c.slug AS category_slug,
            p.created_at AS post_created_at, p.all_features_preprocessed AS all_features_preprocessed,
            p.full_text AS full_text
            FROM posts p
            JOIN ratings r ON r.post_id = p.id
            JOIN users u ON r.user_id = u.id
            JOIN categories c ON c.id = p.category_id
            LEFT JOIN user_categories uc ON uc.category_id = c.id;"""
        else:
            sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, u.id AS user_id,
            p.slug AS post_slug, r.value AS ratings_values, r.created_at AS ratings_created_at,
            c.title AS category_title, c.slug AS category_slug,
            p.created_at AS post_created_at, p.all_features_preprocessed AS all_features_preprocessed,
            p.full_text AS full_text
            FROM posts p
            JOIN ratings r ON r.post_id = p.id
            JOIN users u ON r.user_id = u.id
            JOIN categories c ON c.id = p.category_id
            LEFT JOIN user_categories uc ON uc.category_id = c.id
            WHERE bert_vector_representation IS NOT NULL;"""

        df_ratings = pd.read_sql_query(sql_rating, self.get_cnx())
        logging.debug("df_ratings")
        logging.debug(df_ratings)

        # ### Keep only newest records of same post_id + user_id combination
        # Order by date of creation
        df_ratings = df_ratings.sort_values(by='ratings_created_at')
        df_ratings = df_ratings.drop_duplicates(['post_id', 'user_id'], keep='last')

        if user_id is not None:
            df_ratings = df_ratings.loc[df_ratings['user_id'] == user_id]

        logging.debug("df_ratings after drop_duplicates")
        logging.debug(df_ratings)

        return df_ratings

    def get_posts_users_categories_thumbs(self, user_id=None, get_only_posts_with_prefilled_bert_vectors=False):

        if get_only_posts_with_prefilled_bert_vectors is False:
            sql_thumbs = """SELECT DISTINCT t.id AS thumb_id, p.id AS post_id, u.id AS user_id, p.slug AS post_slug,
            t.value AS thumbs_values, c.title AS category_title, c.slug AS category_slug,
            p.created_at AS post_created_at, t.created_at AS thumbs_created_at,
            p.all_features_preprocessed AS all_features_preprocessed, p.body_preprocessed AS body_preprocessed,
            p.full_text AS full_text,
            p.trigrams_full_text AS short_text, p.trigrams_full_text AS trigrams_full_text, p.title AS title,
            p.keywords AS keywords,
            p.doc2vec_representation AS doc2vec_representation
            FROM posts p
            JOIN thumbs t ON t.post_id = p.id
            JOIN users u ON t.user_id = u.id
            JOIN categories c ON c.id = p.category_id;"""
        else:
            sql_thumbs = """SELECT DISTINCT t.id AS thumb_id, p.id AS post_id, u.id AS user_id, p.slug AS post_slug,
            t.value AS thumbs_values, c.title AS category_title, c.slug AS category_slug,
            p.created_at AS post_created_at, t.created_at AS thumbs_created_at,
            p.all_features_preprocessed AS all_features_preprocessed, p.body_preprocessed AS body_preprocessed,
            p.full_text AS full_text,
            p.trigrams_full_text AS short_text, p.trigrams_full_text AS trigrams_full_text, p.title AS title,
            p.keywords AS keywords,
            p.doc2vec_representation AS doc2vec_representation
            FROM posts p
            JOIN thumbs t ON t.post_id = p.id
            JOIN users u ON t.user_id = u.id
            JOIN categories c ON c.id = p.category_id
            WHERE bert_vector_representation IS NOT NULL;"""

        df_thumbs = pd.read_sql_query(sql_thumbs, self.get_cnx())

        logging.debug("df_thumbs")
        logging.debug(df_thumbs)
        logging.debug(df_thumbs.columns)

        # ### Keep only newest records of same post_id + user_id combination
        # Order by date of creation
        df_thumbs = df_thumbs.sort_values(by='thumbs_created_at')
        df_thumbs = df_thumbs.drop_duplicates(['post_id', 'user_id'], keep='last')

        if user_id is not None:
            df_thumbs = df_thumbs.loc[df_thumbs['user_id'] == user_id]

        logging.debug("df_thumbs after dropping duplicates")
        logging.debug(len(df_thumbs.index))

        if df_thumbs.empty:
            logging.debug("Dataframe empty. Current user has no thumbs clicks in DB.")
            raise ValueError("There are no thumbs for a given user.")

        return df_thumbs

    def get_posts_users_ratings_df(self):
        # EXTRACT RESULTS FROM CURSOR
        sql_rating = """SELECT r.id AS rating_id, p.id AS post_id, p.slug, u.id AS user_id, u.name,
        r.value AS ratings_values FROM posts p JOIN ratings r ON r.post_id = p.id JOIN users u ON r.user_id = u.id;"""
        # LOAD INTO A DATAFRAME
        df_ratings = pd.read_sql_query(sql_rating, self.get_cnx())
        sql_select_all_users = """SELECT u.id AS user_id, u.name FROM users u;"""
        # LOAD INTO A DATAFRAME
        df_users = pd.read_sql_query(sql_select_all_users, self.get_cnx())
        sql_select_all_posts = """SELECT p.id AS post_id, p.slug FROM posts p;"""
        # LOAD INTO A DATAFRAME
        df_posts = pd.read_sql_query(sql_select_all_posts, self.get_cnx())
        return df_posts, df_users, df_ratings

    def get_sql_columns(self):
        sql = """SELECT * FROM posts LIMIT 1;"""

        # LOAD INTO A DATAFRAME
        df = pd.read_sql_query(sql, self.get_cnx())
        return df.columns

    def insert_bert_vector_representation(self, bert_vector_representation, article_id):
        try:
            query = """UPDATE posts SET bert_vector_representation = %s WHERE id = %s;"""
            inserted_values = (bert_vector_representation, article_id)
            if self.cursor is not None and self.cnx is not None:
                self.cursor.execute(query, inserted_values)
                self.cnx.commit()
            else:
                raise ValueError("Cursor is set to None. Cannot continue with next operation.")

        except psycopg2.Error as e:
            print_exception_not_inserted(e)

    def get_results_dataframe(self):
        sql = """SELECT * FROM relevance_testings ORDER BY id;"""
        df = pd.read_sql_query(sql, self.get_cnx())
        return df

    def insert_recommended_json_user_based(self, recommended_json, user_id, db, method):
        if db == "pgsql" or db == "pgsql_heroku_testing":
            try:
                column_name = "recommended_by_" + method
                query = """UPDATE users SET {} = %s WHERE id = %s;""".format(column_name)
                logging.debug("query used:")
                logging.debug(query)
                inserted_values = (recommended_json, user_id)
                if self.cursor is not None and self.cnx is not None:
                    self.cursor.execute(query, inserted_values)
                    self.cnx.commit()
                else:
                    raise ValueError("Cursor is set to None. Cannot continue with next operation.")
            except psycopg2.Error as e:
                print_exception_not_inserted(e)
        elif db == "redis":
            if method == "hybrid":
                r = get_redis_connection()
                r.set(('user:%s:user-hybrid-recommendation' % str(user_id)), recommended_json)
            elif method == "hybrid_fuzzy":
                r = get_redis_connection()
                r.set(('user:%s:user-hybrid-fuzzy-recommendation' % str(user_id)), recommended_json)
            elif method == 'svd':
                r = get_redis_connection()
                r.set(('user:%s:user-svd-recommendation' % str(user_id)), recommended_json)
            elif method == 'user_keywords':
                r = get_redis_connection()
                r.set(('user:%s:user-keywords-recommendation' % str(user_id)), recommended_json)
            elif method == 'best_rated_by_others_in_user_categories':
                r = get_redis_connection()
                r.set(('user:%s:user-preferences-recommendation' % str(user_id)), recommended_json)
            else:
                raise NotImplementedError("Given method for prefilling not implemnted. "
                                          "Cannot insert a recommended JSON.")
        else:
            raise NotImplementedError("Other database source than PostgreSQL not implemented yet.")

    def null_test_user_prefilled_records(self, user_id: int, db_columns: List[str]):
        """
        Method used for testing purposes.
        @param user_id:
        @param db_columns:
        @return:
        """
        for method in db_columns:
            try:
                query = """UPDATE users SET {} = NULL WHERE id = %(id)s;""".format(method)
                queried_values = {'id': user_id}
                logging.debug("Query used in null_test_user_prefilled_records:")
                logging.debug(query)
                if self.cursor is not None and self.cnx is not None:
                    self.cursor.execute(query, queried_values)
                    self.cnx.commit()
            except psycopg2.Error as e:
                print_exception_not_inserted(e)
                logging.debug("psycopg2.Error occurred while trying to update user:")
                logging.debug(str(e))
                raise e

    def null_prefilled_record(self, db_columns: List[str], post_id=None):
        """
        Method used for testing purposes.
        @param post_id:
        @param user_id:
        @param db_columns:
        @return:
        """
        if post_id is None:
            posts = self.get_posts_dataframe(from_cache=False)
            random_post = posts.sample()
            random_post_id = int(random_post['id'].iloc[0])
        else:
            random_post_id = post_id

        self.connect()

        for method in db_columns:
            try:
                query = """UPDATE posts SET {} = NULL WHERE id = %(id)s;""".format(method)
                queried_values = {'id': random_post_id}
                logging.debug("Query used in null_test_user_prefilled_records:")
                logging.debug(query)
                if self.cursor is not None and self.cnx is not None:
                    self.cursor.execute(query, queried_values)
                    self.cnx.commit()
            except psycopg2.Error as e:
                print_exception_not_inserted(e)
                logging.debug("psycopg2.Error occurred while trying to update posts:")
                logging.debug(str(e))
                raise e
            finally:
                self.disconnect()

    def null_post_test_prefilled_record(self):
        posts = self.get_posts_dataframe(from_cache=False)
        random_post = posts.sample()
        random_post_id = random_post['id'].iloc[0]
        self.connect()

        try:
            query = """UPDATE posts SET recommended_test_prefilled_all = NULL WHERE id = %(id)s;"""
            queried_values = {'id': int(random_post_id)}
            logging.debug("Query used in null_post_test_prefilled_record:")
            logging.debug(query)

            if self.cursor is not None and self.cnx is not None:
                self.cursor.execute(query, queried_values)
                self.cnx.commit()
        except psycopg2.Error as e:
            print_exception_not_inserted(e)
            logging.debug("psycopg2.Error occurred while trying to update user:")
            logging.debug(str(e))
            raise e
        finally:
            self.disconnect()

        return int(random_post_id)

    def set_test_json_in_prefilled_records(self, post_id):
        self.connect()
        try:
            query = """UPDATE posts SET recommended_test_prefilled_all = '[{test: json-test}]' WHERE id = %(id)s;"""
            queried_values = {'id': post_id}
            logging.debug("Query used in null_post_test_prefilled_record:")
            logging.debug(query)
            if self.cursor is not None and self.cnx is not None:
                self.cursor.execute(query, queried_values)
                self.cnx.commit()
        except psycopg2.Error as e:
            print_exception_not_inserted(e)
            logging.debug("psycopg2.Error occurred while trying to update user:")
            logging.debug(str(e))
            raise e
        finally:
            self.disconnect()

    def insert_rating(self, post_id, rounded_rating_value):
        self.connect()
        post_id = post_id.item()
        user_id = random.randint(100000, 999999)

        user_name = "test-user-" + str(user_id)
        user_email = "test" + str(user_id) + "@user.cz"
        user_password = "test-user" + str(user_id)

        try:
            query = """INSERT INTO users (id, name, email, password) VALUES (%s, %s, %s, %s);"""
            queried_values = (user_id, user_name, user_email, user_password)
            logging.debug("Query used:")
            logging.debug(query)
            if self.cursor is not None and self.cnx is not None:
                self.cursor.execute(query, queried_values)
                self.cnx.commit()
        except psycopg2.Error as e:
            print_exception_not_inserted(e)
            logging.debug("psycopg2.Error occurred while trying to update users:")
            logging.debug(str(e))
            raise e
        finally:
            self.disconnect()

        self.connect()
        try:
            query = """INSERT INTO ratings (value, user_id, post_id) VALUES (%s, %s, %s);"""
            queried_values = (rounded_rating_value, user_id, post_id)
            logging.debug("Query used:")
            logging.debug(query)
            if self.cursor is not None and self.cnx is not None:
                self.cursor.execute(query, queried_values)
                self.cnx.commit()
        except psycopg2.Error as e:
            print_exception_not_inserted(e)
            logging.debug("psycopg2.Error occurred while trying to update ratings:")
            logging.debug(str(e))
            raise e
        finally:
            self.disconnect()


class RedisConstants:
    def __init__(self):
        self.boost_fresh_keys = {
            1: {
                'hours': 'settings:boost-fresh:1:hours',
                'coeff': 'settings:boost-fresh:1:coeff',
            },
            2: {
                'hours': 'settings:boost-fresh:2:hours',
                'coeff': 'settings:boost-fresh:2:coeff',
            },
            3: {
                'hours': 'settings:boost-fresh:3:hours',
                'coeff': 'settings:boost-fresh:3:coeff',
            },
            4: {
                'hours': 'settings:boost-fresh:4:hours',
                'coeff': 'settings:boost-fresh:4:coeff',
            }
        }


def get_redis_connection():
    if 'REDIS_PASSWORD' in os.environ:
        redis_password = os.environ['REDIS_PASSWORD']
    else:
        raise EnvironmentError("No 'REDIS_PASSWORD' set in environmental variables."
                               "Not possible to connect to Redis.")

    return redis.StrictRedis(host='redis-13695.c1.eu-west-1-3.ec2.cloud.redislabs.com',
                             port=13695, db=0,
                             username="default",
                             password=redis_password,
                             decode_responses=True)
