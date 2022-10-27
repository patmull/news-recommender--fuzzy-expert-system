import logging
import pickle
import random
import time

import spacy_sentence_bert

from src.recommender_core.data_handling.model_methods.user_methods import UserMethods
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.classifier import \
    Classifier
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods


# TODO: This belongs to rabbitmq_consume
def predict_ratings_for_user_store_to_redis(user_id):
    classifier = Classifier()
    print("Loading BERT multilingual model...")
    bert = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    try:
        classifier.predict_relevance_for_user(user_id=user_id, relevance_by='thumbs', bert_model=bert)
    except ValueError:
        print("Value error occurred when trying to get relevant thumbs for user. Skipping "
              "this user.")
        pass
    try:
        classifier.predict_relevance_for_user(user_id=user_id, relevance_by='ratings', bert_model=bert)
    except ValueError:
        print("Value error occurred when trying to get relevant thumbs for user. Skipping "
              "this user.")
        pass


def predict_ratings_for_all_users_store_to_redis():
    user_methods = UserMethods()
    all_users_df = user_methods.get_users_dataframe()
    classifier = Classifier()
    print("Loading BERT multilingual model...")
    bert = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')
    for user_row in zip(*all_users_df.to_dict("list").values()):
        print("user_id:")
        print(user_row[0])
        try:
            classifier.predict_relevance_for_user(user_id=user_row[0], relevance_by='thumbs', bert_model=bert)
        except ValueError:
            print("Value error occurred when trying to get relevant thumbs for user. Skipping "
                  "this user.")
            pass
        try:
            classifier.predict_relevance_for_user(user_id=user_row[0], relevance_by='ratings', bert_model=bert)
        except ValueError:
            print("Value error occurred when trying to get relevant thumbs for user. Skipping "
                  "this user.")
            pass


def fill_bert_vector_representation(skip_already_filled=True, reversed_order=False, random_order=False, db="pgsql"):
    print("Loading sentence bert multilingual model...")
    bert_model = spacy_sentence_bert.load_model('xx_stsb_xlm_r_multilingual')

    database = DatabaseMethods()
    if skip_already_filled is False:
        database.connect()
        posts = database.get_all_posts()
        database.disconnect()
    else:
        database.connect()
        posts = database.get_not_bert_vectors_filled_posts()
        database.disconnect()

    number_of_inserted_rows = 0

    if reversed_order is True:
        logging.debug("Reversing list of posts...")
        posts.reverse()

    if random_order is True:
        print("Starting random_order iteration...")
        time.sleep(5)
        random.shuffle(posts)

    for post in posts:
        if len(posts) < 1:
            break
        post_id = post[0]
        slug = post[3]
        article_title = post[2]
        article_full_text = post[20]
        current_bert_vector_representation = post[41]
        # TODO: Category should be there too

        print("Prefilling body pre-processed in article: " + slug)

        if skip_already_filled is True:
            if current_bert_vector_representation is None:
                if db == "pgsql":
                    bert_vector_representation_of_current_post = bert_model(article_full_text).vector.reshape(1, -1)
                    bert_vector_representation_of_current_post = pickle\
                        .dumps(bert_vector_representation_of_current_post)
                    database.connect()
                    database.insert_bert_vector_representation(
                        bert_vector_representation=bert_vector_representation_of_current_post,
                        article_id=post_id)
                    database.disconnect()
                else:
                    raise NotImplementedError
            else:
                print("Skipping.")
        else:
            if db == "pgsql":
                bert_vector_representation_of_current_post = bert_model(article_full_text).vector.reshape(1, -1)
                print("article_title")
                print(article_title)
                print("bert_vector_representation_of_current_post (full_text)")
                print(bert_vector_representation_of_current_post)
                database.connect()
                database.insert_bert_vector_representation(
                    bert_vector_representation=bert_vector_representation_of_current_post,
                    article_id=post_id)
                database.disconnect()
            else:
                raise NotImplementedError
            number_of_inserted_rows += 1
            if number_of_inserted_rows > 20:
                print("Refreshing list of posts for finding only not prefilled posts.")
                fill_bert_vector_representation(db=db, skip_already_filled=skip_already_filled,
                                                reversed_order=reversed_order,
                                                random_order=random_order)


"""
** prefill_for_user(user_id) METHOD WAS HERE. REMOVED DUE TO UNCLEAR USE CASE. **
"""