import json

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from content_based_algorithms.helper import Helper
from data_conenction import Database


class CosineTransform:

    def __init__(self):
        self.cosine_sim = None
        self.count_matrix = None
        self.similar_articles = None
        self.sorted_similar_articles = None
        self.database = Database()
        self.helper = Helper()
        self.posts_df = None

    def simple_example(self):
        text = ["London Paris London", "Paris Paris London"]
        cv = CountVectorizer(analyzer='word', min_df=10, stop_words='czech', lowercase=True,
                             token_pattern='[a-zA-Z0-9]{3,}')

        count_matrix = cv.fit_transform(text)

        # print(count_matrix.toarray())

        similarity_scores = cosine_similarity(count_matrix)

        # print(similarity_scores)

    def combine_features(self, row):
        return row['title'] + " " + row['keywords'] + " " + row['excerpt']

    def combine_current_posts(self):

        ##Step 1
        self.database.set_row_var()
        # EXTRACT RESULTS FROM CURSOR
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        ##Step 2: Select Features
        features = ['title', 'excerpt', 'keywords']
        ##Step 3: Create a column in DF which combines all selected features
        for feature in features:
            self.posts_df[feature] = self.posts_df[feature].fillna('')

        helper = Helper()
        self.posts_df["combined_features"] = self.posts_df.apply(self.combine_features, axis=1)

        ##Step 4: Create count matrix from this new combined column
        cv = CountVectorizer(analyzer='word', min_df=10, stop_words='czech', lowercase=True,
                             token_pattern='[a-zA-Z0-9]{3,}')

        self.count_matrix = cv.fit_transform(self.posts_df["combined_features"])

    def cosine_similarity(self):
        ##Step 5: Compute the Cosine Similarity based on the count_matrix
        try:
            # self.cosine_sim = self.cosine_similarity_n_space(self.count_matrix)
            self.cosine_sim = cosine_similarity(self.count_matrix)
        except Exception as ex:
            # print(ex)
            pass
        ## print(self.cosine_sim)

    ## Step 6: Get id of this article from its title

    def article_user_likes(self, slug):
        helper = Helper()
        article_id = helper.get_id_from_slug(slug, self.posts_df)
        # print("article_user_likes: " + slug)
        # print("article_id: ")
        ## print(article_id)
        try:
            self.similar_articles = list(enumerate(self.cosine_sim[article_id]))
        except TypeError as te:
            # print(te)
            pass
        ## print(self.similar_articles)

    ## Step 7: Get a list of similar articles in descending order of similarity score
    def get_list_of_similar_articles(self):
        try:
            self.sorted_similar_articles = sorted(self.similar_articles, key=lambda x: x[1], reverse=True)
        except TypeError as te:
            # print(te)
            pass
        ## print(self.sorted_similar_articles)
        return self.sorted_similar_articles

    ## Step 8: Print titles of first 10 articles
    def get_similar(self):
        i = 0
        list_of_article_slugs = []
        list_returned_dict = {}

        for element in self.sorted_similar_articles:

            list_returned_dict['slug'] = self.helper.get_slug_from_id(element[0], self.posts_df)
            list_returned_dict['coefficient'] = element[1]
            list_of_article_slugs.append(list_returned_dict.copy())
            i = i + 1
            if i > 5:
                break

        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_of_article_slugs)
        return json.dumps(list_of_article_slugs)

    def cosine_similarity_n_space(self, m1, m2=None, batch_size=100):
        assert m1.shape[1] == m2.shape[1] and isinstance(batch_size, int) == True

        ret = np.ndarray((m1.shape[0], m2.shape[0]))

        batches = m1.shape[0] // batch_size

        if m1.shape[0] % batch_size != 0:
            batches = batches + 1

        for row_i in range(0, batches):
            start = row_i * batch_size
            end = min([(row_i + 1) * batch_size, m1.shape[0]])
            rows = m1[start: end]
            sim = cosine_similarity(rows, m2)
            ret[start: end] = sim

        return ret

    def fill_recommended_for_all_posts(self, skip_already_filled):
        database = Database()
        database.connect()
        all_posts = database.get_all_posts()

        for post in all_posts:

            post_id = post[0]
            slug = post[3]
            current_recommended = post[15]

            if skip_already_filled is True:
                if current_recommended is None:
                    actual_recommended_json = self.get_by_param(slug)

                    database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                     article_id=post_id)
                else:
                    print("Skipping.")
            else:
                actual_recommended_json = self.get_by_param(slug)

                database.insert_recommended_json(articles_recommended_json=actual_recommended_json, article_id=post_id)