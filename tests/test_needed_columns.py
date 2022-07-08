from data_connection import Database


def test_all_features_preprocessed_column():
    database = Database()
    database.connect()
    posts = database.get_posts_with_no_all_features_preprocessed()
    return len(posts)


def test_body_preprocessed_column():
    database = Database()
    database.connect()
    posts = database.get_posts_with_no_body_preprocessed()
    return len(posts)


def test_keywords_column():
    database = Database()
    database.connect()
    posts = database.get_posts_with_no_keywords()
    return len(posts)


def test_prefilled_features_columns():
    all_features_preprocessed = test_all_features_preprocessed_column()
    body_preprocessed = test_body_preprocessed_column()
    keywords = test_keywords_column()

    assert all_features_preprocessed == 0
    assert body_preprocessed == 0
    assert keywords == 0