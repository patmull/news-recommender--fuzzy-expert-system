import os

import pandas as pd
import sqlalchemy.engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import create_engine, inspect, text
from torch.distributed.argparse_util import env

Base = automap_base()

url = sqlalchemy.engine.URL.create(
    drivername="postgresql+psycopg2",
    username=os.environ.get("DB_RECOMMENDER_USER"),
    password=os.environ.get("DB_RECOMMENDER_PASSWORD"),
    host=os.environ.get("DB_RECOMMENDER_HOST"),
    port=os.environ.get("DB_RECOMMENDER_PORT"),
    database=os.environ.get("DB_RECOMMENDER_NAME"),
)
# engine, suppose it has two tables 'user' and 'address' set up
engine = create_engine(url)
inspector = inspect(engine)

# Get the list of table names
table_names = inspector.get_table_names()
print(table_names)

# reflect the tables
Base.prepare(autoload_with=engine)
Post = Base.classes.posts
Rating = Base.classes.ratings
Thumb = Base.classes.thumbs
UserHistory = Base.classes.user_histories

session = Session(engine)


def load_post(slug):

    # mapped classes are now created with names by default
    # matching that of the table name.
    # collection-based relationships are by default named
    # "<classname>_collection"
    """

    :param slug:
    :return:
    """

    """"
    # For the debug:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM posts LIMIT 1"))
        print(result.fetchone())
    """

    with Session(engine) as session:
        # does not work:
        first_post = session.query(Post).filter_by(slug=slug).first()
        return first_post


def load_post_category(post):
    # Load the post along with its related category
    print("relationships:")
    """
    i = inspect(Post)
    referred_classes = [r.mapper.class_ for r in i.relationships]
    print(referred_classes)
    """

    post_category = session.query(Post).filter_by(slug=post.slug).options(joinedload(Post.categories)).first()

    """
    print("post_category:")
    print(post_category.__dict__)

    print("post_categories")
    print(post_category.categories.__dict__)
    print(post_category.categories.title)
    """

    if post:
        # Access the related category
        category_title = post_category.categories.title
        return category_title
    else:
        return None


def load_post_ratings(post):
    """

    :return: user_id, rating
    """
    ratings = []

    print("relationships:")
    i = inspect(Post)
    referred_classes = [r.mapper.class_ for r in i.relationships]
    print(referred_classes)

    print("post.id")
    print(post.id)

    post_ratings = (session
                    .query(Rating)
                    .filter_by(post_id=post.id)
                    .all())

    print("post_ratings:")
    print(post_ratings)

    for post_rating in post_ratings:
        rating = {'user_id': post_rating.user_id, 'value': post_rating.value}
        ratings.append(rating)

    print("ratings:")
    print(ratings)
    return ratings


def load_posts_df():
    posts_query = session.query(Post)
    sql_posts_query = posts_query.statement.compile(dialect=postgresql.dialect())
    print("sql_posts_query:")
    print(sql_posts_query)
    articles_df = pd.DataFrame(engine.connect().execute(text(str(sql_posts_query))))
    return articles_df


def load_user_thumbs():
    ratings = []

    print("relationships:")
    i = inspect(Post)
    post_ratings = (session.query(Thumb).all())

    for post_rating in post_ratings:
        rating = {
            'user_id': post_rating.user_id,
            'post_id': post_rating.post_id,
            'value': post_rating.value
        }
        ratings.append(rating)

    return ratings


def load_user_view_histories():
    ratings = []

    print("relationships:")
    user_histories = (session.query(UserHistory).all())

    for user_history in user_histories:
        rating = {
            'user_id': user_history.user_id,
            'post_id': user_history.post_id
        }
        ratings.append(rating)

    return ratings


def load_user_thumbs_for_post(post):
    """

    :return: user_id, rating
    """
    ratings = []

    print("relationships:")
    i = inspect(Post)
    referred_classes = [r.mapper.class_ for r in i.relationships]
    print(referred_classes)
    post_ratings = (session.query(Thumb)
                    .filter_by(post_id=post.id)
                    .all())

    for post_rating in post_ratings:
        rating = {
            'user_id': post_rating.user_id,
            'value': post_rating.value
        }
        ratings.append(rating)

    return ratings
