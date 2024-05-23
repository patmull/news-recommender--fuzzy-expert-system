import logging
from pathlib import Path

from src.news_recommender_core_1_1.database import load_post, load_post_ratings, load_user_thumbs_for_post, load_post_category
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.user_evaluation_results import \
    get_thumbs_evaluations, get_playground_evaluations
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods
import json

"""
Proposed new structure of the results:

[
  {
    "user_id": 1,
    "queried": "article-slug-a",
    "recommended": [
      {
        "k": 1,
        "slug": "article-slug-b",
        "relevant": 1
      },
      {
        "k": 2,
        "slug": "article-slug-c",
        "relevant": 0
      }
      // ...
    ],
    "model": "word2vec",
    "queried_category": "sport"
  },
  {
    "user_id": 2,
    "queried": "article-slug-d",
    "recommended": [
      {
        "k": 1,
        "slug": "article-slug-e",
        "relevant": 1
      },
      {
        "k": 2,
        "slug": "article-slug-f",
        "relevant": 0
      },
      // ...
    ],
    "model": "doc2vec",
    "queried_category": "moda"
  },
  // ...
]
"""


def clean_results_playground():
    df_user_testing_results = get_playground_evaluations()
    json_results_new = []

    # Convert DataFrame to JSON string and then to a dictionary
    json_results_old = json.loads(df_user_testing_results.to_json(orient='records'))

    new_structure = []

    # Iterate over the original results and convert them into the new format
    for result in json_results_old:
        user_id = result['user_id']

        # repair the category entry
        category_name_fix = result['results_part_1']['category'][0].lower()

        map_category_name_fix = {
            'mda': 'moda',
            "domc": "domaci",
            "ostatn": "ostatni",
            "zahrani": "zahranicni",
            "zdrav": "zdravi",
            "regionln": "regionalni",
            "vda": "veda"
        }

        if category_name_fix in map_category_name_fix:
            category_name_fix = map_category_name_fix.get(category_name_fix)

        recommendation_entry = {
            "user_id": user_id,
            "queried": result['query_slug'],
            "recommended": [],
            "model": result['model_name'],
            "queried_category": category_name_fix
            # Assuming the first category is the queried one
        }

        # Iterate over the recommendations for the current user
        for k, rec in enumerate(result['results_part_2']['slug']):
            # Check if relevance data is available
            rel_key = 'relevance'
            relevant = result['results_part_2'][rel_key][k]

            # Add the recommendation to the list
            recommendation_entry['recommended'].append({
                "k": k + 1,
                "slug": rec,
                "relevant": relevant
            })

        # Add the current user's recommendations to the new structure
        new_structure.append(recommendation_entry)

    return new_structure


def save_clean_results_playground():
    json_user_testing_results = clean_results_playground()
    path_to_save = Path('evaluations/playground_results.json')
    with open(path_to_save, 'w', encoding='utf-8') as f:
        json.dump(json_user_testing_results, f, indent=4)
        logging.debug(f"Saved to: {path_to_save}")


def add_post_features():
    with open('evaluations/playground_results.json', 'r', encoding='utf-8') as f:
        json_playground_results = json.loads(f.read())

    database_methods = DatabaseMethods()
    database_methods.connect()

    for json_record in json_playground_results:
        post = load_post(json_record['queried'])

        if post is None:
            continue

        json_record['title'] = post.title
        json_record['excerpt'] = post.excerpt
        json_record['body'] = post.body
        json_record['views'] = post.views
        json_record['published_at'] = str(post.published_at)
        json_record['keywords'] = post.keywords
        json_record['category'] = load_post_category(post)
        json_record['post_id'] = post.id
        json_record['post_ratings'] = load_post_ratings(post)
        json_record['thumbs_ratings'] = load_user_thumbs_for_post(post)

        with open(Path('evaluations/playground_results_with_features.json').as_posix(), 'w', encoding='utf-8') as f:
            json.dump(json_playground_results, f, indent=4, ensure_ascii=False)


def clean_results_thumbs():
    df_user_testing_results = get_thumbs_evaluations()
    return df_user_testing_results
