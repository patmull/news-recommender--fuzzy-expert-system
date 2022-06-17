import os

import psycopg2

from content_based_algorithms.prefiller import PreFiller

while True:
    try:
        prefiller = PreFiller()
        prefiller.prefilling_job("doc2vec", "pgsql", full_text=False, reverse=False, random=False)
    except Exception as e:
        print("Exception occured " + str(e))
        print(e)