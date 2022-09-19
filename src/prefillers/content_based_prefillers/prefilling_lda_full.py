import traceback

from src.recommender_core.recommender_algorithms.content_based_algorithms.prefiller import prefilling_job_content_based

while True:
    try:
        prefilling_job_content_based("lda", "pgsql")
    except Exception as e:
        print("Exception occurred " + str(e))
        traceback.print_exception(None, e, e.__traceback__)