from news_recommender_core_1_1.evaluation.utils.cleaning_results import clean_results_playground, clean_results_thumbs

print(clean_results_playground().head(10).to_string())
print(clean_results_thumbs().head(10).to_string())
