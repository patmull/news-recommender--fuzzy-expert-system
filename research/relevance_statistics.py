import logging
import random
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, precision_score, balanced_accuracy_score, confusion_matrix, \
    dcg_score, f1_score, jaccard_score, ndcg_score, precision_recall_curve, top_k_accuracy_score, recall_score, \
    PrecisionRecallDisplay
import seaborn as sns

from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier import \
    user_evaluation_results
from src.recommender_core.data_handling.data_queries import RecommenderMethods
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.user_evaluation_results import \
    get_playground_evaluations

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


def try_statistics():
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array(
        [0.5664147315, 0.4877725552, 0.4053835884, 0.4026770756]
    )
    print("AVERAGE PRECISION:")
    y_pred = np.full((1, 4), 1)[0]
    print(average_precision_score(y_true, y_scores))
    print("PRECISION SCORE:")
    print(precision_score(y_true, y_pred, average='weighted'))
    print("BALANCED_ACCURACY:")
    print(balanced_accuracy_score(y_true, y_pred))
    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_true, y_pred))

    true_relevance = y_true
    scores = y_scores
    print("DCG SCORE:")
    print(dcg_score([true_relevance], [y_pred]))
    print("DCG AT K=5:")
    print(dcg_score([true_relevance], [y_pred], k=5))

    print("F1-SCORE:")
    print(f1_score(y_true, y_pred, average='weighted'))

    print("JACCARD SCORE:")
    print(jaccard_score(y_true, y_pred))

    print("NDCG:")
    print(ndcg_score([true_relevance], [y_pred]))
    print("NDCG AT K:")
    print(ndcg_score([true_relevance], [y_pred], k=5))

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    print("PRECISION:")
    print(precision)
    print("RECALL:")
    print(recall)
    print("THRESHOLD:")
    print(thresholds)

    print("PRECISION SCORE:")
    print(precision_score(y_true, y_pred, average=None))

    print("TOP k=5 ACCURACY:")
    print(top_k_accuracy_score(y_true, scores, k=2))

def model_ap(investigate_by='model_name'):
    evaluation_results_df = evaluation_results.get_playground_evaluations()
    print(evaluation_results_df.head(10).to_string())
    dict_of_model_stats = {}
    list_of_models = []
    list_of_models.append([x for x in evaluation_results_df[investigate_by]])

    list_of_aps = []
    list_of_aps.append([average_precision_score(x['relevance'], x['coefficient'])
                        if len(x['relevance']) == len(x['coefficient'])
                        else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                        for x in evaluation_results_df['results_part_2']])
    print(list_of_models)
    print(list_of_aps)

    dict_of_model_stats = {'ap': list_of_aps[0], investigate_by: list_of_models[0]}
    print(dict_of_model_stats)

    model_ap_results = pd.DataFrame.from_dict(dict_of_model_stats)
    """
    list_of_aps.append([print(x['relevance'])
                        for x in evaluation_results_df['results_part_2']])
    """
    return model_ap_results[[investigate_by, 'ap']].groupby([investigate_by]).max()


def model_variant_ap(variant=None):
    evaluation_results_df = evaluation_results.get_playground_evaluations()
    print(evaluation_results_df.head(10).to_string())

    if variant is not None:
        evaluation_results_df = evaluation_results_df.loc[evaluation_results_df['model_variant'] == variant]

    dict_of_model_stats = {}
    list_of_aps = []
    list_of_aps = []
    list_of_aps.append([average_precision_score(x['relevance'], x['coefficient'])
                        if len(x['relevance']) == len(x['coefficient'])
                        else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                        for x in evaluation_results_df['results_part_2']])
    list_of_models = []
    list_of_models.append([x for x in evaluation_results_df['model_variant']])

    print(list_of_aps)
    print(list_of_models)

    dict_of_model_stats = {'ap': list_of_aps[0], 'model_variant': list_of_models[0]}
    print(dict_of_model_stats)

    model_ap_results = pd.DataFrame.from_dict(dict_of_model_stats)
    """
    list_of_aps.append([print(x['relevance'])
                        for x in evaluation_results_df['results_part_2']])
    """
    return model_ap_results[['model_variant', 'ap']].groupby(['model_variant']).mean()

"""
def save_precision_recall_curve(precision, recall):
    # precision, recall, thresholds = precision_recall_curve(y_score, y_test)
    # create precision recall curve

    precision.sort()
    recall.sort()

    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
    pr_display.plot()
    plt.show()
"""


def models_complete_statistics(investigate_by, k=5, save_results_for_every_item=False, crop_by_date=False,
                               last_n_by_date=None, save_results_for_model=False):
    global list_of_slugs, list_of_created_at
    evaluation_results_df = get_playground_evaluations()

    if crop_by_date:
        if last_n_by_date is not None:
            evaluation_results_df = evaluation_results_df.sort_values(by=['created_at'], ascending=False)
            evaluation_results_df = evaluation_results_df.head(last_n_by_date)
        else:
            raise ValueError("When cropping by date, the 'date' argument needs to be set, "
                             "otherwise it will show the date but not crop the date")
    print("evaluation_results_df")
    print(evaluation_results_df.head(10).to_string())
    list_of_models = []
    list_of_models.append([x for x in evaluation_results_df[investigate_by]])
    list_of_slugs = []
    if save_results_for_every_item:
        evaluation_results_df['query_slug'].dropna(inplace=True)
        list_of_slugs.append([x for x in evaluation_results_df['query_slug']])

    list_of_created_at = []
    if crop_by_date:
        list_of_created_at.append([x for x in evaluation_results_df['created_at']])
    print(list_of_models)
    print("evaluation_results_df['results_part_2']")
    print(evaluation_results_df['results_part_2'])

    list_of_aps = []
    evaluation_results_df['results_part_2'].dropna(inplace=True)

    for index, row in evaluation_results_df.iterrows():
        print("type(row['results_part_2']:")
        print(type(row['results_part_2']))
        results_df = row['results_part_2']
        relevance_dict = dict((k, results_df[k]) for k in ['relevance', 'coefficient']
                              if k in results_df)
        f = len(relevance_dict[next(iter(relevance_dict))])
        if all(len(x) == f for x in relevance_dict.values()):
            print(relevance_dict['relevance'])
            print(relevance_dict['coefficient'])
            try:
                list_of_aps.append(average_precision_score(relevance_dict['relevance'],
                                                           relevance_dict['coefficient'], average='weighted'))
            except TypeError as e:
                print("TypeError:")
                print(e)
                print("Skipping record. Does not have the same number of column")
                continue
        else:
            print("Skipping record. Does not have the same number of column")

    print("AVERAGE PRECISION:")
    print(list_of_aps)

    list_of_ps = []
    list_of_ps.append([precision_score(x['relevance'], np.full((1, len(x['relevance'])), 1)[0], average='macro')
                       for x in evaluation_results_df['results_part_2']
                       if not None in x['relevance']]
                      )
    print("WEIGHTED PRECISION SCORE:")
    print(list_of_ps)

    list_of_rs = []
    list_of_rs.append([recall_score(x['relevance'], np.full((1, len(x['relevance'])), 1)[0], average='macro')
                       for x in evaluation_results_df['results_part_2']
                       if not None in x['relevance']]
                      )
    print("WEIGHTED RECALL SCORE:")
    print(list_of_rs)

    print("BALANCED_ACCURACY:")
    list_of_balanced_accuracies = []
    list_of_balanced_accuracies.append([balanced_accuracy_score(x['relevance'],
                                                                np.full((1, len(x['relevance'])), 1)[0])
                                        for x in evaluation_results_df['results_part_2']
                                        if not None in x['relevance']
                                        ])
    print(list_of_balanced_accuracies)
    """
    true_relevance = y_true
    scores = y_scores
    """
    print("DCG:")
    list_of_dcgs = []
    list_of_dcgs.append([dcg_score([x['relevance']], [np.full((1, len(x['relevance'])), 1)[0]])
                         for x in evaluation_results_df['results_part_2']
                         if not None in x['relevance']
                        ])
    print(list_of_dcgs)

    print("DCG AT K=5:")
    list_of_dcg_at_k = []
    list_of_dcg_at_k.append([dcg_score([x['relevance']], [np.full((1, len(x['relevance'])), 1)[0]], k=k)
                             for x in evaluation_results_df['results_part_2']
                             if not None in x['relevance']
                             ])
    print(list_of_dcg_at_k)

    print("F1-SCORE (WEIGHTED AVERAGE):")
    list_of_f1_score = []
    list_of_f1_score.append([f1_score(x['relevance'], np.full((1, len(x['relevance'])), 1)[0], average='weighted')
                             for x in evaluation_results_df['results_part_2']
                             if not None in x['relevance']
                             ])
    print(list_of_f1_score)

    print("NDCG:")
    list_of_ndcgs = []
    list_of_ndcgs.append([ndcg_score([x['relevance']], [np.full((1, len(x['relevance'])), 1)[0]])
                          for x in evaluation_results_df['results_part_2']
                          if not None in x['relevance']
                          ])
    print(list_of_ndcgs)

    print("NDCG AT 5:")
    list_of_ndcgs_at_k = []
    list_of_ndcgs_at_k.append([ndcg_score([x['relevance']], [np.full((1, len(x['relevance'])), 1)[0]], k=k)
                               for x in evaluation_results_df['results_part_2']
                               if not None in x['relevance']
                               ])
    print(list_of_ndcgs_at_k)

    print("PRECISION, RECALL, THRESHOLDS:")
    list_of_precisions = []
    list_of_recalls = []
    list_of_thresholds = []
    list_of_precisions, list_of_recalls, list_of_thresholds = zip(
        *(precision_recall_curve(x['relevance'], x['coefficient'])
          for x in evaluation_results_df['results_part_2']
          if (len(x['relevance']) == len(x['coefficient'])
              and not (None in x['coefficient'] or None in x['relevance']))
          )
    )

    precision = list_of_precisions
    recall = list_of_recalls
    threshold = list_of_thresholds

    print(list_of_precisions)
    print(list_of_recalls)
    print(list_of_thresholds)

    list_of_precisions = []
    print("PRECISION SCORE:")
    list_of_precisions.append([precision_score(x['relevance'], np.full((1, len(x['relevance'])), 1)[0], average='macro')
                               for x in evaluation_results_df['results_part_2']
                               if not None in x['relevance']
                              ])
    print(list_of_precisions)


    list_of_recalls = []
    print("RECALL SCORE:")
    list_of_recalls.append([recall_score(x['relevance'], np.full((1, len(x['relevance'])), 1)[0], average='macro')
                               for x in evaluation_results_df['results_part_2']
                               if not None in x['relevance']
                              ])
    print(list_of_recalls)
    """
    list_of_top_k_accuracy = []
    print("TOP k=5 ACCURACY:")
    list_of_top_k_accuracy.append([top_k_accuracy_score(x['relevance'], x['coefficient'], k=2)
                               if len(x['relevance']) == len(x['coefficient'])
                               else ValueError(
        "Lengths of arrays in relevance and coefficient does not match.")
                               for x in evaluation_results_df['results_part_2']])
    print(list_of_top_k_accuracy)
    """

    # completation of results
    # TODO: Add other columns
    if save_results_for_every_item is True:
        dict_of_model_stats = {'AP': list_of_aps, 'precision_score': list_of_ps[0],
                               'recall_score': list_of_rs[0],
                               'balanced_accuracies': list_of_balanced_accuracies[0],
                               'DCG': list_of_dcgs[0], 'DCG_AT_' + str(k): list_of_dcg_at_k[0],
                               'F1-SCORE': list_of_f1_score[0], 'NDCG': list_of_ndcgs[0],
                               'NDCG_AT_' + str(k): list_of_ndcgs_at_k[0],
                               investigate_by: list_of_models[0], 'slug': list_of_slugs[0],
                               'created_at': list_of_created_at[0],
                               'user_id': evaluation_results_df['user_id']}
    else:
        dict_of_model_stats = {'AP': list_of_aps, 'precision_score': list_of_ps[0],
                               'recall_score': list_of_rs[0],
                               'balanced_accuracies': list_of_balanced_accuracies[0],
                               'DCG': list_of_dcgs[0], 'DCG_AT_' + str(k): list_of_dcg_at_k[0],
                               'F1-SCORE': list_of_f1_score[0], 'NDCG': list_of_ndcgs[0],
                               'NDCG_AT_' + str(k): list_of_ndcgs_at_k[0],
                               investigate_by: list_of_models[0]}

    print("dict_of_model_stats:")
    print(dict_of_model_stats)

    model_results = pd.DataFrame.from_dict(dict_of_model_stats, orient='index').transpose()
    model_results = model_results.fillna(0)

    print("Model results:")
    print(model_results.to_string())

    if save_results_for_every_item:
        hash = random.getrandbits(128)
        path_for_saving = "research/relevance/playground_relevance/by_model/model_variant_relevances_every_item_" + \
                          str(hash) + ".csv"
        model_results.to_csv(path_for_saving)
        print("Results saved.")

        model_results = model_results.groupby(['user_id']).mean()
        path_for_saving = "research/relevance/playground_relevance/by_model/model_variant_relevances_group_by_user" + \
                          str(hash) + ".csv"
        model_results.to_csv(path_for_saving)
        print("Results saved.")

        # joining with rvaluated posts slugs
        if save_results_for_model is True:
            print("Saving also results for every item.")
            path_to_resuts = 'save_relevance_results_by_queries.csv'
            recommender_methods = RecommenderMethods()
            posts_categories__ratings_df = recommender_methods.get_posts_categories_dataframe()
            print("model_results.columns")
            print(posts_categories__ratings_df.columns)
            categories_df = posts_categories__ratings_df[['user_id', 'category_title', 'slug']]
            model_results = pd.merge(model_results, categories_df, left_on='slug', right_on='slug', how='left')
            # model_results_joined_with_category.to_csv(path_to_resuts, index=False)
            print("model_results.columns")
            print(model_results.columns)
            print(model_results.to_string())
            grouped_results = model_results.groupby(['AP', 'model_variant', 'slug', 'created_at',
                                                     'category_title']).mean().reset_index()
            # full_results = pd.merge(grouped_results, categories_df, left_on='slug', right_on='post_slug', how='left')

            print("full_results.columns")
            print(grouped_results.columns)
            grouped_results = grouped_results.sort_values(by=['created_at', 'category_title'])
            transposed_results = grouped_results.pivot_table('AP', ['slug', 'category_title'], 'model_variant')
            return transposed_results

        return model_results.groupby([investigate_by]).mean().reset_index()
    else:
        return model_results.reset_index()



def plot_confusion_matrix(cm, title):
    plt.tight_layout()

    ax = sns.heatmap(cm, annot=True, cmap='Blues')

    ax.set_title(title)

    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.tight_layout()

    ## Display the visualization of the Confusion Matrix.
    plt.show()


def show_confusion_matrix():
    print("Please be awware that confusion matrix is only ")
    evaluation_results_df = evaluation_results.get_playground_evaluations()
    print(evaluation_results_df.head(10).to_string())
    dict_of_model_stats = {}
    list_of_models = []
    list_of_models.append([x for x in evaluation_results_df['model_variant']])
    print(list_of_models)

    y_pred = np.full((1, 20), 1)[0]

    list_of_models.append([x for x in evaluation_results_df['model_name']])
    print(list_of_models)

    print("AVERAGE PRECISION:")
    list_of_confusion_matrices = []
    list_of_confusion_matrices.append([confusion_matrix(x['relevance'], y_pred)
                                       if len(x['relevance']) == len(x['coefficient'])
                                       else ValueError("Lengths of arrays in relevance and coefficient does not match.")
                                       for x in evaluation_results_df['results_part_2']])

    print("CONFUSION MATRIX:")

    print(list_of_confusion_matrices)
    # for cm in list_of_confusion_matrices:
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    cm = list_of_confusion_matrices[0][0]
    list_of_confusion_matrices_selected = []
    for row in list_of_confusion_matrices:
        for item in row:
            print("item")
            print(type(item))
            if item.shape == (2, 2):
                item = np.asmatrix(item)
                print("item after convrsion to matrix")
                print(item)
                list_of_confusion_matrices_selected.append(item)

    print("list_of_confusion_matrices_selected:")
    print(list_of_confusion_matrices_selected)
    cm_mean = np.mean(list_of_confusion_matrices_selected, axis=0)
    print(cm_mean)
    plt.figure()
    plot_confusion_matrix(cm_mean, "Confusion matrix")


def print_model_variant_relevances():
    stats = models_complete_statistics(investigate_by='model_variant', save_results_for_every_item=False)
    print("Means of model's metrics:")
    print(stats.to_string())


def save_model_variant_relevances(crop_by_date=False, last_n_by_date=None):
    stats = models_complete_statistics(investigate_by='model_variant', save_results_for_every_item=True,
                                       crop_by_date=crop_by_date, last_n_by_date=last_n_by_date)
    print("Means of model's metrics:")
    print(stats.to_string())
    print("Saving CSV with user evaluation results...")
    stats = stats.round(2)
    hash = random.getrandbits(128)
    path_for_saving = "research/relevance/playground_relevance/by_model/model_variant_relevances_" + str(hash) + ".csv"
    stats.to_csv(path_for_saving)
    print("Results saved.")


def print_model_variant_relevances_for_each_article(save_to_csv=False, crop_by_date=False):
    stats = models_complete_statistics(investigate_by='model_variant', save_results_for_every_item=True,
                                       crop_by_date=crop_by_date)
    print("Means of model's metrics:")
    print(stats.to_string())
    if save_to_csv is True:
        stats = stats.round(2)
        path_for_saving = "research/word2vec/evaluation/cswiki/word2vec_tuning_relevance_results_by_each_article.csv"
        stats.to_csv(path_for_saving)
        print("Results saved.")


def print_overall_model_relevances():
    stats = models_complete_statistics(investigate_by='model_name', save_results_for_every_item=True)
    print("Means of model's metrics:")
    print(stats.to_string())


def print_confusion_matrix():
    print(show_confusion_matrix())
