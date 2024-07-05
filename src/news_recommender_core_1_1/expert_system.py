import logging
from functools import lru_cache

import matplotlib.pyplot as plt
from simpful import *

FS_interaction_strength = FuzzySystem()
FS_model_strength = FuzzySystem()
FS_recommendation_strength_hybrid = FuzzySystem()


@lru_cache(maxsize=65536)
def get_interaction_strength(interaction_type, belief_in_interaction_strength, number_of_interactions,
                             max_number_of_interactions):
    logging.debug("interaction_type: {}".format(interaction_type))
    logging.debug("belief_in_interaction_strength: {}".format(belief_in_interaction_strength))
    logging.debug("number_of_interactions: {}".format(number_of_interactions))

    if interaction_type == 0 or belief_in_interaction_strength == 0 or number_of_interactions == 0:
        return 0

    TLV = AutoTriangle(3,
                       terms=['small', 'medium', 'big'],
                       universe_of_discourse=[0, 10])
    FS_interaction_strength.add_linguistic_variable("belief_in_{}_interaction_strength"
                                                    .format(interaction_type), TLV)

    # TODO: Replace the constant of the U with the number of the articles in the dataset
    TLV_2 = AutoTriangle(3,
                         terms=['small', 'medium', 'big'],
                         universe_of_discourse=[0, max_number_of_interactions])
    FS_interaction_strength.add_linguistic_variable("number_of_interactions", TLV_2)

    output_low_upper_bound = 1.0
    output_medium_lower_bound = 0.5
    output_medium_center = output_low_upper_bound
    output_medium_upper_bound = 1.5
    output_high_upper_bound = 2

    # For belief variables
    S1 = TrapezoidFuzzySet(0, 0, 2, 4, term="small")
    S2 = TrapezoidFuzzySet(2, 4, 6, 8, term="medium")
    S3 = TrapezoidFuzzySet(6, 8, 10, 10, term="big")

    belief_var = LinguisticVariable([S1, S2, S3], universe_of_discourse=[0, 10])
    FS_recommendation_strength_hybrid.add_linguistic_variable("belief_in_cf_model", belief_var)
    FS_recommendation_strength_hybrid.add_linguistic_variable("belief_in_cb_model", belief_var)

    # For recommendation coefficient variables
    R1 = TrapezoidFuzzySet(0, 0, 0.3, 0.5, term="small")
    R2 = TrapezoidFuzzySet(0.3, 0.5, 0.7, 0.9, term="medium")
    R3 = TrapezoidFuzzySet(0.7, 0.9, 1, 1, term="big")

    rec_coef_var = LinguisticVariable([R1, R2, R3], universe_of_discourse=[0, 1])
    FS_recommendation_strength_hybrid.add_linguistic_variable("recommendation_coefficient_cf", rec_coef_var)
    FS_recommendation_strength_hybrid.add_linguistic_variable("recommendation_coefficient_cb", rec_coef_var)

    # For output variable (model strength)
    O1 = TrapezoidFuzzySet(0, 0, 0.4, 0.8, term="small")
    O2 = TrapezoidFuzzySet(0.4, 0.8, 1.2, 1.6, term="medium")
    O3 = TrapezoidFuzzySet(1.2, 1.6, 2, 2, term="big")

    O = LinguisticVariable([O1, O2, O3], universe_of_discourse=[0, 2])
    FS_recommendation_strength_hybrid.add_linguistic_variable("model_strength", O)

    FS_interaction_strength.add_rules([
        "IF (belief_in_{}_interaction_strength IS small) THEN (recommendation_strength IS small)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS medium) AND (number_of_interactions IS medium) THEN (recommendation_strength IS medium)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS medium) AND (number_of_interactions IS small) THEN (recommendation_strength IS medium)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS medium) AND (number_of_interactions IS big) THEN (recommendation_strength IS medium)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS big) AND (number_of_interactions IS small) THEN (recommendation_strength IS medium)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS big) AND (number_of_interactions IS medium) THEN (recommendation_strength IS big)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS big) AND (number_of_interactions IS big) THEN (recommendation_strength IS big)".format(
            interaction_type),
    ])

    FS_interaction_strength.set_variable("belief_in_{}_interaction_strength".format(interaction_type),
                                         belief_in_interaction_strength)
    FS_interaction_strength.set_variable("number_of_interactions", number_of_interactions)

    recommendation_strength = FS_interaction_strength.inference()

    # plot_fuzzy(TLV, TLV_2, O)
    # WARNING: We iterate through the all the records in the dataset,
    # so we need to make sure to uncomment this only when necessary

    return recommendation_strength['recommendation_strength']


def plot_fuzzy(TLV, TLV_2, O):
    fig, ax = plt.subplots(2, 2)
    TLV.draw(ax=ax[0][0])
    TLV_2.draw(ax=ax[0][1])
    O.draw(ax=ax[1][0])

    plt.tight_layout()
    plt.show()


@lru_cache(maxsize=65536)
def get_model_strength(model_type, belief_in_model, recommendation_coefficient):
    logging.debug("model_strength: {}".format(model_type))
    logging.debug("belief_in_model: {}".format(belief_in_model))
    logging.debug("recommendation_coefficient: {}".format(recommendation_coefficient))

    TLV = AutoTriangle(3,
                       terms=['small', 'medium', 'big'],
                       universe_of_discourse=[0, 10])
    FS_model_strength.add_linguistic_variable("belief_in_{}_model"
                                              .format(model_type), TLV)

    output_low_upper_bound = 0.75
    output_medium_lower_bound = 0.6
    output_medium_center = output_low_upper_bound
    output_medium_upper_bound = 0.9
    output_high_upper_bound = 1.0
    # TODO: Replace the constant of the U with the number of the articles in the dataset
    I21 = TriangleFuzzySet(0, 0, output_low_upper_bound, term="small")
    I22 = TriangleFuzzySet(output_medium_lower_bound, output_medium_center,
                           output_medium_upper_bound, term="medium")
    I23 = TriangleFuzzySet(output_medium_center, output_high_upper_bound,
                           output_high_upper_bound, term="big")

    TLV_2 = LinguisticVariable([I21, I22, I23], universe_of_discourse=[0.0, 1.0])
    FS_model_strength.add_linguistic_variable("recommendation_coefficient", TLV_2)

    output_low_upper_bound = 1.0
    output_medium_lower_bound = 0.5
    output_medium_center = output_low_upper_bound
    output_medium_upper_bound = 1.5
    output_high_upper_bound = 2.0

    O1 = TriangleFuzzySet(0, 0, output_low_upper_bound, term="small")
    O2 = TriangleFuzzySet(output_medium_lower_bound, output_low_upper_bound,
                          output_medium_upper_bound, term="medium")
    O3 = TriangleFuzzySet(output_medium_center, output_high_upper_bound,
                          output_high_upper_bound, term="big")

    O = LinguisticVariable([O1, O2, O3], universe_of_discourse=[0, output_high_upper_bound])
    FS_model_strength.add_linguistic_variable("model_strength", O)

    """
    FS_model_strength.add_rules([
        "IF (belief_in_{}_model IS small) AND (recommendation_coefficient IS small) THEN (model_strength IS small)".format(
            model_type),
        "IF (belief_in_{}_model IS small) AND (recommendation_coefficient IS medium) THEN (model_strength IS small)".format(
            model_type),
        "IF (belief_in_{}_model IS small) AND (recommendation_coefficient IS big) THEN (model_strength IS medium)".format(
            model_type),
        "IF (belief_in_{}_model IS medium) AND (recommendation_coefficient IS small) THEN (model_strength IS small)".format(
            model_type),
        "IF (belief_in_{}_model IS medium) AND (recommendation_coefficient IS medium) THEN (model_strength IS medium)".format(
            model_type),
        "IF (belief_in_{}_model IS medium) AND (recommendation_coefficient IS big) THEN (model_strength IS big)".format(
            model_type),
        "IF (belief_in_{}_model IS big) AND (recommendation_coefficient IS small) THEN (model_strength IS small)".format(
            model_type),
        "IF (belief_in_{}_model IS big) AND (recommendation_coefficient IS medium) THEN (model_strength IS medium)".format(
            model_type),
        "IF (belief_in_{}_model IS big) AND (recommendation_coefficient IS big) THEN (model_strength IS big)".format(
            model_type),
    ])
    """

    FS_model_strength.add_rules([
        "IF (belief_in_{}_model IS small) THEN (model_strength IS small)".format(
            model_type),
        "IF (belief_in_{}_model IS medium) AND (recommendation_coefficient IS medium) THEN (model_strength IS medium)".format(
            model_type),
        "IF (belief_in_{}_model IS medium) AND (recommendation_coefficient IS small) THEN (model_strength IS medium)".format(
            model_type),
        "IF (belief_in_{}_model IS medium) AND (recommendation_coefficient IS big) THEN (model_strength IS medium)".format(
            model_type),
        "IF (belief_in_{}_model IS big) AND (recommendation_coefficient IS small) THEN (model_strength IS medium)".format(
            model_type),
        "IF (belief_in_{}_model IS big) AND (recommendation_coefficient IS medium) THEN (model_strength IS big)".format(
            model_type),
        "IF (belief_in_{}_model IS big) AND (recommendation_coefficient IS big) THEN (model_strength IS big)".format(
            model_type),
    ])

    FS_model_strength.set_variable("belief_in_{}_model".format(model_type), belief_in_model)
    FS_model_strength.set_variable("recommendation_coefficient", recommendation_coefficient)

    recommendation_fuzzy_hybrid = FS_model_strength.inference()

    # plot_fuzzy(TLV, TLV_2, O)

    logging.debug("fuzzy ensemble coefficient:")
    logging.debug(recommendation_fuzzy_hybrid)

    return recommendation_fuzzy_hybrid['model_strength']


@lru_cache(maxsize=131072)
def get_recommendation_strength_hybrid(belief_in_model_cf, belief_in_model_cb,
                                       recommendation_coefficient_cf,
                                       recommendation_coefficient_cb):
    logging.debug("belief_in_model_cf: {}".format(belief_in_model_cf))
    logging.debug("belief_in_model_cb: {}".format(belief_in_model_cb))
    logging.debug("recommendation_coefficient_cf: {}".format(recommendation_coefficient_cf))
    logging.debug("recommendation_coefficient_cb: {}".format(recommendation_coefficient_cb))

    TLV = AutoTriangle(3,
                       terms=['small', 'medium', 'big'],
                       universe_of_discourse=[0, 10])
    FS_recommendation_strength_hybrid.add_linguistic_variable("belief_in_cf_model", TLV)

    TLV = AutoTriangle(3,
                       terms=['small', 'medium', 'big'],
                       universe_of_discourse=[0, 10])
    FS_recommendation_strength_hybrid.add_linguistic_variable("belief_in_cb_model", TLV)

    output_low_upper_bound = 0.75
    output_medium_lower_bound = 0.6
    output_medium_center = output_low_upper_bound
    output_medium_upper_bound = 0.9
    output_high_upper_bound = 1.0
    # TODO: Replace the constant of the U with the number of the articles in the dataset
    I21 = TriangleFuzzySet(0, 0, output_low_upper_bound, term="small")
    I22 = TriangleFuzzySet(output_medium_lower_bound, output_medium_center,
                           output_medium_upper_bound, term="medium")
    I23 = TriangleFuzzySet(output_medium_center, output_high_upper_bound,
                           output_high_upper_bound, term="big")

    TLV_2 = LinguisticVariable([I21, I22, I23], universe_of_discourse=[0.0, 1.0])
    FS_recommendation_strength_hybrid.add_linguistic_variable("recommendation_coefficient_cf", TLV_2)

    TLV_2 = LinguisticVariable([I21, I22, I23], universe_of_discourse=[0.0, 1.0])
    FS_recommendation_strength_hybrid.add_linguistic_variable("recommendation_coefficient_cb", TLV_2)

    output_low_upper_bound = 1.0
    output_medium_lower_bound = 0.5
    output_medium_center = output_low_upper_bound
    output_medium_upper_bound = 1.5
    output_high_upper_bound = 2.0

    O1 = TriangleFuzzySet(0, 0, output_low_upper_bound, term="small")
    O2 = TriangleFuzzySet(output_medium_lower_bound, output_low_upper_bound,
                          output_medium_upper_bound, term="medium")
    O3 = TriangleFuzzySet(output_medium_center, output_high_upper_bound,
                          output_high_upper_bound, term="big")

    O = LinguisticVariable([O1, O2, O3], universe_of_discourse=[0, output_high_upper_bound])
    FS_recommendation_strength_hybrid.add_linguistic_variable("model_strength", O)

    FS_recommendation_strength_hybrid.add_rules([
        "IF (belief_in_cf_model IS big) AND (recommendation_coefficient_cf IS big) THEN (model_strength IS big)",
        "IF (belief_in_cb_model IS big) AND (recommendation_coefficient_cb IS big) THEN (model_strength IS big)",
        "IF (belief_in_cf_model IS small) AND (recommendation_coefficient_cf IS small) THEN (model_strength IS small)",
        "IF (belief_in_cb_model IS small) AND (recommendation_coefficient_cb IS small) THEN (model_strength IS small)",
        "IF (belief_in_cf_model IS medium) AND (recommendation_coefficient_cf IS medium) THEN (model_strength IS medium)",
        "IF (belief_in_cb_model IS medium) AND (recommendation_coefficient_cb IS medium) THEN (model_strength IS medium)",
        "IF (belief_in_cf_model IS big) AND (recommendation_coefficient_cf IS small) THEN (model_strength IS medium)",
        "IF (belief_in_cb_model IS big) AND (recommendation_coefficient_cb IS small) THEN (model_strength IS medium)",
        "IF (belief_in_cf_model IS small) AND (recommendation_coefficient_cf IS big) THEN (model_strength IS medium)",
        "IF (belief_in_cb_model IS small) AND (recommendation_coefficient_cb IS big) THEN (model_strength IS medium)",
        "IF (belief_in_cf_model IS big) AND (belief_in_cb_model IS big) AND (recommendation_coefficient_cf IS big) AND (recommendation_coefficient_cb IS big) THEN (model_strength IS big)",
        "IF (belief_in_cf_model IS small) AND (belief_in_cb_model IS small) AND (recommendation_coefficient_cf IS small) AND (recommendation_coefficient_cb IS small) THEN (model_strength IS small)",
        "IF (belief_in_cf_model IS medium) OR (belief_in_cb_model IS medium) OR (recommendation_coefficient_cf IS medium) OR (recommendation_coefficient_cb IS medium) THEN (model_strength IS medium)",
        "IF (belief_in_cf_model IS big) AND (recommendation_coefficient_cf IS big) AND (belief_in_cb_model IS small) THEN (model_strength IS big)",
        "IF (belief_in_cb_model IS big) AND (recommendation_coefficient_cb IS big) AND (belief_in_cf_model IS small) THEN (model_strength IS big)",
        "IF (belief_in_cf_model IS medium) AND (recommendation_coefficient_cf IS big) AND (belief_in_cb_model IS medium) AND (recommendation_coefficient_cb IS medium) THEN (model_strength IS big)",
    ])

    FS_recommendation_strength_hybrid.set_variable("belief_in_cf_model", belief_in_model_cf)
    FS_recommendation_strength_hybrid.set_variable("belief_in_cb_model", belief_in_model_cb)

    FS_recommendation_strength_hybrid.set_variable("recommendation_coefficient_cf", recommendation_coefficient_cf)
    FS_recommendation_strength_hybrid.set_variable("recommendation_coefficient_cb", recommendation_coefficient_cb)

    recommendation_fuzzy_hybrid = FS_recommendation_strength_hybrid.inference()

    logging.debug("fuzzy ensemble coefficient:")
    logging.debug(recommendation_fuzzy_hybrid)

    return recommendation_fuzzy_hybrid['model_strength']
