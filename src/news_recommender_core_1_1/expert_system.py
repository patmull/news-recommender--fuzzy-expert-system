import logging
from functools import lru_cache

import matplotlib.pyplot as plt
from simpful import *

FS = FuzzySystem()


def get_interaction_strength(interaction_type, belief_in_interaction_strength, number_of_interactions):

    logging.debug("interaction_type: {}".format(interaction_type))
    logging.debug("belief_in_interaction_strength: {}".format(belief_in_interaction_strength))
    logging.debug("number_of_interactions: {}".format(number_of_interactions))

    TLV = AutoTriangle(3,
                       terms=['low', 'medium', 'high'],
                       universe_of_discourse=[0, 10])
    FS.add_linguistic_variable("belief_in_{}_interaction_strength"
                               .format(interaction_type), TLV)
    # TODO: Replace the constant of the U with the number of the articles in the dataset
    TLV_2 = AutoTriangle(3,
                         terms=['low', 'medium', 'high'],
                         universe_of_discourse=[0, 100])
    FS.add_linguistic_variable("number_of_interactions", TLV_2)

    interaction_strength_low_upper_bound = 0.5
    interaction_strength_medium_upper_bound = 1.25
    interaction_strength_high_upper_bound = 2

    O1 = TriangleFuzzySet(0, 0, interaction_strength_low_upper_bound, term="low")
    O2 = TriangleFuzzySet(0, interaction_strength_low_upper_bound, interaction_strength_medium_upper_bound,
                          term="medium")
    O3 = TriangleFuzzySet(interaction_strength_low_upper_bound, interaction_strength_high_upper_bound,
                          interaction_strength_high_upper_bound, term="high")
    O = LinguisticVariable([O1, O2, O3], universe_of_discourse=[0, interaction_strength_high_upper_bound])
    FS.add_linguistic_variable("interaction_strength", O)

    FS.add_rules([
        "IF (belief_in_{}_interaction_strength IS low) OR (number_of_interactions IS low) THEN (interaction_strength IS low)".format(interaction_type),
        "IF (number_of_interactions IS medium) THEN (interaction_strength IS medium)",
        "IF (belief_in_{}_interaction_strength IS high) THEN (interaction_strength IS high)".format(interaction_type),
        "IF (belief_in_{}_interaction_strength IS medium) AND (number_of_interactions IS medium) THEN (interaction_strength IS medium)".format(interaction_type),
        "IF (belief_in_{}_interaction_strength IS low) AND (number_of_interactions IS medium) THEN (interaction_strength IS low)".format(interaction_type),
        "IF (belief_in_{}_interaction_strength IS medium) AND (number_of_interactions IS high) THEN (interaction_strength IS high)".format(interaction_type),
        "IF (belief_in_{}_interaction_strength IS low) AND (number_of_interactions IS low) THEN (interaction_strength IS low)".format(interaction_type)
    ])

    FS.set_variable("belief_in_{}_interaction_strength".format(interaction_type), belief_in_interaction_strength)
    FS.set_variable("number_of_interactions", number_of_interactions)

    interaction_strength = FS.inference()

    # plot_fuzzy(TLV, TLV_2, O)
    # WARNING: We iterate through the all the records in the dataset,
    # so we need to make sure to uncomment this only when necessary

    return interaction_strength['interaction_strength']


def plot_fuzzy(TLV, TLV_2, O):
    fig, ax = plt.subplots(2, 2)
    TLV.draw(ax=ax[0][0])
    TLV_2.draw(ax=ax[0][1])
    O.draw(ax=ax[1][0])

    plt.tight_layout()
    plt.show()


@lru_cache(maxsize=1024)
def get_model_strength(model_type, belief_in_model, recommendation_coefficient):

    logging.debug("model_type: {}".format(model_type))
    logging.debug("belief_in_model: {}".format(belief_in_model))
    logging.debug("recommendation_coefficient: {}".format(recommendation_coefficient))

    TLV = AutoTriangle(3,
                       terms=['low', 'medium', 'high'],
                       universe_of_discourse=[0, 10])
    FS.add_linguistic_variable("belief_in_{}_model"
                               .format(model_type), TLV)
    # TODO: Replace the constant of the U with the number of the articles in the dataset
    TLV_2 = AutoTriangle(3, terms=['low', 'medium', 'high'], universe_of_discourse=[0, 1.0])
    FS.add_linguistic_variable("recommendation_coefficient", TLV_2)

    interaction_strength_low_upper_bound = 0.5
    interaction_strength_medium_upper_bound = 1.25
    interaction_strength_high_upper_bound = 2

    O1 = TriangleFuzzySet(0, 0, interaction_strength_low_upper_bound, term="low")
    O2 = TriangleFuzzySet(0, interaction_strength_low_upper_bound, interaction_strength_medium_upper_bound,
                          term="medium")
    O3 = TriangleFuzzySet(interaction_strength_low_upper_bound, interaction_strength_high_upper_bound,
                          interaction_strength_high_upper_bound, term="high")
    O = LinguisticVariable([O1, O2, O3], universe_of_discourse=[0, interaction_strength_high_upper_bound])
    FS.add_linguistic_variable("interaction_strength", O)

    FS.add_rules([
        "IF (belief_in_{}_model IS low) OR (recommendation_coefficient IS low) THEN (interaction_strength IS low)"
        .format(model_type),
        "IF (recommendation_coefficient IS medium) THEN (interaction_strength IS medium)",
        "IF (belief_in_{}_model IS high) THEN (interaction_strength IS high)"
        .format(model_type),
        "IF (belief_in_{}_model IS medium) AND (recommendation_coefficient IS medium) THEN (interaction_strength IS medium)"
        .format(model_type),
        "IF (belief_in_{}_model IS low) AND (recommendation_coefficient IS medium) THEN (interaction_strength IS low)"
        .format(model_type),
        "IF (belief_in_{}_model IS medium) AND (recommendation_coefficient IS high) THEN (interaction_strength IS high)"
        .format(model_type),
        "IF (belief_in_{}_model IS low) AND (recommendation_coefficient IS low) THEN (interaction_strength IS low)"
        .format(model_type)
    ])

    FS.set_variable("belief_in_{}_model".format(model_type), belief_in_model)
    FS.set_variable("recommendation_coefficient", recommendation_coefficient)

    recommendation_fuzzy_hybrid = FS.inference()

    logging.debug("recommendation_fuzzy_hybrid in fuzzy methods:")
    logging.debug(recommendation_fuzzy_hybrid)

    return recommendation_fuzzy_hybrid['interaction_strength']
