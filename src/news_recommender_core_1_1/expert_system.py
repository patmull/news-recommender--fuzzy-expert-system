import logging

from simpful import *

FS = FuzzySystem()


def get_interaction_strength(interaction_type, belief_in_interaction_strength, number_of_interactions):

    logging.debug("interaction_type: {}".format(interaction_type))
    logging.debug("belief_in_interaction_strength: {}".format(belief_in_interaction_strength))
    logging.debug("number_of_interactions: {}".format(number_of_interactions))

    TLV = AutoTriangle(3,
                       terms=['small', 'middle', 'great'],
                       universe_of_discourse=[0, 10])
    FS.add_linguistic_variable("belief_in_{}_interaction_strength"
                               .format(interaction_type), TLV)
    # TODO: Replace the constant of the U with the number of the articles in the dataset
    TLV_2 = AutoTriangle(3,
                         terms=['small', 'middle', 'great'],
                         universe_of_discourse=[0, 100])
    FS.add_linguistic_variable("number_of_interactions", TLV_2)

    O1 = TriangleFuzzySet(0, 0, 25, term="low")
    O2 = TriangleFuzzySet(0, 25, 100, term="medium")
    O3 = TriangleFuzzySet(25, 100, 100, term="high")
    FS.add_linguistic_variable("interaction_strength", LinguisticVariable([O1, O2, O3], universe_of_discourse=[0, 100]))

    FS.add_rules([
        "IF (belief_in_{}_interaction_strength IS small) OR (number_of_interactions IS small) THEN (interaction_strength IS low)"
        .format(interaction_type),
        "IF (number_of_interactions IS middle) THEN (interaction_strength IS medium)",
        "IF (belief_in_{}_interaction_strength IS high) THEN (interaction_strength IS high)".format(interaction_type),
        "IF (belief_in_{}_interaction_strength IS medium) AND (number_of_interactions IS medium) THEN (interaction_strength IS medium)".format(interaction_type),
        "IF (belief_in_{}_interaction_strength IS low) AND (number_of_interactions IS medium) THEN (interaction_strength IS low)".format(
            interaction_type)
    ])

    FS.set_variable("belief_in_{}_interaction_strength".format(interaction_type), belief_in_interaction_strength)
    FS.set_variable("number_of_interactions", number_of_interactions)

    interaction_strength = FS.inference()

    return interaction_strength
