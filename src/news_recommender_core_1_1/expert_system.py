from simpful import *

FS = FuzzySystem()


def get_interaction_strength():
    TLV = AutoTriangle(3, terms=['small', 'middle', 'great'], universe_of_discourse=[0, 10])
    FS.add_linguistic_variable("belief_in_views_interaction_strength", TLV)
    # TODO: Replace the constant of the U with the number of the articles in the dataset
    TLV_2 = AutoTriangle(3, terms=['small', 'middle', 'great'], universe_of_discourse=[0, 100])
    FS.add_linguistic_variable("number_of_interactions", TLV_2)

    O1 = TriangleFuzzySet(0, 0, 25, term="low")
    O2 = TriangleFuzzySet(0, 25, 100, term="medium")
    O3 = TriangleFuzzySet(25, 100, 100, term="high")
    FS.add_linguistic_variable("interaction_strength", LinguisticVariable([O1, O2, O3], universe_of_discourse=[0, 100]))

    FS.add_rules([
        "IF (belief_in_views_interaction_strength IS small) OR (number_of_interactions IS small) THEN (interaction_strength IS low)",
        "IF (number_of_interactions IS middle) THEN (interaction_strength IS medium)",
        "IF (belief_in_views_interaction_strength IS middle) OR (number_of_interactions IS middle) THEN (interaction_strength IS high)"
    ])

    FS.set_variable("belief_in_views_interaction_strength", 8)
    FS.set_variable("number_of_interactions", 30)

    interaction_strength = FS.inference()

    return interaction_strength
