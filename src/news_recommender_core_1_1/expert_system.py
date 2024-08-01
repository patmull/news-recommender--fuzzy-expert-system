import logging
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
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
    logging.debug("max_number_of_interactions: {}".format(max_number_of_interactions))

    if (interaction_type == 0
            or belief_in_interaction_strength == 0
            or number_of_interactions == 0):
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

    """
    output_low_upper_bound = 1.0
    output_medium_lower_bound = 0.5
    output_medium_center = output_low_upper_bound
    output_medium_upper_bound = 1.5
    output_high_upper_bound = 2.0
    """
    output_low_upper_bound = 5.0
    output_medium_lower_bound = 2.5
    output_medium_center = output_low_upper_bound
    output_medium_upper_bound = 7.5
    output_high_upper_bound = 10.0

    ### Triangle ###
    O1 = TriangleFuzzySet(0, 0, output_low_upper_bound, term="small")
    O2 = TriangleFuzzySet(output_medium_lower_bound, output_low_upper_bound,
                          output_medium_upper_bound, term="medium")
    # todo: try with big coming from output_medium_upper_bound, low can be kept as that
    O3 = TriangleFuzzySet(output_medium_upper_bound, output_high_upper_bound,
                          output_high_upper_bound, term="big")
    O1_5 = TriangleFuzzySet(0.0, output_medium_lower_bound, output_medium_center,
                            term="medium-small")
    O2_5 = TriangleFuzzySet(output_medium_center, output_medium_upper_bound, output_high_upper_bound, term="medium-big")

    O = LinguisticVariable([O1, O1_5, O2, O2_5, O3], universe_of_discourse=[0, output_high_upper_bound])
    FS_interaction_strength.add_linguistic_variable("recommendation_strength", O)

    ### Trapezoid ###
    """
    O1 = TrapezoidFuzzySet(0, 0, output_low_upper_bound / 2, output_low_upper_bound, term="small")
    O1_5 = TrapezoidFuzzySet(output_low_upper_bound / 4, output_low_upper_bound / 2, output_low_upper_bound,
                             output_medium_center, term="medium-small")
    O2 = TrapezoidFuzzySet(output_medium_lower_bound, output_low_upper_bound, output_medium_center,
                           output_medium_upper_bound, term="medium")
    O2_5 = TrapezoidFuzzySet(output_medium_center, output_medium_upper_bound, output_high_upper_bound * 0.75,
                             output_high_upper_bound, term="medium-big")
    O3 = TrapezoidFuzzySet(output_medium_upper_bound, output_high_upper_bound * 0.75, output_high_upper_bound,
                           output_high_upper_bound, term="big")

    O = LinguisticVariable([O1, O1_5, O2, O2_5, O3], universe_of_discourse=[0, output_high_upper_bound])
    FS_interaction_strength.add_linguistic_variable("recommendation_strength", O)
    """

    ### Gaussian #1 ###
    """
    sigma = (output_high_upper_bound - 0) / 10  # Adjust this value to control the width of the Gaussian curves

    O1 = GaussianFuzzySet(0, sigma, term="small")
    O1_5 = GaussianFuzzySet(output_low_upper_bound / 2, sigma, term="medium-small")
    O2 = GaussianFuzzySet(output_medium_center, sigma, term="medium")
    O2_5 = GaussianFuzzySet((output_medium_upper_bound + output_high_upper_bound) / 2, sigma, term="medium-big")
    O3 = GaussianFuzzySet(output_high_upper_bound, sigma, term="big")

    O = LinguisticVariable([O1, O1_5, O2, O2_5, O3], universe_of_discourse=[0, output_high_upper_bound])
    FS_interaction_strength.add_linguistic_variable("recommendation_strength", O)
    """

    ### Gaussian #2: modified sigma ###
    """
    sigma = (output_high_upper_bound - 0) / 10  # Adjust this value to control the width of the Gaussian curves

    O1 = GaussianFuzzySet(0, sigma * 0.5, term="small")
    O1_5 = GaussianFuzzySet(output_low_upper_bound / 2, sigma * 0.75, term="medium-small")
    O2 = GaussianFuzzySet(output_medium_center, sigma, term="medium")
    O2_5 = GaussianFuzzySet((output_medium_upper_bound + output_high_upper_bound) / 2, sigma * 0.75, term="medium-big")
    O3 = GaussianFuzzySet(output_high_upper_bound, sigma * 0.5, term="big")

    O = LinguisticVariable([O1, O1_5, O2, O2_5, O3], universe_of_discourse=[0, output_high_upper_bound])
    FS_interaction_strength.add_linguistic_variable("recommendation_strength", O)
    """

    FS_interaction_strength.add_rules([
        "IF (belief_in_{}_interaction_strength IS small) AND (number_of_interactions IS small) THEN (recommendation_strength IS small)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS small) AND (number_of_interactions IS medium) THEN (recommendation_strength IS small)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS small) AND (number_of_interactions IS big) THEN (recommendation_strength IS medium-small)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS medium) AND (number_of_interactions IS small) THEN (recommendation_strength IS medium-small)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS medium) AND (number_of_interactions IS medium) THEN (recommendation_strength IS medium)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS medium) AND (number_of_interactions IS big) THEN (recommendation_strength IS medium-big)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS big) AND (number_of_interactions IS small) THEN (recommendation_strength IS medium)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS big) AND (number_of_interactions IS medium) THEN (recommendation_strength IS medium-big)".format(
            interaction_type),
        "IF (belief_in_{}_interaction_strength IS big) AND (number_of_interactions IS big) THEN (recommendation_strength IS big)".format(
            interaction_type),
    ])

    FS_interaction_strength.set_variable("belief_in_{}_interaction_strength".format(interaction_type),
                                         belief_in_interaction_strength)
    FS_interaction_strength.set_variable("number_of_interactions", number_of_interactions)

    recommendation_strength = FS_interaction_strength.inference()

    titles = ["Belief in Interaction Strength " + interaction_type, "Number of Interactions", "Recommendation Strength"]
    x_labels = ['Membership Value', 'Membership Value', 'Membership Value']
    y_labels = ['Degree of Belief', 'Degree of Belief', 'Interaction Strength']

    #plot_fuzzy(TLV, TLV_2, O, titles, x_labels, y_labels)
    # WARNING: We iterate through the all the records in the dataset,
    # so we need to make sure to uncomment this only when necessary

    return recommendation_strength['recommendation_strength']


def plot_fuzzy(TLV, TLV_2, O, titles, x_labels, y_labels):
    fig, ax = plt.subplots(2, 2)
    TLV.draw(ax=ax[0][0])
    ax[0][0].set_title(titles[0])
    ax[0][0].set_xlabel(x_labels[0])
    ax[0][0].set_ylabel(y_labels[0])

    TLV_2.draw(ax=ax[0][1])
    ax[0][1].set_title(titles[1])
    ax[0][1].set_xlabel(x_labels[1])
    ax[0][1].set_ylabel(y_labels[1])

    O.draw(ax=ax[1][0])
    ax[1][0].set_title(titles[2])
    ax[1][0].set_xlabel(x_labels[2])
    ax[1][0].set_ylabel(y_labels[2])

    plt.tight_layout()
    plt.show()


def plot_membership_functions(LV1, LV2, LV3, LV4, O):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    LV1.draw(ax=ax[0][0])
    LV2.draw(ax=ax[0][1])
    LV3.draw(ax=ax[1][0])
    LV4.draw(ax=ax[1][1])
    O.draw(ax=ax[1][2])
    plt.tight_layout()
    plt.show()


def plot_output_surface(fs, input_var1, input_var2, output_var):
    x = np.linspace(input_var1.get_universe_of_discourse()[0], input_var1.get_universe_of_discourse()[1], 50)
    y = np.linspace(input_var2.get_universe_of_discourse()[0], input_var2.get_universe_of_discourse()[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fs.set_variable(input_var1.name, X[i, j])
            fs.set_variable(input_var2.name, Y[i, j])
            Z[i, j] = fs.inference()[output_var.name]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel(input_var1.name)
    ax.set_ylabel(input_var2.name)
    ax.set_zlabel(output_var.name)
    ax.set_title(f"Output Surface: {input_var1.name} vs {input_var2.name} -> {output_var.name}")
    fig.colorbar(surf)
    plt.show()


@lru_cache(maxsize=131072)
def get_recommendation_strength_hybrid(belief_in_model_cf, belief_in_model_cb,
                                       recommendation_coefficient_cf,
                                       recommendation_coefficient_cb):
    logging.debug("belief_in_model_cf: {}".format(belief_in_model_cf))
    logging.debug("belief_in_model_cb: {}".format(belief_in_model_cb))
    logging.debug("recommendation_coefficient_cf: {}".format(recommendation_coefficient_cf))
    logging.debug("recommendation_coefficient_cb: {}".format(recommendation_coefficient_cb))

    ### Triangle ###
    """    
    TLV = AutoTriangle(5,
                       terms=['very_small', 'small', 'medium', 'big', 'very_big'],
                       universe_of_discourse=[0.0, 1.0])
    FS_recommendation_strength_hybrid.add_linguistic_variable("belief_in_cf_model", TLV)

    TLV = AutoTriangle(5,
                       terms=['very_small', 'small', 'medium', 'big', 'very_big'],
                       universe_of_discourse=[0.0, 1.0])
    FS_recommendation_strength_hybrid.add_linguistic_variable("belief_in_cb_model", TLV)


    TLV = AutoTriangle(5,
                       terms=['very_small', 'small', 'medium', 'big', 'very_big'],
                       universe_of_discourse=[0.0, 1.0])
    FS_recommendation_strength_hybrid.add_linguistic_variable("recommendation_coefficient_cf", TLV)

    TLV = AutoTriangle(5,
                       terms=['very_small', 'small', 'medium', 'big', 'very_big'],
                       universe_of_discourse=[0.0, 1.0])
    FS_recommendation_strength_hybrid.add_linguistic_variable("recommendation_coefficient_cb", TLV)

    # Output variable (model strength)
    O1 = TriangleFuzzySet(0.0, 0.1, 0.2, term="very_small")
    O2 = TriangleFuzzySet(0.1,0.3, 0.5,  term="small")
    O3 = TriangleFuzzySet(0.25, 0.5, 0.75, term="medium")
    O4 = TriangleFuzzySet(0.7,0.8, 0.9, term="big")
    O5 = TriangleFuzzySet(0.85, 0.9, 1.0, term="very_big")

    output_var = LinguisticVariable([O1, O2, O3, O4, O5], universe_of_discourse=[0.0, 1.0])
    FS_recommendation_strength_hybrid.add_linguistic_variable("model_strength", output_var)
    """
    ### Trapezoid ###
    """
    B1 = TrapezoidFuzzySet(0, 0, 0.1, 0.2, term="very_small")
    B2 = TrapezoidFuzzySet(0.1, 0.2, 0.3, 0.4, term="small")
    B3 = TrapezoidFuzzySet(0.3, 0.4, 0.6, 0.7, term="medium")
    B4 = TrapezoidFuzzySet(0.6, 0.7, 0.8, 0.9, term="big")
    B5 = TrapezoidFuzzySet(0.8, 0.9, 1.0, 1.0, term="very_big")

    belief_var = LinguisticVariable([B1, B2, B3, B4, B5], universe_of_discourse=[0, 1])
    FS_recommendation_strength_hybrid.add_linguistic_variable("belief_in_cf_model", belief_var)
    FS_recommendation_strength_hybrid.add_linguistic_variable("belief_in_cb_model", belief_var)

    # Recommendation coefficient variables (CF and CB)
    R1 = TrapezoidFuzzySet(0, 0, 0.1, 0.2, term="very_small")
    R2 = TrapezoidFuzzySet(0.1, 0.2, 0.3, 0.4, term="small")
    R3 = TrapezoidFuzzySet(0.3, 0.4, 0.6, 0.7, term="medium")
    R4 = TrapezoidFuzzySet(0.6, 0.7, 0.8, 0.9, term="big")
    R5 = TrapezoidFuzzySet(0.8, 0.9, 1.0, 1.0, term="very_big")

    rec_coef_var = LinguisticVariable([R1, R2, R3, R4, R5], universe_of_discourse=[0, 1])
    FS_recommendation_strength_hybrid.add_linguistic_variable("recommendation_coefficient_cf", rec_coef_var)
    FS_recommendation_strength_hybrid.add_linguistic_variable("recommendation_coefficient_cb", rec_coef_var)

    # Output variable (model strength)
    O1 = TrapezoidFuzzySet(0, 0, 0.1, 0.2, term="very_small")
    O2 = TrapezoidFuzzySet(0.1, 0.2, 0.3, 0.4, term="small")
    O3 = TrapezoidFuzzySet(0.3, 0.4, 0.6, 0.7, term="medium")
    O4 = TrapezoidFuzzySet(0.6, 0.7, 0.8, 0.9, term="big")
    O5 = TrapezoidFuzzySet(0.8, 0.9, 1.0, 1.0, term="very_big")

    O = LinguisticVariable([O1, O2, O3, O4, O5], universe_of_discourse=[0, 1])
    FS_recommendation_strength_hybrid.add_linguistic_variable("model_strength", O)
    """
    ### Gaussian ###
    # Belief variables (CF and CB)

    # Belief variables
    B1 = GaussianFuzzySet(0.1, 0.05, term="very_small")
    B2 = GaussianFuzzySet(0.3, 0.1, term="small")
    B3 = GaussianFuzzySet(0.5, 0.1, term="medium")
    B4 = GaussianFuzzySet(0.7, 0.1, term="big")
    B5 = GaussianFuzzySet(0.9, 0.05, term="very_big")

    belief_var = LinguisticVariable([B1, B2, B3, B4, B5], universe_of_discourse=[0, 1.0])
    FS_recommendation_strength_hybrid.add_linguistic_variable("belief_in_cf_model", belief_var)
    FS_recommendation_strength_hybrid.add_linguistic_variable("belief_in_cb_model", belief_var)

    # Recommendation coefficient variables (CF and CB)
    R1 = GaussianFuzzySet(0.1, 0.1, term="very_small")
    R2 = GaussianFuzzySet(0.3, 0.1, term="small")
    R3 = GaussianFuzzySet(0.5, 0.1, term="medium")
    R4 = GaussianFuzzySet(0.75, 0.1, term="big")
    R5 = GaussianFuzzySet(0.9, 0.05, term="very_big")

    rec_coef_var = LinguisticVariable([R1, R2, R3, R4, R5], universe_of_discourse=[0, 1])
    FS_recommendation_strength_hybrid.add_linguistic_variable("recommendation_coefficient_cf", rec_coef_var)
    FS_recommendation_strength_hybrid.add_linguistic_variable("recommendation_coefficient_cb", rec_coef_var)

    # Output variable (model strength)
    O1 = GaussianFuzzySet(0.1, 0.1, term="very_small")
    O2 = GaussianFuzzySet(0.3, 0.1, term="small")
    O3 = GaussianFuzzySet(0.5, 0.1, term="medium")
    O4 = GaussianFuzzySet(0.75, 0.1, term="big")
    O5 = GaussianFuzzySet(0.9, 0.05, term="very_big")

    O = LinguisticVariable([O1, O2, O3, O4, O5], universe_of_discourse=[0, 1])
    FS_recommendation_strength_hybrid.add_linguistic_variable("model_strength", O)

    FS_recommendation_strength_hybrid.add_rules([
        "IF (belief_in_cf_model IS very_big) AND (recommendation_coefficient_cf IS very_big) THEN (model_strength IS very_big)",
        "IF (belief_in_cb_model IS very_big) AND (recommendation_coefficient_cb IS very_big) THEN (model_strength IS very_big)",
        "IF (belief_in_cf_model IS very_small) AND (recommendation_coefficient_cf IS very_small) THEN (model_strength IS very_small)",
        "IF (belief_in_cb_model IS very_small) AND (recommendation_coefficient_cb IS very_small) THEN (model_strength IS very_small)",
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
        "IF (belief_in_cf_model IS very_big) AND (belief_in_cb_model IS very_big) AND (recommendation_coefficient_cf IS very_big) AND (recommendation_coefficient_cb IS very_big) THEN (model_strength IS very_big)",
        "IF (belief_in_cf_model IS very_small) AND (belief_in_cb_model IS very_small) AND (recommendation_coefficient_cf IS very_small) AND (recommendation_coefficient_cb IS very_small) THEN (model_strength IS very_small)",
        "IF (belief_in_cf_model IS medium) OR (belief_in_cb_model IS medium) OR (recommendation_coefficient_cf IS medium) OR (recommendation_coefficient_cb IS medium) THEN (model_strength IS medium)",
        "IF (belief_in_cf_model IS very_big) AND (recommendation_coefficient_cf IS very_big) AND (belief_in_cb_model IS very_small) THEN (model_strength IS big)",
        "IF (belief_in_cb_model IS very_big) AND (recommendation_coefficient_cb IS very_big) AND (belief_in_cf_model IS very_small) THEN (model_strength IS big)",
        "IF (belief_in_cf_model IS big) AND (recommendation_coefficient_cf IS very_big) AND (belief_in_cb_model IS medium) AND (recommendation_coefficient_cb IS medium) THEN (model_strength IS very_big)",
        "IF (belief_in_cf_model IS very_big) AND (recommendation_coefficient_cf IS big) THEN (model_strength IS very_big)",
        "IF (belief_in_cb_model IS very_big) AND (recommendation_coefficient_cb IS big) THEN (model_strength IS very_big)",
        "IF (belief_in_cf_model IS small) AND (recommendation_coefficient_cf IS very_small) THEN (model_strength IS very_small)",
        "IF (belief_in_cb_model IS small) AND (recommendation_coefficient_cb IS very_small) THEN (model_strength IS very_small)",
        "IF (belief_in_cf_model IS big) AND (recommendation_coefficient_cf IS medium) AND (belief_in_cb_model IS small) THEN (model_strength IS big)",
        "IF (belief_in_cf_model IS very_big) AND (recommendation_coefficient_cf IS very_small) AND (belief_in_cb_model IS medium) AND (recommendation_coefficient_cb IS medium) THEN (model_strength IS medium)",
        "IF (belief_in_cf_model IS big) AND (recommendation_coefficient_cf IS medium) THEN (model_strength IS big)",
        "IF (belief_in_cb_model IS big) AND (recommendation_coefficient_cb IS medium) THEN (model_strength IS big)",
        "IF (belief_in_cf_model IS big) AND (belief_in_cb_model IS small) THEN (model_strength IS big)",
        "IF (belief_in_cb_model IS big) AND (belief_in_cf_model IS small) THEN (model_strength IS big)",
        "IF (belief_in_cf_model IS very_big) OR (recommendation_coefficient_cf IS very_big) THEN (model_strength IS very_big)",
        "IF (belief_in_cb_model IS very_big) OR (recommendation_coefficient_cb IS very_big) THEN (model_strength IS very_big)",
        "IF (belief_in_cf_model IS medium) AND (belief_in_cb_model IS medium) AND (recommendation_coefficient_cf IS medium) AND (recommendation_coefficient_cb IS medium) THEN (model_strength IS medium)",
    ])

    FS_recommendation_strength_hybrid.set_variable("belief_in_cf_model", belief_in_model_cf)
    FS_recommendation_strength_hybrid.set_variable("belief_in_cb_model", belief_in_model_cb)

    FS_recommendation_strength_hybrid.set_variable("recommendation_coefficient_cf", recommendation_coefficient_cf)
    FS_recommendation_strength_hybrid.set_variable("recommendation_coefficient_cb", recommendation_coefficient_cb)

    recommendation_fuzzy_hybrid = FS_recommendation_strength_hybrid.inference()

    # Plot membership functions for each variable
    titles = ['Belief in CF Model', 'Belief in CB Model', 'Model Strength']
    x_labels = ['Membership Value', 'Membership Value', 'Membership Value']
    y_labels = ['Degree of Belief', 'Degree of Belief', 'Model Strength']

    plot_fuzzy(belief_var, rec_coef_var, O, titles, x_labels, y_labels)

    """
    # Plot output surface for belief_in_cf_model and recommendation_coefficient_cf
    plot_output_surface(FS_recommendation_strength_hybrid,
                        FS_recommendation_strength_hybrid.get_linguistic_variable("belief_in_cf_model"),
                        FS_recommendation_strength_hybrid.get_linguistic_variable("recommendation_coefficient_cf"),
                        FS_recommendation_strength_hybrid.get_linguistic_variable("model_strength"))

    # Plot output surface for belief_in_cb_model and recommendation_coefficient_cb
    plot_output_surface(FS_recommendation_strength_hybrid,
                        FS_recommendation_strength_hybrid.get_linguistic_variable("belief_in_cb_model"),
                        FS_recommendation_strength_hybrid.get_linguistic_variable("recommendation_coefficient_cb"),
                        FS_recommendation_strength_hybrid.get_linguistic_variable("model_strength"))
    """

    logging.debug("fuzzy ensemble coefficient:")
    logging.debug(recommendation_fuzzy_hybrid)

    return recommendation_fuzzy_hybrid['model_strength']




