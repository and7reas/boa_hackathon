import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Plotting:

def plot_sim_scores(doc_sim_scores):
    '''
    Plots the documentation names in descending order of similarity scores,
    with background color intensity representing the degree of similarity.
    
    Args:
        doc_sim_scores (List[Tuple[str, float]]): List of tuples of the form
        (documentation name, similarity score with documentation of interest),
        sorted in descending order of similarity scores.

    Returns:
        matplotlib.figure.Figure : the plotted figure to appear to the streamlit interface
    '''
    labels, scores = zip(*doc_sim_scores)
    # Normalize scores to [0, 1] for color mapping
    norm_scores = np.array(scores)
    # Create a color map from light blue to deep blue
    cmap = sns.light_palette("blue", as_cmap=True)
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, len(labels) * 0.6))
    for i, (label, score) in enumerate(doc_sim_scores):
        ax.barh(i, 1, color=cmap(score))
        ax.text(0.5, i, f"{label} ({score:.2f})", va='center', ha='center', color='white', fontsize=12)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Similarity Scores by Documentation", fontsize=14)
    fig.tight_layout()
    return fig