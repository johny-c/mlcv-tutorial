import numpy as np
import matplotlib.colors as colors
import seaborn as sns; sns.set()

from .config import CMAP


def proba_to_rgba(proba, return_most_likely=False):
    """Convert class probabilities to RGBA colors.

    Parameters
    ----------
    proba : array, shape (n_samples, n_classes)
        Class probabilities

    Returns
    -------
    rgba : array, shape (n_samples, 4)
        RGB and alpha equal to the max probability.

    """

    n_classes = proba.shape[1]
    norm = colors.Normalize(vmin=0, vmax=n_classes-1)

    most_likely_labels = proba.argmax(axis=1)
    sample_range = np.arange(proba.shape[0])
    intensities = proba[sample_range, most_likely_labels]
    np.clip(intensities, 0.0, 1.0, intensities)

    rgba = CMAP(norm(most_likely_labels))
    rgba[:, -1] = intensities

    if return_most_likely:
        return rgba, most_likely_labels
    else:
        return rgba