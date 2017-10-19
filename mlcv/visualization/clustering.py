import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from .config import MLCV_CMAP


###########################################################################
def scatter_iteration(ax, data, title='', colors='b', marker='o',
                      pause_time=0.3):
    """Scatter points as estimated in a single iteration of an algorithm.

    Parameters
    ----------
    ax : matplotlib.axes.Axes instance
        The axes to draw in.

    data : array, shape (n_samples, n_features)
        The data to scatter plot.

    title : str (optional)
        Title of the plot.

    colors : array-like, shape (n_samples,) or str(optional
        RGBA color per sample or strings or single string.

    marker : str, (optional)
        The representation of the points.

    pause_time : float (optional)
        How long to wait so the drawing can be rendered and observed.

    """

    plt.sca(ax)
    plt.scatter(data[:, 0], data[:, 1], c=colors, marker=marker, lw=0, s=50)
    plt.title('{}'.format(title), fontweight='bold')
    plt.draw()
    plt.pause(pause_time)


###########################################################################
def draw_ellipses_iteration(ax, data, covs, title='', colors='b', marker='o',
                            ellipses_to_remove=None, pause_time=0.3):
    """Draw ellipses as estimated in a single iteration of an algorithm.

    Parameters
    ----------
    ax : matplotlib.axes.Axes instance
        The axes to draw in.

    data : array, shape (n_components, n_features)
        The data to scatter plot.

    covs : array, shape (n_components, n_features, n_features)
        The covariance matrices of the components

    title : str (optional)
        Title of the plot.

    colors : array, shape (n_samples, 4) or str (optional)
        RGBA color per sample or single string

    marker : str, (optional)
        The representation of the points.

    pause_time : float (optional)
        How long to wait so the drawing can be rendered and observed.

    ellipses_to_remove : list
        List of ellipses from previous iteration(s) to be cleared.

    Returns
    -------
    ellipses : list[matplotlib.patches.Ellipse]
        The drawn ellipses objects.

    """

    if ellipses_to_remove is not None:
        [e.remove() for e in ellipses_to_remove]

    ellipses = []
    plt.sca(ax)
    plt.scatter(data[:, 0], data[:, 1], c=colors, marker=marker, lw=0, s=50)
    n_components, n_features = data.shape
    for k in range(n_components):
        ellipse = draw_ellipse(ax, data[k, :], covs[k, :, :])
        ellipses.append(ellipse)

    plt.title('{}'.format(title), fontweight='bold')
    plt.draw()
    plt.pause(pause_time)
    return ellipses


###########################################################################
def draw_ellipse(ax, center, cov, facecolor='lightblue', edgecolor='r'):
    """

    Parameters
    ----------
    ax : matplotlib.axes.Axes instance
        The axes to draw in.

    center : array-like, shape (2,)
        Center coordinates of the ellipse.

    cov : array, shape (n_features, n_features)
        Covariance matrix associated with this ellipse.

    facecolor : array, shape (n_components, 4) or str (optional)
        Face color of the ellipse.

    edgecolor : str (optional)
        Perimeter color of the eliipse

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
        The drawn ellipse object.

    """
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ellipse = mpatches.Ellipse(xy=center, width=2 * v[0] ** 0.5,
                               height=2 * v[1] ** 0.5, angle=180 + angle,
                               facecolor=facecolor, edgecolor=edgecolor,
                               linewidth=2, zorder=2)
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.5)
    ax.add_artist(ellipse)
    return ellipse


############################################################################
def init_clustering_plot(n_components=2):
    """Initialize a figure to show the consecutive estimations of a clustering.

    Parameters
    ----------
    n_components : int
        Number of components to show.

    Returns
    -------
    ax1 : matplotlib.axes.Axes instance
        First axis, will show the data points.

    ax2 : matplotlib.axes.Axes instance
        Seconds axis, will show the log-likelihood.

    component_colors : array, shape (n_components, 4)
        RGBA colors of the components.
    """

    norm = mcolors.Normalize(vmin=0, vmax=n_components - 1)
    # component_colors = np.array([MLCV_CMAP(float(a + 1) / n_components)
    #                     for a in range(n_components)])
    component_colors = MLCV_CMAP(norm(np.arange(n_components)))

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title('Expectation Maximization')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.title('Log-likelihood Maximization', fontweight='bold')

    return ax1, ax2, component_colors
