import matplotlib.cm as cm


MLCV_CMAP = cm.get_cmap('rainbow')


def set_cmap(cmap):
    global MLCV_CMAP
    MLCV_CMAP = cmap
