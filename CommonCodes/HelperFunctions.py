import os
import numpy as np
import pandas as pd
from reduction import svfs
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from imblearn.under_sampling import RandomUnderSampler

def SVFS(train_x, train_y, th_irr=3, diff_threshold=1.7, th_red=4, k=100, alpha=50, beta=5):   
    _features = []
    fs = svfs(train_x, train_y, th_irr, diff_threshold, th_red, k, alpha, beta)
    reduced_data = fs.reduction()
    high_x = fs.high_rank_x()
    clean_features = high_x[reduced_data]
    dic_cls = fs.selection(high_x,reduced_data,clean_features)
    J = list(dic_cls.keys())
    selected_features = [clean_features[i] for i in J]
    _features.append(selected_features)
    return _features[0]

def oversample(df):
    smt = SMOTETomek()
    y = df.iloc[:,-1]
    X = df.iloc[:,:-1]
    X, y = smt.fit_resample(X, y)
    X['label'] = y
    X.columns = np.arange(len(X.columns))
    df_oversample = X.copy()
    return df_oversample

def undersample(df):
    rus = RandomUnderSampler()
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    X_resampled, y_resampled = rus.fit_resample(X, y)
    df_undersampled = pd.concat([X_resampled, y_resampled], axis=1)
    df_undersampled.columns = np.arange(len(df_undersampled.columns))
    return df_undersampled

def minmax_scaler(vector):
    scaler = MinMaxScaler()
    vector_scaled = scaler.fit_transform(np.array(vector).reshape(-1, 1))
    return vector_scaled.flatten()

def datasetXY_S(df):
    x = df.iloc[:,:-1].copy()
    y = df.iloc[:,-1].copy()
    return x, y

def datasetXY_M(x, y):
    xx = x.copy()
    if type(xx) == np.ndarray:
        xx = pd.DataFrame(xx)
        xx['label'] = list(y)
    else:
        xx['label'] = list(y)
    return xx

def probToDummy(arr):
    max_indices = np.argmax(arr, axis=1)
    new_arr = np.zeros_like(arr)
    new_arr[np.arange(len(arr)), max_indices] = 1
    return new_arr

def initializeMATPLOT():
    plt.rcParams['axes.facecolor'] = [0.9,0.9,0.9]
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize' ] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['savefig.transparent'] = False

    plt.rcParams.update({
    "figure.facecolor":  (0.9, 0.9, 0.9, 1),
    "axes.facecolor":    (0.9, 0.9, 0.9, 1),
    "savefig.facecolor": (0.0, 0.0, 1.0, 0.0),
    })

def dummy_pred_proba_to_normal(lst):
    r = []
    for a in lst:
        if a[0] > a[1]:
            r.append(0)
        else:
            r.append(1)
    r = np.array(r)
    return r

def precision_recall(y_test, ensemble_pred_probs):
    y_test_dummy = np.array(pd.get_dummies(y_test))
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(2):
        precision[i], recall[i], _ = precision_recall_curve(y_test_dummy[:, i], ensemble_pred_probs[:, i])
        average_precision[i] = average_precision_score(y_test_dummy[:, i], ensemble_pred_probs[:, i])
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_dummy.ravel(), ensemble_pred_probs.ravel())
    average_precision["micro"] = average_precision_score(y_test_dummy, ensemble_pred_probs, average="micro")
    return precision, recall, average_precision

def init_precision_recall_plots(fig, ax, name, save):
    if not os.path.exists('Pictures'):
        os.mkdir('Pictures')
    ax.set_aspect('equal', 'box')

    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)

    ax.plot([0, 1.04], [0, 1.04], color='grey', linewidth=1, linestyle='--')

    ax.legend(loc='lower right', frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    if save:
        fig.savefig('Pictures/'+name+'_precision_recall.png', format='png', dpi=500)
    return ax

def writeToFile(name, lst):
    with open(name, mode='a+') as file:
        file.seek(0)
        content = file.read().strip()
        if content:
            file.write(',' + ','.join(lst))
        else:
            file.write(','.join(lst))

def plot_confusion_matrix(
    _USE_FIRST_N_GENES_CONF,
    conf_mat,
    hide_spines=False,
    hide_ticks=False,
    figsize=None,
    cmap=None,
    colorbar=False,
    show_absolute=True,
    show_normed=False,
    norm_colormap=None,
    class_names=None,
    figure=None,
    axis=None,
    fontcolor_threshold=0.5,
    name=None
):
    if not os.path.exists('Pictures'):
        os.mkdir('Pictures')

    if not (show_absolute or show_normed):
        raise AssertionError("Both show_absolute and show_normed are False")

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype("float") / total_samples

    if figure is None and axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif axis is None:
        fig = figure
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig, ax = figure, axis

    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat) * 1.25, len(conf_mat) * 1.25)

    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap, norm=norm_colormap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap, norm=norm_colormap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                num = conf_mat[i, j].astype(np.int64)
                cell_text += format(num, "d")
                if show_normed:
                    cell_text += "\n" + "("
                    cell_text += format(normed_conf_mat[i, j], ".2f") + ")"
            else:
                cell_text += format(normed_conf_mat[i, j], ".2f")

            if show_normed:
                ax.text(
                    x=j,
                    y=i,
                    s=cell_text,
                    va="center",
                    ha="center",
                    color=(
                        "white"
                        if normed_conf_mat[i, j] > 1 * fontcolor_threshold
                        else "black"
                    ),
                )
            else:
                ax.text(
                    x=j,
                    y=i,
                    s=cell_text,
                    va="center",
                    ha="center",
                    color="white"
                    if conf_mat[i, j] > np.max(conf_mat) * fontcolor_threshold
                    else "black",
                )
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(
            tick_marks, class_names, rotation=45, ha="right", rotation_mode="anchor"
        )
        plt.yticks(tick_marks, class_names)

    if hide_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    ax.set_title(name+f' with the first {_USE_FIRST_N_GENES_CONF} genes')
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    plt.tight_layout()
    plt.savefig('Pictures/'+name+'_CONF.png', format='png', dpi=500)
    return fig, ax