import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# matplotlib.use('TKAgg',warn=False, force=True)
def plot_confusion_matrix(cm_matrix, savename):
    cm = cm_matrix
    plt.title('Confusion matrix')
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True classes')
    plt.savefig(savename)

# /home/dipesh/video-action-recognition/run/concat_fwd_rvs/C3D-ucf101_epoch-99_train.npy
# /home/dipesh/video-action-recognition/run/concat_fwd_rvs/C3D-ucf101_epoch-99_val.npy
npy_path = "/home/dipesh/video-action-recognition/run/concat_rvs_fwd/C3D-ucf101_epoch-99_train.npy"
cm_matrix = np.load(npy_path)
save_name = "conf_mat/concat_rvs_fwd_train"
np.savetxt(f"{save_name}.csv", cm_matrix, delimiter=",")
# plot_confusion_matrix(cm_matrix,f"{save_name}.jpg")
