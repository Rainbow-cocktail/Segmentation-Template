import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
plt.rcParams['font.family'] = 'SimHei'

npz_path = r"D:\python_project\HSI\HSI_segmentation_template\outputs\unet-mobilenetv3\run_20250327-205731\results\classification_data_epoch7.npz"
data = np.load(npz_path)
y_true = data["y_true"]  # [100000,]
y_pred = data["y_pred"]  # [100000, 2]
num_classes = y_pred.shape[1]
y_true_bin = np.eye(num_classes)[y_true]  # shape: (100000, 2)

class_indices = range(0, num_classes)
class_names = ["背景", "建筑"]
plt.figure(figsize=(6, 5))

for i in class_indices:
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {score:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()