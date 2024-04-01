import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import yaml
import catboost as ctb
from catboost import CatBoostClassifier, Pool
from my_utils import show_metrics

from sklearn.metrics import recall_score, precision_score, confusion_matrix, \
                            matthews_corrcoef, f1_score, roc_auc_score, \
                            average_precision_score, roc_curve, classification_report

# Красивое отображение внутри ноутбука

class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

# Необходимая метрика

def AUC_ctb(ytrue,ypred):
    
    # Формируем датасет
    df = pd.DataFrame({'target': ytrue, 'prediction': ypred})
    auc_ctb = roc_auc_score(df.target, df.prediction)
    return auc_ctb
        
def color(x):
    return bcolors.BOLD + bcolors.OKBLUE + x + bcolors.ENDC

def selection_threshold_by_f1(y_true, y_score):
    aps =[]
    thresholds = np.linspace(0.01, 1, 1000)

    for threshold in thresholds:
        aps.append(f1_score(y_true, y_score > threshold))
    index = np.argmax(aps)
    return thresholds[index]

def main_metrics(y_true, y_score, threshold, name, k):
    """
    y_true - vector with classes
    y_score - vector with probabilities
    """
    y_pred = (y_score >= threshold).astype(int)
    
    df = pd.DataFrame({'target': y_true, 'prediction': y_score})
    df = df.sort_values('prediction', ascending=False)
    k = int(df.shape[0]*k)
    rec_metric = df.iloc[:k]['target'].sum() / df['target'].sum()
    
    print(color('Recall@{}:'.format(k)), rec_metric)
    print(color('AP: '), average_precision_score(y_true, y_score))
    print(color("MCC: "), matthews_corrcoef(y_true, y_pred))
    print(color("F1: "), f1_score(y_true, y_pred))
    print(color("ROC-AUC: "), roc_auc_score(y_true, y_score))
    print(color('GINI: '), 2 * roc_auc_score(y_true, y_score) - 1)
    print(classification_report(y_true, y_pred))

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='');
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix ' + name, fontsize=20)
    
def plot_roc_curve_image(data,target,score,name):
    plt.figure(figsize=(10, 10));

    fpr_ens, tpr_ens, _ = roc_curve(data[target].values, data[score].values)
    auc_score_ens = roc_auc_score(y_true=data[target].values, y_score=data[score].values)

    lw = 2
    plt.plot(fpr_ens, tpr_ens, color='green',
             lw=lw, label='Финальная модель (GINI = {:.3f})'.format(2 * auc_score_ens - 1));
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--', label = 'Случайная модель');
    plt.xlim([-0.05, 1.05]);
    plt.ylim([-0.05, 1.05]);
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    lgd = plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol = 2);
    plt.xticks(np.arange(0, 1.01, 0.05), rotation = 45);
    plt.yticks(np.arange(0, 1.01, 0.05));
    plt.grid(color='gray', linestyle='-', linewidth=1);
    plt.title('ROC кривая (Финальный GINI = {:.3f}) '.format(2 * auc_score_ens - 1) + name)

def show_monitoring_metrics(df,score,target,features,model,k):

    df.sort_values(score, ascending = False, inplace = True)
   
    # Counts
    count_df = df.shape[0]

    # Target counts
    df_target_cnt = df[target].sum()

    print(color('Количество единиц в df: '), str(int(df_target_cnt)))
    print()
    print(color('Количество строк в df: '), str(int(count_df)))
    print()
    
    # Подбор Threshold
    threshold = selection_threshold_by_f1(df[target], df[score])
    print(color('Threshold by f1: '), threshold)
    print()

    # Результаты классификации на df
    print(bcolors.BOLD + bcolors.OKGREEN + 'df' + bcolors.ENDC)
    main_metrics(df[target], df[score], threshold, 'df', k)
    plot_roc_curve_image(df,target,score,'df')    



    # feature importance
    feature_imp = pd.DataFrame(sorted(zip(model.get_feature_importance(), df[features].columns)),
                               columns=['Value', 'Feature'])

    plt.figure(figsize=(10, 15))
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[:40]);
    plt.show()
