# Загрузка необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import regex as re
from transliterate import translit
from datetime import datetime as dt
import yaml
import seaborn as sns
import shap
import pickle
from pdpbox import pdp
from sklearn.model_selection import StratifiedGroupKFold
from eli5.permutation_importance import get_score_importances
from sklearn.metrics import recall_score, precision_recall_curve, precision_score, confusion_matrix, \
                            matthews_corrcoef, make_scorer, f1_score, roc_auc_score, classification_report, \
                            precision_recall_fscore_support, roc_curve, average_precision_score, \
                            average_precision_score, roc_auc_score, roc_curve

from scipy.stats import spearmanr
from tqdm import tqdm, tqdm_notebook

from catboost import cv, CatBoostClassifier, Pool
import catboost as ctb
from hyperopt import fmin, hp, Trials, tpe, space_eval, STATUS_OK

# Подбор гиперпараметров

def my_tune_params(X, 
                   y, 
                   space,
                   params,
                   int_params,
                   folds,
                   early_stopping_rounds,
                   categorical_feature,
                   metric,
                   hyperopt_iters=200):
    
    k = 0
    fold_scores = []
    cvpred = np.zeros(X.shape[0]) + np.nan

    best_iter = []

    target = y

    for tr, te in folds:
        xtr, xte = X.iloc[tr, :], X.iloc[te, :]
        ytr, yte = y.iloc[tr], y.iloc[te]
        xtr = ctb.Pool(xtr, label=ytr, cat_features=categorical_feature)
        xte = ctb.Pool(xte, label=yte, cat_features=categorical_feature)
        model = ctb.train(pool=xtr,
                          params=params,
                          eval_set=[xtr, xte],
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=100
                              )
        cvpred[te] = model.predict(xte)
        fold_scores.append(metric(yte, cvpred[te]))
        best_iter.append(model.best_iteration_)

        k += 1

    def objective(hyperparams):
        for param in int_params:
            hyperparams[param] = int(hyperparams[param])
        for key in hyperparams.keys():
            params[key] = hyperparams[key]
            
        return np.mean(fold_scores)*((-1)**True)

    trials = Trials()
    best = fmin(fn=objective, 
                space=space, 
                trials=trials, 
                algo=tpe.suggest,
                max_evals=hyperopt_iters, 
                verbose=-1, 
                rstate=np.random.RandomState(42),
                catch_eval_exceptions=True)

    hyperparams = space_eval(space, best)
    hyper_score = abs(trials.best_trial['result']['loss'])

    print('Метрика качества и параметры:','\n')
    print("Метрика после подбора параметров {:0.4f}".format(hyper_score),'\n')
    print("Оптимальный набор параметров {}".format(hyperparams))
    
    # Сохраним фичи
    params_to_save = {'hyperparams':hyperparams}

    with open('best_params.yaml', 'w') as f:
        yaml.dump(params_to_save, f)

    return hyperparams


# Обучение CatBoost по кросс-валидации

# вспомогательная функция
def break_rec(x, rec, higher_is_better):
    if higher_is_better:
        return x > rec
    else:
        return x < rec

# обертка
class SklearnWrapper(object):
    def __init__(self, model, app='binary'):
        self.model = model
        self.app = app
        
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
    

# основной класс для CTB
class CTB_cv(object):
    def __init__(self, params, cross_params):
        self.params = params
        self.cross_params = cross_params
        
    def fit(self, X, y, test, name):
        
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
       
        def _score(X, y):
            mean_pred = self.model.predict(X)
            return self.cross_params['metric'](y, mean_pred)
        
        
        k = 0
        self.fold_scores = []
        self.fold_scores_tr = []
        self.fold_preds = []
        self.feature_names = X.columns
        self.cvpred = np.zeros(X.shape[0]) + np.nan
        self.mean_pred = np.zeros(test.shape[0]) + np.nan

        self.imp = np.zeros(X.shape[1])
        
        self.best_iter = []
        
        self.target = y
        
        for tr, te in self.cross_params['folds']:
            xt_r, x_te = X.iloc[tr, :], X.iloc[te, :]
            ytr, yte = y.iloc[tr], y.iloc[te]
            xtr = ctb.Pool(xt_r, label=ytr, cat_features=self.cross_params['categorical_feature'])
            xte = ctb.Pool(x_te, label=yte, cat_features=self.cross_params['categorical_feature'])
            self.model = ctb.train(pool=xtr,
                                   params=self.params,
                                   num_boost_round=self.cross_params['nround'],
                                   eval_set=[xtr, xte],
                                   early_stopping_rounds=self.cross_params['early_stopping_rounds'],
                                   verbose_eval=500
                                  )
            
            self.cvpred[te] = self.model.predict(xte)
            self.fold_scores.append(self.cross_params['metric'](yte, self.cvpred[te]))
            self.fold_scores_tr.append(self.cross_params['metric'](ytr, self.model.predict(xtr)))
            self.best_iter.append(self.model.best_iteration_)
            
            print('Fold {}: {:.5f}'.format(k, self.fold_scores[-1]))
            
            self.imp += self.model.get_feature_importance()
                         
            k += 1

        self.imp /= k
        
        self.mean_pred = np.mean(self.fold_preds, axis=0)
        
        self.all_imp = pd.DataFrame(index=self.feature_names)
        self.all_imp['importance'] = self.imp
        self.all_imp = self.all_imp.sort_values(by='importance', ascending=False)

        self.scores_diff = pd.DataFrame(index=['fold ' + str(i) for i in range(len(self.fold_scores))])
        self.scores_diff['Кросс-валидация (обучение) {}'.format(str(self.cross_params['metric']).upper())] = np.round(self.fold_scores_tr,2)
        self.scores_diff['Кросс-валидация (тест) {}'.format(str(self.cross_params['metric']).upper())] = np.round(self.fold_scores,2)
        self.scores_diff['Относительная разница, %'] = np.round((self.scores_diff['Кросс-валидация (тест) {}'.format(str(self.cross_params['metric']).upper())]-self.scores_diff['Кросс-валидация (обучение) {}'.format(str(self.cross_params['metric']).upper())]) / \
                                   self.scores_diff['Кросс-валидация (тест) {}'.format(str(self.cross_params['metric']).upper())]*100,2)
        
        self.base_score = np.mean(self.fold_scores)
        
        # Test prediction
        xtr = ctb.Pool(X, label=y, 
                       cat_features= self.cross_params['categorical_feature'])
        self.model = ctb.train(pool=xtr,
                               params=self.params,
                               num_boost_round=self.cross_params['nround'],
                               eval_set=[xtr, xte],
                               early_stopping_rounds=self.cross_params['early_stopping_rounds'],
                               verbose_eval=500)
        
        model_name = name
        pickle.dump(self.model, open(model_name+'.pkl', "wb"))
        
        self.train_pred = self.model.predict(X)
        self.test_pred = self.model.predict(test)
        
        eps = 1e-5
        self.error = pd.DataFrame()
        self.error['y-yhat'] = self.target - self.cvpred
        self.error['MAE'] = np.abs(self.error['y-yhat'])
        self.error['MSE'] = (self.target - self.cvpred) ** 2
        self.error['MAPE'] = np.abs((self.target - self.cvpred) / (self.target + eps))
        self.error['MSPE'] = ((self.target - self.cvpred) / (self.target + eps)) ** 2
        
        print()
        print('Mean score: {:.5f}'.format(np.mean(self.fold_scores)))
        print('Std of score: {:.5f}'.format(np.std(self.fold_scores)))
        print('Mean std of test objects: {:.5f}'.format(np.mean(np.std(self.fold_preds, axis=0))))
        
    def score(self, X, y):
        pred = self.model.predict(X)
        
        return self.cross_params['metric'](y,pred)



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
    
def plot_roc_curve_image(data, name):
    plt.figure(figsize=(10, 10));

    fpr_ens, tpr_ens, _ = roc_curve(data['TRUE'].values, data['PREDICTED'].values)
    auc_score_ens = roc_auc_score(y_true=data['TRUE'].values, y_score=data['PREDICTED'].values)

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

def show_metrics(data_train, data_test, model, X_shap, k):

    data_test.sort_values('PREDICTED', ascending = False, inplace = True)
    data_train.sort_values('PREDICTED', ascending = False, inplace = True)
   
    # Counts
    count_train = data_train.shape[0]
    count_test = data_test.shape[0]
    count_full = count_train + count_test

    # Part percent
    train_part_perc = count_train / count_full * 100
    test_part_perc = count_test / count_full * 100

    # Target counts
    train_target_cnt = data_train['TRUE'].sum()
    test_target_cnt = data_test['TRUE'].sum()
    full_target_cnt = train_target_cnt + test_target_cnt

    print(color('Количество единиц в train: '), str(int(train_target_cnt)))
    print(color('Количество единиц в test: '), str(int(test_target_cnt)))
    print(color('Общее количество единиц (train и test): '), str(int(full_target_cnt)))
    print()
    print(color('Количество строк в train: '), str(int(count_train)))
    print(color('Количество строк в test: '), str(int(count_test)))
    print(color('Общее количество строк (train и test): '), str(int(count_full)))
    print()
    print(color('Размер train %: '), str(int(train_part_perc)))
    print(color('Размер test %: '), str(int(test_part_perc)))
    print()
    
    # Подбор Threshold
    threshold = selection_threshold_by_f1(data_test['TRUE'], data_test['PREDICTED'])
    print(color('Threshold by f1: '), threshold)
    print()

    # Результаты классификации на data_train
    print(bcolors.BOLD + bcolors.OKGREEN + 'TRAIN' + bcolors.ENDC)
    main_metrics(data_train['TRUE'], data_train['PREDICTED'], threshold, 'train', k)
    plot_roc_curve_image(data_train, 'train')    

    # Результаты классификации на test
    print(bcolors.BOLD + bcolors.OKGREEN + 'TEST' + bcolors.ENDC)
    main_metrics(data_test['TRUE'], data_test['PREDICTED'], threshold, 'test', k)
    plot_roc_curve_image(data_test, 'test')


    # feature importance
    feature_imp = pd.DataFrame(sorted(zip(model.get_feature_importance(), X_shap.columns)),
                               columns=['Value', 'Feature'])

    plt.figure(figsize=(10, 15))
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[:40]);
    plt.show()
    
    # Shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    shap.summary_plot(shap_values, X_shap, max_display=20, plot_size=True)


# Вызов важность признаков

def get_importances(model, sort='importance'):
    return model.all_imp.sort_values(by=sort, ascending=False)


# Жадный отбор постепенным добавлением признаков

def forward_selection(model, features, X, y, top=None, q=0.5, tol=10):
    model.forward_rec_scores = []

    current_features = []
    if top:
        importances = features[:top]
    else:
        importances = features
    indicator = (not model.cross_params['higher_is_better'] and -1 or 1)
    k = 0

    if model.cross_params['higher_is_better']:
        rec = 0
    else:
        rec = np.inf

    lag_fold_scores = [rec for i in range(len(model.cross_params['folds']))]

    for i in tqdm_notebook(importances):
        current_features.append(i)
        _cvpred = np.zeros(X.shape[0])
        _fold_scores = []

        for tr, te in model.cross_params['folds']:
            xtr, xte = X.loc[tr, current_features], X.loc[te, current_features]
            ytr, yte = y[tr], y[te]
            cat_cols = list(filter(lambda x: x in model.cross_params['categorical_feature'], current_features))
            xtr = ctb.Pool(xtr, label=ytr, cat_features=cat_cols)
            xval = ctb.Pool(xte, label=yte, cat_features=cat_cols)
            f_model = ctb.train(pool=xtr,
                               params=model.params,
                               num_boost_round=model.cross_params['nround'],
                               eval_set=[xtr, xval],
                               early_stopping_rounds=model.cross_params['early_stopping_rounds'],
                               verbose_eval=0
                                  )

            _cvpred[te] = f_model.predict(xte)
            _fold_scores.append(model.cross_params['metric'](yte, _cvpred[te]))

        avg_better = ((indicator * (np.array(_fold_scores) - np.array(lag_fold_scores))) > 0).sum()

        if avg_better > int(q * len(model.cross_params['folds'])):
            score = np.mean(_fold_scores)
        else:
            score = (-1) * indicator * np.inf

        if break_rec(score, rec, model.cross_params['higher_is_better']):
            k = 0
            rec = score
            lag_fold_scores = _fold_scores.copy()
            model.forward_selected_features = current_features
            model.forward_rec_scores.append(rec)
            print('Feature {} has been added -> current score: {}'.format(i, rec))
            print('Score has improved on {} folds'.format(avg_better))
            print()

        else:
            current_features.pop()
            k += 1
            if k > tol:
                model.forward_selected_features = current_features
                return model.forward_selected_features

    model.forward_selected_features = current_features
    return model.forward_selected_features

def plot_forward(model, metric_name, figsize=(17, 6)):
    fig = plt.figure(figsize=figsize)
    plt.ylabel(metric_name)
    plt.title('Forward selection scores')
    plt.plot(model.forward_selected_features, model.forward_rec_scores, linewidth=2, markersize=10, marker='s', linestyle='-')
    plt.plot([model.base_score for c in model.forward_rec_scores], linewidth=4, linestyle=':')
    plt.savefig('forward_{}.png'.format(metric_name), dpi=300)
    plt.show()

    model.forward_table = pd.DataFrame()
    model.forward_table[metric_name] = model.forward_rec_scores
    model.forward_table['uplift'] = (model.forward_table[metric_name] - model.forward_table[metric_name].shift(1).fillna(0)).round(4)
    model.forward_table.index = model.forward_selected_features

# Жадный отбор признаков с постепенным удалением

def backward_selection(model,features,X,y,top=None,tol=10):
    to_drop = []
    if top:
        importances = features[:top]
    else:
        importances = features
    rec = model.base_score
    k = 0

    for i in tqdm_notebook(importances):
        to_drop.append(i)
        current_features = list(set(importances) - set(to_drop))
        _cvpred = np.zeros(X.shape[0])
        _fold_scores = []

        for tr, te in model.cross_params['folds']:
            if len(current_features)>0:
                xtr, xte = X.loc[tr, current_features], X.loc[te, current_features]
                ytr, yte = y[tr], y[te]
                cat_cols = list(filter(lambda x: x in model.cross_params['categorical_feature'], current_features))
                xtr = ctb.Pool(xtr, label=ytr, cat_features=cat_cols)
                xval = ctb.Pool(xte, label=yte, cat_features=cat_cols)

                b_model = ctb.train(pool=xtr,
                                    params=model.params,
                                    num_boost_round=model.cross_params['nround'],
                                    eval_set=[xtr, xval],
                                    early_stopping_rounds=model.cross_params['early_stopping_rounds'],
                                    verbose_eval=0
                                  )

            _cvpred[te] = b_model.predict(xte)
            _fold_scores.append(model.cross_params['metric'](yte, _cvpred[te]))

        score = np.mean(_fold_scores)

        if break_rec(score, rec, model.cross_params['higher_is_better']):
            rec = score
            k = 0
            print('Feature {} has been removed -> current score: {}'.format(i, rec))
        else:
            to_drop.pop()
            k += 1
            if k > tol:
                model.backward_selected_features = list(set(importances) - set(to_drop))
                return model.backward_selected_features

    model.backward_selected_features = list(set(importances) - set(to_drop))
    return model.backward_selected_features

# Проверка на переобучение

def overfitting(y_tr,scr_tr,y_te,scr_te,metric,name):
    metric_tr = np.round(metric(y_tr, scr_tr),2)
    metric_te = np.round(metric(y_te, scr_te),2)
    otn = np.round((metric_te-metric_tr)/metric_tr*100,2)
    return pd.DataFrame({'':['Обучение '+name,'Тест '+name,'Относительная разница, %'],'Значение':[metric_tr,metric_te,otn]}).set_index('').T

# Проверка и отрисовка переобучения по кросс-фолдам

def overfitting_folds_plot(model,name,figsize=(12,7)):
    metric_train = model.scores_diff.iloc[:,0]
    metric_test = model.scores_diff.iloc[:,1]
    folds = model.scores_diff.index

    plt.figure(figsize=figsize)
    plt.plot(folds, metric_train, label=name+' train')
    plt.plot(folds, metric_test, label=name+' test')
    plt.legend()
    plt.title(name+' by folds')
    plt.show()

# Значение метрики на тестовых фолдах (отрисовка)

def plot_dynamic(model,name_metric,figsize=(17, 6)):
    fig = plt.figure(figsize=figsize)
    plt.ylabel('{}'.format(name_metric))
    plt.title('{} by folds'.format(name_metric))
    plt.plot(['fold_' + str(i) for i in range(len(model.cross_params['folds']))], model.fold_scores, linewidth=2,
             markersize=3, marker='s', linestyle='-')
    plt.savefig('{}_by_folds.png'.format(name_metric), dpi=300)
    plt.show()


# Отрисовка среднего таргет рейта и скора по месяцам

def plot_dynamic_time(df,metric,name_metric,date,freq,target,score):

    date_field = date
    date_freq = freq
    tar = target
    scr = score
    aggs = []

    for name, data_grp in tqdm_notebook(df.groupby(date_freq)):

        if np.sum(data_grp[tar]) > 0:
            data_grp = data_grp.copy()
        else:
            continue
        metric_calc = metric(data_grp[tar],data_grp[scr])

        n_obs = np.shape(data_grp)[0]
        avg_tr = np.mean(data_grp[tar])
        avg_pred = np.mean(data_grp[scr])

        agg = {
            'months': name,
            'metric': metric_calc,
            'Колчиество наблюдений': n_obs,
            'AVG_TR': avg_tr,
            'AVG_PRED': avg_pred
        }

        aggs += [agg]


    aggs_dynamic = pd.DataFrame(aggs)
    aggs_dynamic.sort_values(date_freq,
                             inplace=True)
    aggs_dynamic.reset_index(inplace=True,
                             drop=True)

    # plot
    sns.set_style('white')
    fig, ax1 = plt.subplots(figsize=(14, 6))
    plt.ylabel('Количество наблюдений', fontdict={'fontsize': 16})
    plt.xlabel('Период наблюдений', fontdict={'fontsize': 16})

    # N_OBS
    plt.bar(np.arange(0, len(aggs_dynamic)), aggs_dynamic['Колчиество наблюдений'].values,
            color='lightblue', 
            label='Колчиество' +'\n'+ 'наблюдений', align='center')
    plt.grid(axis='y')
    ax1.legend(loc='upper left')
    plt.ylim(0, np.max(aggs_dynamic['Колчиество наблюдений']) + np.std(aggs_dynamic['Колчиество наблюдений']))

    # METRIC and TR
    ax2 = ax1.twinx()
    plt.ylabel('Значение метрики и средний таргет рейт', fontdict={'fontsize': 16})
    plt.plot(aggs_dynamic['metric'].values, linewidth=2, linestyle='-', label=name_metric, marker='o', markersize=8,
             color='grey'
            )
    ax3 = ax1.twinx()
    plt.plot(df.groupby('months',as_index=True)['target'].mean().values, linewidth=2, linestyle='--', label='Target rate', marker='o', markersize=8,
             color='red'
            )
    plt.title('Динамика среднего таргет рейта и '+name_metric, fontdict={'fontsize': 18})


    if aggs_dynamic[date_freq].dtype == '<M8[ns]':
        plt.xticks(np.arange(0, len(aggs_dynamic)
                             ), [str(i.year
                                     ) + '-' + str(i.month) for i in aggs_dynamic[date_freq]])
    else:
        plt.xticks(np.arange(0, len(aggs_dynamic)), [str(i) for i in aggs_dynamic[date_freq]])

    ax1.spines['left'].set_color('w')
    ax1.xaxis.set_tick_params(rotation=45)
    ax2.legend()
    ax3.legend()

    plt.tight_layout()
    plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
    plt.rc('legend', fontsize=15)  # legend fontsize
    plt.show()

# PDP-ICE plots

def plot_pdp(model, X, y, feature, frac_to_plot=5000, name=''):
    pdp_iso = pdp.pdp_isolate(model=model.model, dataset=X, feature=feature, model_features=model.feature_names)
    pdp.pdp_plot(pdp_iso, feature, plot_lines=True, frac_to_plot=frac_to_plot, x_quantile=True, center=True)
    plt.savefig(name + '_PDP_{}_{}_samples.png'.format(feature, frac_to_plot), dpi=300)
    plt.show()
