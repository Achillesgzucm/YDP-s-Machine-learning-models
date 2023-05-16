import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import optuna
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
import time
import timeit
import shap
import catboost
from catboost import CatBoostClassifier
import matplotlib.pylab as plt
start = time.time()
import lightgbm as lgb
plt.style.use('seaborn-white')
#%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
import matplotlib
from matplotlib import pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
#import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,StratifiedKFold
from sklearn import preprocessing
import operator
#from boruta import BorutaPy
#import lightgbm as lgb
#from imblearn.over_sampling import SMOTE #Over sampling
import time 
from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, f1_score, auc
# Load classifiers
# ----------------
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
#from yellowbrick.classifier import ClassificationReport
import xgboost as xgb
from xgboost import XGBClassifier
# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import tree
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
simplefilter(action='ignore', category=FutureWarning)
# Helper functions
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import openpyxl
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
import feather
from sklearn.model_selection import cross_val_score, StratifiedKFold

outputPath = ''

X_train_scale = ''
y_train = ''
X_train_scale = ''
y_test = ''
X_train = ''

def plot_bootstrap_roc(m, ci, filename=outputPath+r'\img\Bootstrap_ROC_confint.pdf'):
    x = np.linspace(0,1,100)
    plt.figure(figsize=(6,6))
    plt.plot(x, m, c='blue', label='ROC Mean')
    plt.plot(x, ci[0], c='grey', label='95% CI')
    plt.plot(x, ci[1], c='grey')
    plt.fill_between(x, ci[0], ci[1], color='grey', alpha=0.25)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.legend(loc='lower right', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Bootstrap ROC Curve', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(filename)
    plt.show()
def bootstrap_model(model, X, y, X_test, y_test, n_bootstrap, thresh):
    total_recall = []
    total_precision = []
    total_fscore = []
    total_fpr_tpr = []
    size = X.shape[0]

    for _ in range(n_bootstrap):
        boot_ind = np.random.randint(size, size=size)
        X_boot = X.loc[boot_ind]
        y_boot = y.loc[boot_ind]

        clf = model.fit(X_boot, y_boot)
        y_pred = clf.predict_proba(X_test)[:,0]
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            y_test, np.where(y_pred > thresh, 0, 1))

        fpr, tpr, thresholds = metrics.roc_curve(y_test, 1 - y_pred)
        fpr_tpr = (fpr, tpr)
        total_fpr_tpr.append(fpr_tpr)
        total_recall.append(recall[1])
        total_precision.append(precision[1])
        total_fscore.append(fscore[1])

    results = dict(recall=total_recall,
                   precision=total_precision,
                   fscore=total_fscore,
                   fpr_tpr=total_fpr_tpr)

    return results

def roc_interp(fpr_tpr):
    linsp = np.linspace(0, 1, 100)
    n_boot = len(fpr_tpr)
    ys = []
    for n in fpr_tpr:
        x, y = n
        interp = np.interp(linsp, x, y)
        ys.append(interp)
    return ys
def precision_recall_thershold(pred_proba, y_test):
    t_recall_nodiab, t_recall_diab = [], []
    t_precision_nodiab, t_precision_diab = [], []

    for thresh in np.arange(0, 1, 0.01):
        precision, recall, fscore, support = \
                metrics.precision_recall_fscore_support(
                        y_test,
                        np.where(pred_proba[:,0] > thresh, 0, 1))
        recall_nodiab, recall_diab = recall
        precision_nodiab, precision_diab = precision

        t_recall_nodiab.append(recall_nodiab)
        t_recall_diab.append(recall_diab)

        t_precision_nodiab.append(precision_nodiab)
        t_precision_diab.append(precision_diab)

    return t_precision_nodiab, t_precision_diab, \
            t_recall_nodiab, t_recall_diab

def rf_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 2000)
    max_depth = trial.suggest_int('max_depth', 10, 1000)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 200)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 200)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])

    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                criterion=criterion,
                                max_features=max_features,
                                n_jobs=-1,
                                random_state=42)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(rf, X_train_scale, y_train, cv=kf, scoring='accuracy')
    return score.mean()

from optuna.pruners import MedianPruner



n_trials = 1000
print("Optimizing RandomForestClassifier...")
rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(rf_objective, n_trials=n_trials)



print("Best parameters for RandomForestClassifier:")
rf_best_params = rf_study.best_params
print(rf_best_params)


def lgbm_objective(trial):
    num_leaves = trial.suggest_int('num_leaves', 2, 550)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1.0)
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
    max_depth = trial.suggest_int('max_depth', 1, 15)
    min_split_gain = trial.suggest_float('min_split_gain', 0, 1)
    subsample = trial.suggest_float('subsample', 0.1, 1)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1)
    reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-8, 1)
    reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-8, 1)

    lgbm = lgb.LGBMClassifier(num_leaves=num_leaves,
                              learning_rate=learning_rate,
                              n_estimators=n_estimators,
                              max_depth=max_depth,
                              min_split_gain=min_split_gain,
                              subsample=subsample,
                              colsample_bytree=colsample_bytree,
                              reg_alpha=reg_alpha,
                              reg_lambda=reg_lambda,
                              n_jobs=-1,
                              random_state=42)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(lgbm, X_train_scale, y_train, cv=kf, scoring='accuracy') 
    return score.mean()





print("Optimizing LGBMClassifier...")
lgbm_study = optuna.create_study(direction="maximize")
lgbm_study.optimize(lgbm_objective, n_trials=n_trials)



print("Best parameters for LGBMClassifier:")
LGBM_best_params = lgbm_study.best_params
print(LGBM_best_params)



rf_clf = RandomForestClassifier(n_jobs=-1, **rf_best_params).fit(X_train_scale, y_train)
LGBM_clf = lgb.LGBMClassifier(silent=True, n_jobs=-1, **LGBM_best_params).fit(X_train_scale, y_train)




def lr_objective(trial):
    C = trial.suggest_float('C', 1e-6, 1e+6, log=True)
    penalty = trial.suggest_categorical('penalty', [ 'l2', 'none'])
    if penalty == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
    else:
        l1_ratio = None
    model = LogisticRegression(C=C, penalty=penalty, l1_ratio=l1_ratio, random_state=42,n_jobs=-1)



    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score

print("Optimizing LogisticRegression...")
lr_study = optuna.create_study(direction="maximize")
lr_study.optimize(lr_objective, n_trials=n_trials)

print("Best parameters for LogisticRegression:")
lr_best_params = lr_study.best_params
print(lr_best_params)

lr_clf = LogisticRegression(n_jobs=-1, **lr_best_params).fit(X_train_scale, y_train)


def svm_objective(trial):
    C = trial.suggest_float("C", 1e-5, 1e1, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    
    model = SVC(C=C, kernel=kernel, probability=True)
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score

print("Optimizing SVM...")
svm_study = optuna.create_study(direction="maximize")
svm_study.optimize(svm_objective, n_trials=n_trials)

print("Best parameters for SVM:")
svm_best_params = svm_study.best_params
print(svm_best_params)

svm_clf = SVC(**svm_best_params).fit(X_train_scale, y_train)






def gbc_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 1000)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e1, log=True)
    max_depth= trial.suggest_int("max_depth", 1, 400)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 40)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 40)
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])


    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    )
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score
print("Optimizing GradientBoostingClassifier...")
gbc_study = optuna.create_study(direction="maximize")
gbc_study.optimize(gbc_objective, n_trials=n_trials)

print("Best parameters for GradientBoostingClassifier:")
gbc_best_params = gbc_study.best_params
print(gbc_best_params)

gbc_clf = GradientBoostingClassifier(**gbc_best_params).fit(X_train_scale, y_train)
def rf_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 500)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 1, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1
    )
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score

print("Optimizing RandomForestClassifier...")
rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(rf_objective, n_trials=n_trials)

print("Best parameters for RandomForestClassifier:")
rf_best_params = rf_study.best_params
print(rf_best_params)

rf_clf = RandomForestClassifier(**rf_best_params).fit(X_train_scale, y_train)
def knn_objective(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 1, 60)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
    leaf_size = trial.suggest_int("leaf_size", 1, 1000)
    p = trial.suggest_int("p", 1, 5)
    metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "chebyshev", "minkowski"])

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric,
        n_jobs=-1
    )
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score

print("Optimizing KNeighborsClassifier...")
knn_study = optuna.create_study(direction="maximize")
knn_study.optimize(knn_objective, n_trials=n_trials)

print("Best parameters for KNeighborsClassifier:")
knn_best_params = knn_study.best_params
print(knn_best_params)

knn_clf = KNeighborsClassifier(**knn_best_params).fit(X_train_scale, y_train)


def dt_objective(trial):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    splitter = trial.suggest_categorical("splitter", ["best", "random"])
    max_depth = trial.suggest_int("max_depth", 1, 500)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 200)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 200)
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])

    model = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    )
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score

print("Optimizing DecisionTreeClassifier...")
dt_study = optuna.create_study(direction="maximize")
dt_study.optimize(dt_objective, n_trials=n_trials)

print("Best parameters for DecisionTreeClassifier:")
dt_best_params = dt_study.best_params
print(dt_best_params)

dt_clf = DecisionTreeClassifier(**dt_best_params).fit(X_train_scale, y_train)


def ad_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1)
    algorithm = trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"])

    base_estimator = DecisionTreeClassifier(
        max_depth=dt_best_params["max_depth"],
        max_leaf_nodes=62
    )

    model = AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm=algorithm
    )
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score

print("Optimizing AdaBoostClassifier...")
ad_study = optuna.create_study(direction="maximize")
ad_study.optimize(ad_objective, n_trials=n_trials)

print("Best parameters for AdaBoostClassifier:")
ad_best_params = ad_study.best_params
print(ad_best_params)

ad_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=dt_best_params["max_depth"], max_leaf_nodes=62), **ad_best_params).fit(X_train_scale, y_train)
def cat_objective(trial):
    depth = trial.suggest_int("depth", 2, 12)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1)
    random_strength = trial.suggest_int("random_strength", 1, 10)
    iterations = trial.suggest_int("iterations", 50, 500)
    

    model = CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        random_strength=random_strength,
        iterations=iterations,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        silent=True
    )
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score

print("Optimizing CatBoostClassifier...")
cat_study = optuna.create_study(direction="maximize")
cat_study.optimize(cat_objective, n_trials=n_trials)

print("Best parameters for CatBoostClassifier:")
cat_best_params = cat_study.best_params
print(cat_best_params)

cat_clf = CatBoostClassifier(**cat_best_params, silent=True).fit(X_train_scale, y_train)

def xgb_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 1000, step=100)
    max_depth = trial.suggest_int("max_depth", 1, 100)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 1.0, log=True)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1
    )
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score
print("Optimizing XGBoost...")
xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=n_trials)

print("Best parameters for XGBoost:")
xgb_best_params = xgb_study.best_params
print(xgb_best_params)

xgb_clf = XGBClassifier(**xgb_best_params).fit(X_train_scale, y_train)





def nb_objective(trial):
    var_smoothing = trial.suggest_float("var_smoothing", 1e-11, 1e-7, log=True)

    model = GaussianNB(var_smoothing=var_smoothing)
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score

print("Optimizing Naive Bayes...")
nb_study = optuna.create_study(direction="maximize")
nb_study.optimize(nb_objective, n_trials=n_trials)

print("Best parameters for Naive Bayes:")
nb_best_params = nb_study.best_params
print(nb_best_params)

nb_clf = GaussianNB(**nb_best_params).fit(X_train_scale, y_train)






def mlp_objective(trial):
    hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes", 10, 200)
    activation = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])
    solver = trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
    learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 1e-1, log=True)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        random_state=42,
        max_iter=1000
    )
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score

print("Optimizing Multilayer Perceptrons...")
mlp_study = optuna.create_study(direction="maximize")
mlp_study.optimize(mlp_objective, n_trials=n_trials)

print("Best parameters for Multilayer Perceptrons:")
mlp_best_params = mlp_study.best_params
print(mlp_best_params)

mlp_clf = MLPClassifier(**mlp_best_params).fit(X_train_scale, y_train)




from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def lda_objective(trial):
    shrinkage = trial.suggest_float("shrinkage", 0, 1)
    solver = trial.suggest_categorical("solver", ["lsqr", "eigen"])

    if solver == "svd":
        model = LinearDiscriminantAnalysis(solver=solver)
    else:
        model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score

print("Optimizing Linear Discriminant Analysis...")
lda_study = optuna.create_study(direction="maximize")
lda_study.optimize(lda_objective, n_trials=n_trials)

print("Best parameters for Linear Discriminant Analysis:")
lda_best_params = lda_study.best_params
print(lda_best_params)

lda_clf = LinearDiscriminantAnalysis(**lda_best_params).fit(X_train_scale, y_train)



def qda_objective(trial):
    reg_param = trial.suggest_float("reg_param", 0, 1)

    model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score

print("Optimizing Quadratic Discriminant Analysis...")
qda_study = optuna.create_study(direction="maximize")
qda_study.optimize(qda_objective, n_trials=n_trials)

print("Best parameters for Quadratic Discriminant Analysis:")
qda_best_params = qda_study.best_params
print(qda_best_params)

qda_clf = QuadraticDiscriminantAnalysis(**qda_best_params).fit(X_train_scale, y_train)



from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestCentroid

def sgd_objective(trial):
    alpha = trial.suggest_float("alpha", 1e-6, 1e-3, log=True)
    loss = trial.suggest_categorical("loss", ["log", "modified_huber"])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    learning_rate = trial.suggest_categorical("learning_rate", ["constant", "optimal", "invscaling", "adaptive"])


    eta0 = trial.suggest_float("eta0", 1e-5, 1, log=True)

    model = SGDClassifier(alpha=alpha, loss=loss, penalty=penalty, learning_rate=learning_rate, eta0=eta0, random_state=42)
    score = cross_val_score(model, X_train_scale, y_train, cv=5, n_jobs=-1).mean()
    return score

print("Optimizing Stochastic Gradient Descent...")
sgd_study = optuna.create_study(direction="maximize")
sgd_study.optimize(sgd_objective, n_trials=n_trials)

print("Best parameters for Stochastic Gradient Descent:")
sgd_best_params = sgd_study.best_params
print(sgd_best_params)

sgd_clf = SGDClassifier(**sgd_best_params).fit(X_train_scale, y_train)







def multinomial_logreg_objective(trial):
    C = trial.suggest_float("C", 1e-6, 1e6, log=True)
    multi_class = 'multinomial'
    solver = trial.suggest_categorical("solver", ["newton-cg", "sag", "saga", "lbfgs"])
    penalty = trial.suggest_categorical('penalty', [ 'l2', 'none'])


    model = LogisticRegression(penalty=penalty,C=C, multi_class=multi_class, solver=solver)
    score = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1).mean()
    return score

multinomial_logreg_study = optuna.create_study(direction="maximize")
multinomial_logreg_study.optimize(multinomial_logreg_objective, n_trials=n_trials)
multinomial_logreg_best_params = multinomial_logreg_study.best_params




from sklearn.naive_bayes import MultinomialNB

def multinomial_nb_objective(trial):
    alpha = trial.suggest_float("alpha", 1e-6, 1e6, log=True)

    model = MultinomialNB(alpha=alpha)
    score = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1).mean()
    return score

multinomial_nb_study = optuna.create_study(direction="maximize")
multinomial_nb_study.optimize(multinomial_nb_objective, n_trials=n_trials)
multinomial_nb_best_params = multinomial_nb_study.best_params




from sklearn.naive_bayes import ComplementNB

def complement_nb_objective(trial):
    alpha = trial.suggest_float("alpha", 1e-6, 1e6, log=True)

    model = ComplementNB(alpha=alpha)
    score = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1).mean()
    return score

complement_nb_study = optuna.create_study(direction="maximize")
complement_nb_study.optimize(complement_nb_objective, n_trials=n_trials)
complement_nb_best_params = complement_nb_study.best_params



from sklearn.ensemble import ExtraTreesClassifier

def extra_trees_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 1000)
    max_depth = trial.suggest_int("max_depth", 1, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])

    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
    )
    score = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1).mean()
    return score

extra_trees_study = optuna.create_study(direction="maximize")
extra_trees_study.optimize(extra_trees_objective, n_trials=n_trials)
extra_trees_best_params = extra_trees_study.best_params




from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, WhiteKernel



from sklearn.neighbors import RadiusNeighborsClassifier

def radius_neighbors_objective(trial):
    radius = trial.suggest_float("radius", 10, 100)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    p = trial.suggest_int("p", 1, 5)

    model = RadiusNeighborsClassifier(radius=radius, weights=weights, p=p,outlier_label=-1)
    score = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1).mean()
    return score

radius_neighbors_study = optuna.create_study(direction="maximize")
radius_neighbors_study.optimize(radius_neighbors_objective, n_trials=n_trials)
radius_neighbors_best_params = radius_neighbors_study.best_params



from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

def hist_gb_objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1, log=True)
    max_iter = trial.suggest_int("max_iter", 10, 1000)
    max_depth = trial.suggest_int("max_depth", 1, 50)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 128)
    l2_regularization = trial.suggest_float("l2_regularization", 1e-6, 1, log=True)

    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
    )
    score = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1).mean()
    return score

hist_gb_study = optuna.create_study(direction="maximize")
hist_gb_study.optimize(hist_gb_objective, n_trials=n_trials)
hist_gb_best_params = hist_gb_study.best_params