import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, r2_score, mean_squared_error, classification_report, make_scorer
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix


from rfpimp import importances


pd.set_option('display.max_columns', 500)
pd.options.display.max_rows = 500


def random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators='warn',
                                criterion='gini',
                                max_depth=None,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_features='auto',
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_impurity_split=None,
                                bootstrap=True,
                                oob_score=False,
                                n_jobs=None,
                                random_state=None,
                                verbose=0,
                                warm_start=False,
                                class_weight=None)
    rf.fit(X_train, y_train)
    RF_pred = rf.predict(X_test)

    return classification_report(y_test, RF_pred))



#permutation importance plots
def permutation_importance(model, X_test, y_test):
    imp = importances(rf, X_test, y_test)
    viz = plot_importances(imp[0:9],  yrot=0,
                            label_fontsize=12,
                            width=12,
                            minheight=1.5,
                            vscale=2.0,
                            imp_range=(0, imp['Importance'].max() + .03),
                            color='#484c51',
                            bgcolor='#F1F8FE',  # seaborn uses '#F1F8FE'
                            xtick_precision=2,
                            title='Permutation Importances')

    viz.view()


#plot partial dependences
def partial_dependence(feature_index, X):
    features = [feature_index]
    plot_partial_dependence(rf, X, features, feature_names=X.columns,
                                n_jobs=3, grid_resolution=50, target="high")
    fig = plt.gcf()
    fig.suptitle('Partial Dependence of Negative Sentiment \n On High Obesity Prevalance\n'
                     )
    plt.subplots_adjust(top=0.8) 
    plt.savefig('images/pdp_negative.png')
