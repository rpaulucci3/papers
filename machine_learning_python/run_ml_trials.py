import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def dict_to_sorted_dataframe(cv_results):
    cv_results_dataframe = pd.DataFrame.from_dict(cv_results)
    cv_results_dataframe.sort_values("rank_test_score", ascending=False)
    return cv_results_dataframe


def load_and_clean_dataset():
    # Load dataset only once
    (x, y) = load_breast_cancer(return_X_y=True)
    # Perform in-place numerical encoding of categorical data
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    # Split train and test data
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.25, random_state=0)
    # Scale X data
    sc = StandardScaler()
    train_x = sc.fit_transform(train_x)
    test_x = sc.fit_transform(test_x)
    
    print("Loaded dataset")
    return (train_x, test_x, train_y, test_y)


def rows_within_1_sem_of_best(results):
    K = len([x for x in list(results.keys()) if x.startswith("split")])

    # all_params = pd.Series(results["params"])
    all_means = pd.Series(results["mean_test_score"])
    all_stdevs = pd.Series(results["std_test_score"])
    all_sems = all_stdevs / np.sqrt(K)

    max_score_mean = all_means.max()
    sem = all_sems[all_means.idxmax()]
    acceptable_rows = results[all_means >= (max_score_mean - sem)]
    acceptable_rows
    return acceptable_rows


def _run_trials(x, y, estimator_func, p_grid, K):
    cv_split = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
    print("Starting GSCV with K={} for function {} with grid {}".format(
        K, type(estimator_func), p_grid))
    clf = GridSearchCV(
        estimator=estimator_func,
        param_grid=p_grid,
        cv=cv_split,
        n_jobs=-1,
        # refit=best_alpha_index
        verbose=1
    )
    clf.fit(x, y)
    clf.cv_results_
    return clf.cv_results_


def _svc_trials(train_x, test_x, train_y, test_y, K):
    p_grid_svc = {
        "C": [0.1, 1, 10],
        "gamma": [.001, .01, .1],
        "kernel": ["poly", "rbf", "sigmoid"],
        "random_state": [0],
        "cache_size": [1000]
    }
    print("Starting SVC trials")
    cv_results_svc = _run_trials(train_x, train_y, SVC(), p_grid_svc, K)
    cv_results_svc_sorted_df = dict_to_sorted_dataframe(cv_results_svc)
    best_rows_svc = rows_within_1_sem_of_best(cv_results_svc_sorted_df)
    best_params_svc = best_rows_svc["params"].tolist()

    finalists_svc_list = []
    for i in range(len(best_params_svc)):
        final_svc = SVC(**best_params_svc[i])
        final_svc.fit(train_x, train_y)
        curr_score = final_svc.score(test_x, test_y)
        finalists_svc_list.append(
            {"score": curr_score, **best_params_svc[i]})
        print("SVC Score: {} for params {}".format(
            curr_score, best_params_svc[i]))
    finalists_svc = pd.DataFrame(finalists_svc_list)
    finalists_svc.sort_values(["score"], ascending=False, inplace=True)
    return finalists_svc


def _mlp_trials(train_x, test_x, train_y, test_y, K):
    p_grid_mlp = {
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "momentum": [0.1, 0.9],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "alpha": [0.0001, 0.001],
        "random_state": [0],
        "max_iter": [5000]
    }
    print("Starting MLP trials")
    cv_results_mlp = _run_trials(
        train_x, train_y, MLPClassifier(), p_grid_mlp, K)
    cv_results_mlp_sorted_df = dict_to_sorted_dataframe(cv_results_mlp)
    best_rows_mlp = rows_within_1_sem_of_best(cv_results_mlp_sorted_df)
    best_params_mlp = best_rows_mlp["params"].tolist()

    finalists_mlp_list = []
    for j in range(len(best_params_mlp)):
        final_mlp = MLPClassifier(**best_params_mlp[j])
        final_mlp.fit(train_x, train_y)
        curr_score = final_mlp.score(test_x, test_y)
        finalists_mlp_list.append(
            {"score": curr_score, **best_params_mlp[j]})
        print("MLP Score: {} for params {}".format(
            curr_score, best_params_mlp[j]))
    finalists_mlp = pd.DataFrame(finalists_mlp_list)
    finalists_mlp.sort_values(["score"], ascending=False, inplace=True)
    return finalists_mlp


def _xgb_trials(train_x, test_x, train_y, test_y, K):
    p_grid_xgb = {
        "objective": ["binary:logistic"],
        "booster": ["gbtree", "dart"],
        "learning_rate": [0.1, 0.3],
        "max_depth": [3, 6, 9],
        "min_child_weight": [1, 5, 10],
        "n_estimators": [500],
        # "tree_method": "gpu_hist",  # Comment this out if no CUDA GPU
        "seed": [0]
    }
    print("Starting XGB trials")
    cv_results_xgb = _run_trials(
        train_x, train_y, XGBClassifier(), p_grid_xgb, K)
    cv_results_xgb_sorted_df = dict_to_sorted_dataframe(cv_results_xgb)
    best_rows_xgb = rows_within_1_sem_of_best(cv_results_xgb_sorted_df)
    best_params_xgb = best_rows_xgb["params"].tolist()

    finalists_xgb_list = []
    for l in range(len(best_params_xgb)):
        final_xgb = XGBClassifier(**best_params_xgb[l])
        final_xgb.fit(train_x,
                      train_y,
                      verbose=True
                      )
        curr_score = final_xgb.score(test_x, test_y)
        finalists_xgb_list.append(
            {"score": curr_score, **best_params_xgb[l]})
        print("XGB Score: {} for params {}".format(
            curr_score, best_params_xgb[l]))
    finalists_xgb = pd.DataFrame(finalists_xgb_list)
    finalists_xgb.sort_values(["score"], ascending=False, inplace=True)
    return finalists_xgb


def main():
    train_x, test_x, train_y, test_y = load_and_clean_dataset()
    K = 15
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.precision', 15)

    start_time_svc = time.time()
    start_cpu_time_svc = time.process_time()
    
    finalists_svc = _svc_trials(train_x, test_x, train_y, test_y, K)
    
    end_time_svc = time.time()
    end_cpu_time_svc = time.process_time()
    
    print("SVC results:")
    print(finalists_svc)
    
    time_elapsed_svc = end_time_svc - start_time_svc
    cpu_time_elapsed_svc = end_cpu_time_svc - start_cpu_time_svc
    print(f"Time (seconds) elapsed for SVC:  {time_elapsed_svc}")
    print(f"CPU Time (seconds) elapsed for SVC:  {cpu_time_elapsed_svc}")

    start_time_mlp = time.time()
    start_cpu_time_mlp = time.process_time()
    
    finalists_mlp = _mlp_trials(train_x, test_x, train_y, test_y, K)
    
    end_time_mlp = time.time()
    end_cpu_time_mlp = time.process_time()
    
    print("MLP results:")
    print(finalists_mlp)
    
    time_elapsed_mlp = end_time_mlp - start_time_mlp
    cpu_time_elapsed_mlp = end_cpu_time_mlp - start_cpu_time_mlp
    print(f"Time (seconds) elapsed for MLP: {time_elapsed_mlp}")
    print(f"CPU Time (seconds) elapsed for MLP: {cpu_time_elapsed_mlp}")

    start_time_xgb = time.time()
    start_cpu_time_xgb = time.process_time()
    
    finalists_xgb = _xgb_trials(train_x, test_x, train_y, test_y, K)
    
    end_time_xgb = time.time()
    end_cpu_time_xgb = time.process_time()
    
    print("XGB results:")
    print(finalists_xgb)
    
    time_elapsed_xgb = end_time_xgb - start_time_xgb
    cpu_time_elapsed_xgb = end_cpu_time_xgb - start_cpu_time_xgb
    print("Time (seconds) elapsed for XGB: {time_elapsed_xgb}")
    print("CPU Time (seconds) elapsed for XGB: {cpu_time_elapsed_xgb}")
    
    print("-------------------------")
    print("Summary of final results:")
    print("Best SVC: ")
    print(finalists_svc.head(1))
    print("Best MLP: ")
    print(finalists_mlp.head(1))
    print("Best XGB: ")
    print(finalists_xgb.head(1))


if __name__ == "__main__":
    main()
