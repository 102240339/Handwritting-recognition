"""Reproduce the MNIST model comparison and training-size experiment."""
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import SGD

ROOT = Path(__file__).resolve().parent
SEED = 42


def make_cnn():
    model = Sequential([
        Conv2D(6, (5, 5), padding="same", activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(16, (5, 5), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(120, activation="relu"),
        Dense(10, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=SGD(learning_rate=0.01), metrics=["accuracy"])
    return model


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train_flat = x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0
    x_test_flat = x_test.reshape(len(x_test), -1).astype(np.float32) / 255.0

    # Cross-validation is confined to a reproducible, stratified sample of
    # the official training split. The official test split stays untouched.
    x_cv, _, y_cv, _ = train_test_split(
        x_train_flat, y_train, train_size=5000, stratify=y_train,
        random_state=SEED)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_models = {
        "Most frequent baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=100,
                                                 solver="lbfgs", tol=0.01)),
        "Random forest": RandomForestClassifier(
            n_estimators=80, max_depth=None, min_samples_leaf=1,
            n_jobs=-1, random_state=SEED),
    }
    results = {
        "seed": SEED,
        "cv_sample_size": int(len(y_cv)),
        "cv_folds": 5,
        "cross_validation": {},
        "test": {},
        "tuning": {},
    }
    for name, estimator in cv_models.items():
        scores = cross_val_score(estimator, x_cv, y_cv, cv=folds,
                                 scoring="accuracy", n_jobs=1)
        results["cross_validation"][name] = {
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std(ddof=1)),
            "fold_scores": [float(v) for v in scores],
        }
        print("CV", name, scores, flush=True)

    # Tune one hyperparameter for each learned baseline on training-only CV.
    logistic_grid = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=100, solver="lbfgs", tol=0.01)),
        {"logisticregression__C": [0.1, 1.0]}, cv=3,
        scoring="accuracy", n_jobs=1)
    logistic_grid.fit(x_cv, y_cv)
    results["tuning"]["logistic_regression"] = {
        "best_params": logistic_grid.best_params_,
        "best_cv_accuracy": float(logistic_grid.best_score_),
    }
    print("Logistic grid", results["tuning"]["logistic_regression"], flush=True)

    forest_grid = GridSearchCV(
        RandomForestClassifier(max_depth=None, min_samples_leaf=1,
                               n_jobs=-1, random_state=SEED),
        {"n_estimators": [40, 80]}, cv=3,
        scoring="accuracy", n_jobs=1)
    forest_grid.fit(x_cv, y_cv)
    results["tuning"]["random_forest"] = {
        "best_params": forest_grid.best_params_,
        "best_cv_accuracy": float(forest_grid.best_score_),
    }
    print("Forest grid", results["tuning"]["random_forest"], flush=True)

    # Fit final classical candidates on all 60,000 training examples; evaluate
    # exactly once on the official 10,000-example test set.
    final_models = dict(cv_models)
    final_models["Logistic regression"] = logistic_grid.best_estimator_
    final_models["Random forest"] = forest_grid.best_estimator_
    for name, estimator in final_models.items():
        estimator.fit(x_train_flat, y_train)
        pred = estimator.predict(x_test_flat)
        results["test"][name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
        }
        print("Test", name, results["test"][name], flush=True)

    keras_model = load_model(ROOT / "handwriting.h5", compile=False)
    p = keras_model.predict(x_test.astype(np.float32)[..., None] / 255.0,
                            batch_size=256, verbose=0)
    pred = p.argmax(axis=1)
    results["test"]["Saved Keras CNN"] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }
    # Rows are true labels and columns are predicted labels, ordered 0-9.
    # Keep the numeric matrix in the results file for MATLAB plotting.
    results["cnn_confusion_matrix"] = confusion_matrix(
        y_test, pred, labels=np.arange(10)
    ).tolist()
    print("Test saved CNN", results["test"]["Saved Keras CNN"], flush=True)

    # Training-size experiment. Reserve 6,000 examples from the official
    # training split for validation; do not use the test set in this experiment.
    x_pool, x_val, y_pool, y_val = train_test_split(
        x_train, y_train, test_size=6000, stratify=y_train,
        random_state=SEED)
    size_results = []
    for size in [6000, 15000, 30000, 54000]:
        model = make_cnn()
        model.fit(x_pool[:size, ..., None].astype(np.float32) / 255.0,
                  y_pool[:size], epochs=5, batch_size=128, verbose=0,
                  shuffle=True)
        train_score = model.evaluate(
            x_pool[:size, ..., None].astype(np.float32) / 255.0,
            y_pool[:size], verbose=0, return_dict=True)
        val_score = model.evaluate(
            x_val[..., None].astype(np.float32) / 255.0,
            y_val, verbose=0, return_dict=True)
        item = {"training_examples": size,
                "train_accuracy": float(train_score["accuracy"]),
                "validation_accuracy": float(val_score["accuracy"])}
        size_results.append(item)
        print("Training-size experiment", item, flush=True)
        del model

    results["training_size_experiment"] = {
        "validation_size": int(len(y_val)),
        "epochs": 5,
        "batch_size": 128,
        "rows": size_results,
    }
    output = ROOT / "mnist_experiment_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
