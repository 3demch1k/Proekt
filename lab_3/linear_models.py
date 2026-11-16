import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def _compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = np.asarray(data.X_train, dtype=float)
    y_train = np.asarray(data.y_train, dtype=float)
    X_test = np.asarray(data.X_test, dtype=float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = getattr(data, "learning_rate", 0.05) or 0.05
    epochs = getattr(data, "epochs", None) or getattr(data, "n_iterations", 1000) or 1000

    n_samples, n_features = X_train_scaled.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    mse_history: list[float] = []

    for _ in range(epochs):
        y_pred = X_train_scaled @ weights + bias
        error = y_pred - y_train

        mse_val = float(np.mean(error ** 2))
        mse_history.append(mse_val)

        dw = (2.0 / n_samples) * (X_train_scaled.T @ error)
        db = (2.0 / n_samples) * float(np.sum(error))

        weights -= lr * dw
        bias -= lr * db

    predictions = X_test_scaled @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        mse_history=mse_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X_train = np.asarray(data.X_train, dtype=float)
    y_train = np.asarray(data.y_train, dtype=float)
    X_test = np.asarray(data.X_test, dtype=float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = getattr(data, "learning_rate", 0.1) or 0.1
    epochs = getattr(data, "epochs", None) or getattr(data, "n_iterations", 1000) or 1000
    threshold = getattr(data, "threshold", 0.5) or 0.5

    n_samples, n_features = X_train_scaled.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    loss_history: list[float] = []
    eps = 1e-15

    for _ in range(epochs):
        linear_out = X_train_scaled @ weights + bias
        probs = _sigmoid(linear_out)

        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        bce = -np.mean(
            y_train * np.log(probs_clipped) + (1.0 - y_train) * np.log(1.0 - probs_clipped)
        )
        loss_history.append(float(bce))

        diff = probs - y_train
        dw = (1.0 / n_samples) * (X_train_scaled.T @ diff)
        db = (1.0 / n_samples) * float(np.sum(diff))

        weights -= lr * dw
        bias -= lr * db

    test_probabilities = _sigmoid(X_test_scaled @ weights + bias)
    test_predictions = (test_probabilities >= threshold).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=test_predictions,
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    X_train = np.asarray(data.X_train, dtype=float)
    y_train = np.asarray(data.y_train, dtype=int)
    X_test = np.asarray(data.X_test, dtype=float)
    y_test = np.asarray(data.y_test, dtype=int)

    own_res = train_logistic_classifier(data)
    own_metrics = _compute_classification_metrics(
        y_test, own_res.predictions, own_res.probabilities
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_preds = lr_model.predict(X_test_scaled)
    lr_metrics = _compute_classification_metrics(y_test, lr_preds, lr_probs)

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    dt_probs = dt_model.predict_proba(X_test_scaled)[:, 1]
    dt_preds = dt_model.predict(X_test_scaled)
    dt_metrics = _compute_classification_metrics(y_test, dt_preds, dt_probs)

    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_scaled, y_train)
    knn_probs = knn_model.predict_proba(X_test_scaled)[:, 1]
    knn_preds = knn_model.predict(X_test_scaled)
    knn_metrics = _compute_classification_metrics(y_test, knn_preds, knn_probs)

    return BaselineComparisonResult(
        own_logistic=own_metrics,
        logistic_regression=lr_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics,
    )