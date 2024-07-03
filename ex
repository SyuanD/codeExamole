import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample imbalanced dataset
np.random.seed(42)
X, y = np.random.rand(1000, 10), np.random.choice([0, 1, 2, 3, 4], size=1000, p=[0.7, 0.1, 0.1, 0.05, 0.05])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define class weights (example weights, adjust as needed)
class_weights = {0: 1, 1: 5, 2: 5, 3: 10, 4: 10}

# Custom weighted multi-class loss function
def weighted_log_loss(y_true, y_pred):
    y_pred = y_pred.reshape(len(y_true), -1)  # Reshape to (num_samples, num_classes)
    y_true_one_hot = np.eye(y_pred.shape[1])[y_true.astype(int)]  # One-hot encode true labels
    weights = np.array([class_weights[int(label)] for label in y_true])
    softmax = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)  # Softmax function
    grad = - (y_true_one_hot - softmax) * weights[:, np.newaxis]
    hess = softmax * (1 - softmax) * weights[:, np.newaxis]
    return grad.ravel(), hess.ravel()

# Custom evaluation function (using accuracy for simplicity)
def custom_eval(y_true, y_pred):
    y_pred = y_pred.reshape(len(y_true), -1).argmax(axis=1)
    return 'accuracy', accuracy_score(y_true, y_pred), True

# Create dataset for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'None',
    'boosting_type': 'gbdt',
    'verbose': -1
}

# Train the model
gbm = lgb.train(params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                fobj=weighted_log_loss,  # Custom loss function
                feval=custom_eval)  # Custom evaluation function

# Make predictions
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_labels = y_pred.argmax(axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_labels)
print(f"Accuracy: {accuracy}")
