import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split


# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the images for the machine learning models
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# Normalize the pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Optionally combine train and test sets for k-fold cross-validation
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)


# Split the Dataset
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Feature Extraction using PCA and LDA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Perform PCA
pca = PCA(n_components=0.95)  # retain 95% variance
X_pca = pca.fit_transform(X)

# Perform LDA
lda = LDA()
X_lda = lda.fit_transform(X, y)

#Train Classifiers and Evaluate Performance
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    # AUC is not applicable for multi-class directly, it requires one-vs-all approach
    # auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    return accuracy, f1, precision, recall

models = {
    'SVM_linear': SVC(kernel='linear'),
    'SVM_rbf': SVC(kernel='rbf'),
    'kNN': KNeighborsClassifier(),
    'Naive_Bayes': GaussianNB(),
    'Random_Forest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

results = {}

for name, model in models.items():
    print(f"Evaluating {name}...")
    for train_index, test_index in kf.split(X_pca):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)

print(results)

#Ablation Studies
# Example: Varying number of folds
kf_3 = KFold(n_splits=3, shuffle=True, random_state=42)
kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)

# Evaluate with different k-folds
for name, model in models.items():
    print(f"Evaluating {name} with 3-fold cross-validation...")
    for train_index, test_index in kf_3.split(X_pca):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]
        results[f"{name}_3_fold"] = evaluate_model(model, X_train, X_test, y_train, y_test)

for name, model in models.items():
    print(f"Evaluating {name} with 10-fold cross-validation...")
    for train_index, test_index in kf_10.split(X_pca):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]
        results[f"{name}_10_fold"] = evaluate_model(model, X_train, X_test, y_train, y_test)

#Predict on New Data
from PIL import Image

# Load and preprocess your own image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape(1, -1)
    img = img.astype('float32') / 255.0
    return img

my_digit = preprocess_image('path_to_your_digit_image.png')

# Use the best performing model (for example, Random Forest)
best_model = RandomForestClassifier()
best_model.fit(X_pca, y)

my_digit_pca = pca.transform(my_digit)
predicted_digit = best_model.predict(my_digit_pca)
print(f"Predicted digit: {predicted_digit[0]}")

