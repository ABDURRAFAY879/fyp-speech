# Importing Libraries
from collections import Counter

import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import librosa
import scikitplot as skplt
from spafe.features.lpc import lpc, lpcc
from spafe.features.rplp import rplp, plp

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
import joblib
import seaborn as sns
import features as fs
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

from dev_config import root
from dev_config import model_path


def mfcc_feature(audio, sample_rate):
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    return mfcc  # it returns a np.array with size (40,'n') where n is the number of audio frames.


def melspectrogram_feature(audio, sample_rate):
    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=2048)

    return melspectrogram  # it returns a np.array with size (128,'n') where n is the number of audio frames.


def poly_feature(audio, sample_rate):
    poly_features = librosa.feature.poly_features(y=audio, sr=sample_rate, n_fft=2048)

    return poly_features  # it returns a np.array with size (2,'n') where n is the number of audio frames.


def zero_crossing_rate_features(audio):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)

    return zero_crossing_rate  # it returns a np.array with size (1,'n') where n is the number of audio frames.


def normalize(data):
    data = (data - min(data)) / (max(data) - min(data))

    return data


def results(target_test, predicted_test, ModelName, labels):
    target_names = labels
    # print(classification_report(target_test, y_predd1, target_names=target_names))
    y_test = target_test
    preds = predicted_test
    rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(preds)), 2)))
    score = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    pearson_coef, p_value = stats.pearsonr(y_test, preds)

    print("root mean square:", rms)
    print("score:", score)
    print("mean absolute error:", mae)
    print("mean squared error:", mse)
    print("pearson_coef:", pearson_coef)
    print("p_value:", p_value)
    print("=======================================================================\n\n")
    skplt.metrics.plot_confusion_matrix(
        y_test,
        preds,
        figsize=(10, 6),
        title="Confusion matrix\n Deposite Category " + ModelName,
    )
    plt.xlim(-0.5, len(np.unique(y_test)) - 0.5)
    plt.ylim(len(np.unique(y_test)) - 0.5, -0.5)
    plt.savefig("cvroc.png")
    plt.show()


warnings.filterwarnings("ignore")

# setting the path where all file's folder are

Features_data = pd.DataFrame(columns=["features", "class"])

i = 0
sample_rate = 16000
no_of_samples = 300
MainFolder = "ayat"
labels = sorted(os.listdir(MainFolder))


# Loading the features in the dataframe
for label in labels:

    print(label)
    folders = os.path.join(root, label)
    items = sorted(os.listdir(folders))
    print('items', items)
    for item in items[:no_of_samples]:

        path = os.path.join(folders, item)

        # Convert .wave into array
        samples, sample_rate = librosa.load(path, sr=sample_rate)

        # Extract Feautures
        MFCC = mfcc_feature(samples, sample_rate)
        MSS = melspectrogram_feature(samples, sample_rate)
        poly = poly_feature(samples, sample_rate)
        ZCR = zero_crossing_rate_features(samples)

        # flatten an array
        MFCC = MFCC.flatten()
        MSS = MSS.flatten()
        poly = poly.flatten()
        ZCR = ZCR.flatten()

        # normalizing
        # MFCC = normalize(MFCC)

        features = np.concatenate((MFCC, MSS, poly, ZCR))

        # padding and trimming
        max_len = 6000

        pad_width = max_len - features.shape[0]
        if pad_width > 0:
            features = np.pad(features, pad_width=((0, pad_width)), mode="constant")

        features = features[:max_len]

        Features_data.loc[i] = [features, label]
        i += 1
np.set_printoptions(threshold=sys.maxsize)
feature = np.array(Features_data["features"].tolist())
target = Features_data.iloc[:, -1]
# converting labels into numeric
le = preprocessing.LabelEncoder()
target = le.fit_transform(target)
# features = preprocessing.MinMaxScaler().fit_transform(features)
feature_train, feature_test, target_train, target_test = train_test_split(
    feature, target
)
# Create a Gaussian Classifier
clff = RandomForestClassifier(n_estimators=800)
# Train the model using the training sets y_pred=clf.predict(X_test)
clff = clff.fit(feature_train, target_train)
y_predd1 = clff.predict(feature_test)
# Model Accuracy, how often is the classifier correct?
print("Random Forest Accuracy:", metrics.accuracy_score(target_test, y_predd1))
results(target_test, y_predd1, "Random Forest", labels)

target_names = labels

sns.heatmap(confusion_matrix(target_test, y_predd1), annot=True, cmap="Blues")
joblib.dump(clff, model_path + "model_3000.sav")
# Create a KNN Classifier
knn = KNeighborsClassifier()
# Train the model using the training sets y_pred=clf.predict(X_test)
knn = knn.fit(feature_train, target_train)
y_predd2 = knn.predict(feature_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy KNN:", metrics.accuracy_score(target_test, y_predd2))
results(target_test, y_predd2, "KNN", labels)
joblib.dump(knn, model_path + "model_knn.sav")
# training a linear SVM classifier
svm_model_linear = SVC(kernel="linear", C=1).fit(feature_train, target_train)
y_predd3 = svm_model_linear.predict(feature_test)
# model accuracy for X_test
accuracy = svm_model_linear.score(feature_test, target_test)
print("Accuracy SVM:", metrics.accuracy_score(target_test, y_predd3))
results(target_test, y_predd3, "SVM", labels)
joblib.dump(svm_model_linear, model_path + "model_svm.sav")
model1 = RandomForestClassifier()
model2 = KNeighborsClassifier()
model3 = LogisticRegression()
Voting = VotingClassifier(
    estimators=[("RF", model1), ("knn", model2), ("lr", model3)], voting="hard"
)
Voting.fit(feature_train, target_train)
vpredictions = Voting.predict(feature_test)
vscore = Voting.score(feature_test, target_test)
print("Voting Score", vscore)
results(target_test, vpredictions, "Voting Classifier", labels)
joblib.dump(Voting, model_path + "model_voting.sav")
