from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import time
import os
import urllib.request
import subprocess
import zipfile
import statistics
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

from experiment_utils import run_experiment, load_results, summarize_results, compare_methods, plot_metric_vs_missing_rate, plot_metric_bar_chart

#!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo


##
seed = 42
np.random.seed(seed)

# đổi số lượng ứng viên n_neighbors được thử khi tune ở đây, mọi hàm tune bên dưới dùng chung giá trị này
MAX_TUNING_NEIGHBORS = 15

# phương pháp truyền thống: impute -> classify named KNN-IC
#
def knnIC_approach(x_train_missing, x_test_missing, y_train, y_test, random_state=None):

  t0 = time.perf_counter()
  best_n_neighbors = find_best_n_neighbors(x_train_missing, y_train, random_state=random_state)
  tuning_time = time.perf_counter() - t0

  t1 = time.perf_counter()
  x_train_impute = impute_knn(x_train_missing, best_n_neighbors)
  x_test_impute = impute_knn(x_test_missing, best_n_neighbors)
  train_time = time.perf_counter() - t1

  t2 = time.perf_counter()
  accuracy = train_and_evaluate_knn(x_train_impute, y_train,
                         x_test_impute, y_test,
                         best_n_neighbors)
  predict_time = time.perf_counter() - t2

  return {"accuracy": accuracy, "n_neighbors": best_n_neighbors,
          "tuning_time_sec": tuning_time, "train_time_sec": train_time,
          "predict_time_sec": predict_time}
#

def train_and_evaluate_knn(x_train, y_train,
                           x_test, y_test,
                           n_neighbors):

  knn_classifier = KNNClassifier(n_neighbors=n_neighbors)
  knn_classifier.fit(x_train, y_train)
  y_predict = knn_classifier.predict(x_test)
  accuracy = accuracy_score(y_test, y_predict)
  return accuracy


def impute_knn(missing_data, number_of_neighbors):
  imputer = KNNImputer(n_neighbors=number_of_neighbors)
  imputed_data = imputer.fit_transform(missing_data)
  return imputed_data

def find_best_n_neighbors(x_train_missing, y_train, random_state=None):

  x_actual_train, x_validation, y_actual_train, y_validation = train_test_split(x_train_missing,
                                                                                y_train,
                                                                                test_size=0.2,
                                                                                random_state=random_state)
  best_accuracy = 0
  best_n_neighbors = 1

  for n_neighbors in range(1, MAX_TUNING_NEIGHBORS+1):
      x_actual_train_imputed = impute_knn(x_actual_train, n_neighbors)
      x_validation_imputed = impute_knn(x_validation, n_neighbors)

      current_accuracy = train_and_evaluate_knn(x_actual_train_imputed, y_actual_train,
                                        x_validation_imputed, y_validation,
                                        n_neighbors)


      if (current_accuracy > best_accuracy):
          best_accuracy = current_accuracy
          best_n_neighbors = n_neighbors

  return best_n_neighbors


# baseline non-KNN: impute bằng mean (không phụ thuộc n_neighbors) -> classify bằng KNN, named Mean-IC
def meanIC_approach(x_train_missing, x_test_missing, y_train, y_test, random_state=None):

  t0 = time.perf_counter()
  best_n_neighbors = find_best_n_neighbors_mean(x_train_missing, y_train, random_state=random_state)
  tuning_time = time.perf_counter() - t0

  t1 = time.perf_counter()
  x_train_impute = impute_mean(x_train_missing)
  x_test_impute = impute_mean(x_test_missing)
  train_time = time.perf_counter() - t1

  t2 = time.perf_counter()
  accuracy = train_and_evaluate_knn(x_train_impute, y_train,
                         x_test_impute, y_test,
                         best_n_neighbors)
  predict_time = time.perf_counter() - t2

  return {"accuracy": accuracy, "n_neighbors": best_n_neighbors,
          "tuning_time_sec": tuning_time, "train_time_sec": train_time,
          "predict_time_sec": predict_time}


def impute_mean(missing_data):
  imputer = SimpleImputer(strategy='mean')
  return imputer.fit_transform(missing_data)


def find_best_n_neighbors_mean(x_train_missing, y_train, random_state=None):

  x_actual_train, x_validation, y_actual_train, y_validation = train_test_split(x_train_missing,
                                                                                y_train,
                                                                                test_size=0.2,
                                                                                random_state=random_state)
  # mean imputation không phụ thuộc n_neighbors nên chỉ cần impute một lần
  x_actual_train_imputed = impute_mean(x_actual_train)
  x_validation_imputed = impute_mean(x_validation)

  best_accuracy = 0
  best_n_neighbors = 1

  for n_neighbors in range(1, MAX_TUNING_NEIGHBORS+1):
      current_accuracy = train_and_evaluate_knn(x_actual_train_imputed, y_actual_train,
                                        x_validation_imputed, y_validation,
                                        n_neighbors)

      if (current_accuracy > best_accuracy):
          best_accuracy = current_accuracy
          best_n_neighbors = n_neighbors

  return best_n_neighbors


# phương pháp trực tiếp: KNN với Euclidean distance chấp nhận missing values (không impute) named KNN-Direct
def knnDirect_approach(x_train_missing, x_test_missing, y_train, y_test, distance_func=None, random_state=None):

  t0 = time.perf_counter()
  best_n_neighbors = find_best_n_neighbors_direct(x_train_missing, y_train, distance_func, random_state=random_state)
  tuning_time = time.perf_counter() - t0

  train_time = 0.0  # không có bước tiền xử lý/impute

  t1 = time.perf_counter()
  accuracy = train_and_evaluate_knn_direct(x_train_missing, y_train,
                         x_test_missing, y_test,
                         best_n_neighbors, distance_func)
  predict_time = time.perf_counter() - t1

  return {"accuracy": accuracy, "n_neighbors": best_n_neighbors,
          "tuning_time_sec": tuning_time, "train_time_sec": train_time,
          "predict_time_sec": predict_time}


# biến thể dùng euclidean_na_distance_weighted, cùng interface để plug vào run_experiment
def knnDirectWeighted_approach(x_train_missing, x_test_missing, y_train, y_test, random_state=None):
  return knnDirect_approach(x_train_missing, x_test_missing, y_train, y_test,
                             distance_func=euclidean_na_distance_weighted, random_state=random_state)


# biến thể dùng manhattan_na_distance, cùng interface để plug vào run_experiment
def knnDirectManhattan_approach(x_train_missing, x_test_missing, y_train, y_test, random_state=None):
  return knnDirect_approach(x_train_missing, x_test_missing, y_train, y_test,
                             distance_func=manhattan_na_distance, random_state=random_state)


def train_and_evaluate_knn_direct(x_train, y_train,
                                  x_test, y_test,
                                  n_neighbors, distance_func=None):

  knn_classifier = KNNClassifier(n_neighbors=n_neighbors,
                                  distance_func=distance_func or euclidean_na_distance)
  knn_classifier.fit(x_train, y_train)
  y_predict = knn_classifier.predict(x_test)
  accuracy = accuracy_score(y_test, y_predict)
  return accuracy


def find_best_n_neighbors_direct(x_train_missing, y_train, distance_func=None, random_state=None):

  x_actual_train, x_validation, y_actual_train, y_validation = train_test_split(x_train_missing,
                                                                                y_train,
                                                                                test_size=0.2,
                                                                                random_state=random_state)
  best_accuracy = 0
  best_n_neighbors = 1

  for n_neighbors in range(1, MAX_TUNING_NEIGHBORS+1):
      current_accuracy = train_and_evaluate_knn_direct(x_actual_train, y_actual_train,
                                        x_validation, y_validation,
                                        n_neighbors, distance_func)

      if (current_accuracy > best_accuracy):
          best_accuracy = current_accuracy
          best_n_neighbors = n_neighbors

  return best_n_neighbors

#

# General code for two methods: KNNI_KNNC and KNN_Directly
class KNNClassifier:
    # if distance_func=None then this implement the KNN Classifer with Euclid distance

    def __init__(self, n_neighbors=3, distance_func=None):
        self.n_neighbors = n_neighbors
        self.distance_func = distance_func if distance_func is not None else self.euclidean_distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = np.array([self.distance_func(x, x_train) for x_train in self.X_train])
        # Sort by distance and return the indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = self.y_train[k_indices]

        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def get_params(self, deep=True):
        """
        Return a dictionary of the parameters of the classifier.

        :param deep: If True, will return a deep copy of the parameters.
        :return: A dictionary of parameter names mapped to their values.
        """
        return {"n_neighbors": self.n_neighbors}#, "distance_func": self.distance_func}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

def euclidean_na_distance(x, y): #Euclidean distance that accept input as missing values
    mask = ~np.isnan(x) & ~np.isnan(y)  # Only consider non-missing values
    return np.sqrt(np.sum((x[mask] - y[mask]) ** 2))

def euclidean_na_distance_weighted(x, y): #Euclidean distance scaled by valid-dimension count, like sklearn's nan_euclidean_distances
    mask = ~np.isnan(x) & ~np.isnan(y)  # Only consider non-missing values
    n_valid = mask.sum()
    if n_valid == 0:
        return np.inf
    n_features = len(x)
    return np.sqrt((n_features / n_valid) * np.sum((x[mask] - y[mask]) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def manhattan_na_distance(x, y): #Manhatan distance that accept input as missing values
    mask = ~np.isnan(x) & ~np.isnan(y)  # Only consider non-missing values
    return np.sum(np.abs(x[mask] - y[mask]))

# Generate missing value on given data
def generate_missing_values(original_data, missing_rate):
  data_shape = original_data.shape
  missing_id = np.random.randint(0,original_data.size,round(missing_rate*original_data.size))
  missing_data = original_data.flatten()
  missing_data[missing_id] = np.nan
  return missing_data.reshape(data_shape)

# Bước : tạo dữ liệu
def get_Toxicity_data():

  # fetch dataset
  toxicity = fetch_ucirepo(id=728)

  # data (as pandas dataframes)
  x_features = toxicity.data.features
  y_labels = toxicity.data.targets
  # # metadata
  # print(toxicity.metadata)
  #
  # # variable information
  # print(toxicity.variables)


  #X = X.drop("name", axis = 'columns')
  x_features = pd.DataFrame.to_numpy(x_features)
  x_features = x_features[:,2:].astype(np.float32)
  y_labels = pd.DataFrame.to_numpy(y_labels)
  y_labels = np.reshape(y_labels, (171,))

  label_encoder = LabelEncoder()
  y_numbered_labels = label_encoder.fit_transform(y_labels)
  return x_features, y_numbered_labels

def get_Darwin_data():

  # fetch dataset
  darwin = fetch_ucirepo(id=732)

  # data (as pandas dataframes)
  x_features = darwin.data.features
  y_labels = darwin.data.targets

  # variable information
  print(darwin.variables)

  #X = X.drop("name", axis = 'columns')
  x_features = pd.DataFrame.to_numpy(x_features)
  x_features = x_features[:,2:].astype(np.float32)
  y_labels = pd.DataFrame.to_numpy(y_labels)
  y_labels = np.reshape(y_labels, (174,))

  label_encoder = LabelEncoder()
  y_numbered_labels = label_encoder.fit_transform(y_labels)

  return x_features, y_numbered_labels

def get_Micromass_data():
  if not (os.path.exists('mixed_spectra_metadata.csv') and os.path.exists('mixed_spectra_matrix.csv')):
      zip_path = 'micromass.zip'
      if not os.path.exists(zip_path):
          urllib.request.urlretrieve(
              'https://archive.ics.uci.edu/ml/machine-learning-databases/00253/micromass.zip',
              zip_path)
      with zipfile.ZipFile(zip_path) as z:
          z.extractall('.')

  label_data = pd.read_csv('mixed_spectra_metadata.csv', sep = ';')
  label = label_data[['Mixture_Label']]
  label = label.to_numpy()
  print(len(np.unique(label)))
  label_encoder = LabelEncoder()
  y = label_encoder.fit_transform(label)

  df = pd.read_csv('mixed_spectra_matrix.csv', sep = ';', header = None)
  print(df.shape)
  print(df.head())
  X = df.to_numpy()
  var_vec = np.array([np.var(X[:,i]) for i in range(X.shape[1])])
  id = np.where(var_vec > 1e-5)
  X = X[:,id].reshape((len(X),-1))

  x_features = X
  y_numbered_labels = np.asarray(y)
  return x_features, y_numbered_labels

def get_Parkinson_data():
    rar_path = 'pd_speech_features.rar'
    csv_path = 'pd_speech_features.csv'

    if not os.path.exists(csv_path):
        if not os.path.exists(rar_path):
            urllib.request.urlretrieve(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/00470/pd_speech_features.rar',
                rar_path)
        try:
            subprocess.run(['tar', '-xf', rar_path], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Không giải nén được pd_speech_features.rar bằng 'tar'. "
                "Hãy tự giải nén file này (vd: unrar x pd_speech_features.rar) "
                "rồi chạy lại."
            ) from e

    # cột 0 là 'id' (lặp lại theo bệnh nhân, không phải feature), cột cuối là 'class'
    data = pd.read_csv(csv_path, header=[0, 1]).to_numpy()
    x_features = data[:, 1:-1].astype(np.float32)
    y_labels = data[:, -1].astype(int)
    return x_features, y_labels

def get_Parkinson2clc_data():
    data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data',
                       sep=",")
    data = data.drop(['name'], axis=1)
    X, y = data.drop(['status'], axis=1), data['status']

    x_features = X.to_numpy().astype(np.float32)
    y_labels = np.asarray(y).astype(int)
    return x_features, y_labels

# chuẩn hóa dữ liệu
def normalize_data(data):
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    normalized_data = (data - min_values) / (max_values - min_values)
    return normalized_data
#
def standardize_data(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data

def get_split(trial_seed):
    return train_test_split(features_x, labels_y, test_size=0.2, random_state=trial_seed)

methods = {"KNN-IC": knnIC_approach,
           "Mean-IC": meanIC_approach,
           "KNN-Direct": knnDirect_approach,
           "KNN-Direct-Weighted": knnDirectWeighted_approach,
           "KNN-Direct-Manhattan": knnDirectManhattan_approach}

DATASETS = {
    "Toxicity": get_Toxicity_data,
    "Darwin": get_Darwin_data,
    "Micromass": get_Micromass_data,
    "Parkinson": get_Parkinson_data,
    "Parkinson2clc": get_Parkinson2clc_data,
}

DATASET_DOMAINS = {
    "Toxicity": "Toxicology / QSAR",
    "Darwin": "Alzheimer's handwriting",
    "Micromass": "MALDI-TOF bacteria proteomics",
    "Parkinson": "Parkinson speech signal",
    "Parkinson2clc": "Parkinson voice signal",
}


def dataset_summary_table():
  rows = []
  for name, loader in DATASETS.items():
      x, y = loader()
      class_counts = np.bincount(y)
      rows.append({
          "Dataset": name,
          "Domain": DATASET_DOMAINS[name],
          "Samples": x.shape[0],
          "Features": x.shape[1],
          "Classes": len(class_counts),
          "Class distribution": "/".join(map(str, class_counts.tolist())),
      })
  return pd.DataFrame(rows)

# đổi tên dataset ở đây, mọi đường dẫn/kết quả bên dưới sẽ tự cập nhật theo
DATASET_NAME = "Parkinson"


if __name__ == '__main__':
    results_csv_path = f"results/{DATASET_NAME.lower()}_results.csv"
    accuracy_plot_path = f"results/{DATASET_NAME.lower()}_accuracy_plot.png"
    time_plot_path = f"results/{DATASET_NAME.lower()}_time_plot.png"

    # step 1: lấy data
    features_x, labels_y = DATASETS[DATASET_NAME]()

    # step 2: chuẩn hóa data
    features_x = standardize_data(features_x)

    # step 3: chia tập dữ liệu
    x_train, x_test, y_train, y_test = train_test_split(features_x, labels_y, test_size=0.2, random_state=seed)

    # step 4: chạy thí nghiệm đầy đủ (nhiều missing rate x nhiều trial) để lấy kết quả cho paper
    run_experiment(methods, get_split,
                   missing_rates=[0.1, 0.2, 0.3, 0.4, 0.5],
                   n_trials=40, dataset_name=DATASET_NAME,
                   output_path=results_csv_path)

    df = load_results(results_csv_path)
    print(summarize_results(df))  # bảng mean±std theo missing_rate
    print(compare_methods(df, "KNN-IC", "KNN-Direct", missing_rate=0.3))  # p-value
    print(compare_methods(df, "KNN-Direct", "KNN-Direct-Weighted", missing_rate=0.3))  # p-value
    plot_metric_vs_missing_rate(df, metric="accuracy", output_path=accuracy_plot_path)
    plot_metric_bar_chart(df, metric="total_time_sec", output_path=time_plot_path)

