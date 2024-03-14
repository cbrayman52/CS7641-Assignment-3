import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def get_datasets():
    #############
    # Dataset 1 #
    #############
    wine_quality = pd.read_csv(r'Dataset/wine_quality.csv')

    # Clean data
    wine_quality = wine_quality.drop_duplicates()   # Remove rows that contain duplicate data
    wine_quality = wine_quality.dropna()            # Remove rows that contain missing values

    # Manually handle outliers
    lower_limits = {'fixed acidity': 4.0,  'volatile acidity': 0.1, 'citric acid': 0.0, 'residual sugar': 0.0,  'chlorides': 0.0, 'free sulfur dioxide': 0,  'total sulfur dioxide': 0,   'density': 0.9, 'ph': 3.0, 'sulphates': 0.3, 'alcohol': 8.0}
    upper_limits = {'fixed acidity': 15.0, 'volatile acidity': 1.1, 'citric acid': 0.8, 'residual sugar': 10.0, 'chlorides': 0.3, 'free sulfur dioxide': 60, 'total sulfur dioxide': 170, 'density': 1.1, 'ph': 4.0, 'sulphates': 1.5, 'alcohol': 13.6}
    for column in wine_quality.columns:
            lower_limit = lower_limits.get(column, None)
            upper_limit = upper_limits.get(column, None)        
            if lower_limit is not None and upper_limit is not None:
                wine_quality[column] = np.where((wine_quality[column] < lower_limit) | (wine_quality[column] > upper_limit), wine_quality[column].mean(), wine_quality[column])

    # Alter dataset to be binary classificaion
    bins = (2, 6.5, 8)
    group_names = ['bad', 'good']
    wine_quality['quality'] = pd.cut(wine_quality['quality'], bins = bins, labels = group_names)

    # Assign labels to quality variable
    label_quality = LabelEncoder()

    # Bad becomes 0 and good becomes 1 
    wine_quality['quality'] = label_quality.fit_transform(wine_quality['quality'])
    value_count = wine_quality['quality'].value_counts()

    # Seperate the dataset as response variable and feature variabes
    x = wine_quality.drop('quality', axis = 1)
    y = wine_quality['quality']

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=None)

    # Preprocess the data (Standard Scaling)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    y_train = y_train.values
    y_test = y_test.values

    dataset1 = {"x_train": x_train,
                "x_train_scaled": x_train_scaled,
                "x_test": x_test,
                "x_test_scaled": x_test_scaled, 
                "y_train": y_train,
                "y_test": y_test}


    #############
    # Dataset 2 #
    #############
    x2, y2 = make_classification(n_samples=1500,
                                n_features=11,
                                n_informative=8,
                                n_redundant=3,
                                n_classes=5,
                                flip_y=0.1,
                                class_sep=2,
                                random_state=None)

    # Add noise to the dataset
    x2 += np.random.normal(scale=0.5, size=x2.shape)

    # Introduce outliers
    outliers_indices = np.random.choice(range(len(x2)), size=50, replace=False)
    x2[outliers_indices] += np.random.normal(loc=10, scale=5, size=x2[outliers_indices].shape)

    # Convert to a pandas DataFrame
    columns = [f'feature_{i}' for i in range(x2.shape[1])]
    x2 = pd.DataFrame(x2, columns=columns)
    y2 = pd.DataFrame(y2, columns=['target']).squeeze()

    # Split the data into training and testing sets
    x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.25)

    # Scale features
    scaler = StandardScaler()
    x2_train_scaled = x2_train.copy()
    x2_train_scaled[:] = scaler.fit_transform(x2_train)
    x2_test_scaled = x2_test.copy()
    x2_test_scaled[:] = scaler.transform(x2_test)

    dataset2 = {"x2_train": x2_train,
                "x2_train_scaled": x2_train_scaled,
                "x2_test": x2_test,
                "x2_test_scaled": x2_test_scaled, 
                "y2_train": y2_train,
                "y2_test": y2_test}

    return dataset1, dataset2