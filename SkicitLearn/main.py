import numpy as np
import pandas as pd
import mglearn
import faces as faces


from sklearn import datasets, __all__, svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
from sklearn.datasets import make_classification
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report,mean_squared_error, r2_score, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC, SVC

# PhanTriDuy_somay31
iris = datasets.load_iris()
digits = datasets.load_digits()


def Cau1():
    iris = datasets.load_iris()
    print(iris.data)


def Cau2():
    iris = pd.read_csv('Iris.csv')
    print(iris['Species'].value_counts())


def Cau3():


    # Load the iris dataset
    iris = sns.load_dataset('iris')

    # Get unique species
    species = iris['species'].unique()

    # Define colors for each species
    colors = {'setosa': 'blue', 'versicolor': 'green', 'virginica': 'red'}

    # Create a new figure
    fig, ax = plt.subplots()

    # Loop through each species
    for s in species:
        # Filter data for current species
        df = iris[iris['species'] == s]

        # Plot data with current species color
        ax.scatter(df['sepal_length'], df['sepal_width'], color=colors[s], label=s)

    # Set plot title and labels
    ax.set_title('Sepal Length vs Sepal Width by Species', fontsize=14)
    ax.set_xlabel('Sepal Length (cm)', fontsize=14)
    ax.set_ylabel('Sepal Width (cm)', fontsize=14)

    # Set axis limits
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)

    # Set tick marks
    ax.set_xticks(np.arange(0, 8.1, 1))
    ax.set_yticks(np.arange(0, 4.1, 1))

    # Add legend
    ax.legend(loc='upper left')

    # Remove whitespace
    sns.despine(ax=ax, offset=2, trim=True)

    # Display the plot
    plt.show()

def Cau4():
    # Example Data
    np.random.seed(42)
    X = np.random.rand(100, 30)  # 100 samples with 30 features

    # Step 1: Standardize the Data
    X_std = (X - np.mean(X)) / np.std(X)

    # Step 2-5: PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_std)

    # Plot the data in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c='r')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.show()

def Cau5():
    np.random.seed(42)
    X = np.random.rand(150, 30)  # 150 samples with 30 features
    y = np.random.randint(0, 2, 150)  # 150 samples with binary labels

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)

    # Apply PCA to reduce the dimensionality of the data to 3
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train a k-NN classifier with k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_pca, y_train)

    # Predict the labels of the test set
    y_pred = knn.predict(X_test_pca)

    # Print the classification report
    print(classification_report(y_test, y_pred))


def Cau6():
    np.random.seed(42)
    X = np.random.rand(150, 30)  # 150 samples with 30 features
    y = np.random.randint(0, 2, 150)  # 150 samples with binary labels

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)

    # Apply PCA to reduce the dimensionality of the data to 3
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train a k-NN classifier with k=5
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)

    # Predict the labels of the test set
    y_pred = knn.predict(X_test_pca)

    # Evaluate the model with accuracy, precision, and recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)


def Cau7():
    # Example Data
    np.random.seed(42)
    X = np.random.rand(150, 30)  # 150 samples with 30 features
    y = np.random.randint(0, 2, 150)  # 150 samples with binary labels

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)

    # Apply PCA to reduce the dimensionality of the data to 3
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train a k-NN classifier with k=5
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)

    # Predict the labels of the test set
    y_pred = knn.predict(X_test_pca)

    # Compare the predicted labels with the actual labels
    print("Predicted labels:", y_pred)
    print("Actual labels:", y_test)

    # Evaluate the model with accuracy, precision, and recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)


def Cau8():
    iris = load_iris()
    X = iris.data[:, :2]  # Sepal length and width only
    y = iris.target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a K-nearest neighbor classifier with k=5
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Plot the decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')

    # Plot the training and test data points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', s=50, label='Training data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', s=50, label='Test data')

    # Add legend and labels
    plt.legend()
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('K-nearest neighbor decision boundaries (k=5)')
    plt.show()



def Cau9():
    # Load the diabetes dataset
    diabetes = load_diabetes()

    # Take only one feature and one label from the dataset
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    diabetes_X_train = diabetes_X[:-30]
    diabetes_X_test = diabetes_X[-30:]
    diabetes_y_train = diabetes.target[:-30]
    diabetes_y_test = diabetes.target[-30:]

    # Train a linear regression model
    model = LinearRegression()
    model.fit(diabetes_X_train, diabetes_y_train)
    diabetes_y_predicted = model.predict(diabetes_X_test)

    # Print the mean squared error
    print("Mean squared error is:   ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
    print("Weights:   ", model.coef_)
    print("Intercept:   ", model.intercept_)

    # Plot the graph
    plt.scatter(diabetes_X_test, diabetes_y_test)
    plt.plot(diabetes_X_test, diabetes_y_predicted)
    plt.show()



def Cau10():
    # Load the diabetes dataset
    diabetes_df = pd.read_csv('diabetes.csv')

    # Split the dataset into training and test sets
    train_df = diabetes_df.iloc[:422]
    test_df = diabetes_df.iloc[422:]

    # Print the number of rows in each dataset
    print("Number of rows in training set:", train_df.shape[0])
    print("Number of rows in test set:", test_df.shape[0])



def Cau11():
    diabetes_df = pd.read_csv('diabetes.csv')

    # Split the dataset into training and test sets
    X_train = diabetes_df.iloc[:422, :].values
    y_train = diabetes_df.iloc[:422, 8].values
    X_test = diabetes_df.iloc[422:, :].values
    y_test = diabetes_df.iloc[422:, 8].values

    # Apply linear regression to the training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predict the test set results
    y_pred = regressor.predict(X_test)

    # Print the predicted values
    print(y_pred)



def Cau12():
    # Load the diabetes dataset
    diabetes_df = pd.read_csv('diabetes.csv')

    # Break the dataset into a training dataset (composed of the first 422 patients) and a test set (the last 20 patients)
    X_train = diabetes_df.iloc[:422, :].values
    y_train = diabetes_df.iloc[:422, 8].values
    X_test = diabetes_df.iloc[422:, :].values
    y_test = diabetes_df.iloc[422:, 8].values

    # Apply linear regression to the training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Get the beta coefficients
    beta_coefs = regressor.coef_

    # Print the beta coefficients
    print(beta_coefs)



def Cau13():
    # Load the diabetes dataset
    diabetes_df = pd.read_csv('diabetes.csv')

    # Break the dataset into a training dataset (composed of the first 422 patients) and a test set (the last 20 patients)
    X_train = diabetes_df.iloc[:422, :].values
    y_train = diabetes_df.iloc[:422, 8].values
    X_test = diabetes_df.iloc[422:, :].values
    y_test = diabetes_df.iloc[422:, 8].values

    # Apply linear regression to the training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predict the target values for the test set
    y_pred = regressor.predict(X_test)

    # Print the predicted target values
    print(y_pred)


def Cau14():
    # Load the diabetes dataset
    diabetes_df = pd.read_csv('diabetes.csv')

    # Break the dataset into a training dataset (composed of the first 422 patients) and a test set (the last 20 patients)
    X_train = diabetes_df.iloc[:422, 0].values.reshape(-1, 1)
    y_train = diabetes_df.iloc[:422, 8].values
    X_test = diabetes_df.iloc[422:, 0].values.reshape(-1, 1)
    y_test = diabetes_df.iloc[422:, 8].values

    # Apply linear regression to the training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predict the target values for the test set
    y_pred = regressor.predict(X_test)

    # Print the predicted target values
    print(y_pred)

    # Calculate the evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    r2_adj = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    print("Mean Squared Error: ", mse)
    print("Root Mean Squared Error: ", rmse)
    print("R-squared: ", r2)
    print("Adjusted R-squared: ", r2_adj)



def Cau15():
    # Load the diabetes dataset
    diabetes_df = pd.read_csv('diabetes.csv')

    # Break the dataset into a training dataset (composed of the first 422 patients) and a test set (the last 20 patients)
    X_train = diabetes_df.iloc[:422, 0].values.reshape(-1, 1)
    y_train = diabetes_df.iloc[:422, 8].values
    X_test = diabetes_df.iloc[422:, 0].values.reshape(-1, 1)
    y_test = diabetes_df.iloc[422:, 8].values

    # Apply linear regression to the training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predict the target values for the test set
    y_pred = regressor.predict(X_test)

    # Print the predicted target values
    print(y_pred)

    # Calculate the evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    r2_adj = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    print("Mean Squared Error: ", mse)
    print("Root Mean Squared Error: ", rmse)
    print("R-squared: ", r2)
    print("Adjusted R-squared: ", r2_adj)
def Cau16():
    # Load the diabetes dataset
    data = pd.read_csv("diabetes.csv")

    # Define the input features and target variable
    X = data.drop("target", axis=1)
    y = data["target"]

    # Define the list of physiological factors
    factors = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

    # Create a linear regression model for each factor
    models = []
    for factor in factors:
        model = LinearRegression()
        model.fit(X[[factor]], y)
        models.append(model)

    # Plot the results for each factor
    for i, factor in enumerate(factors):
        plt.figure()
        plt.scatter(X[factor], y)
        plt.plot(X[factor], models[i].predict(X[[factor]]), color="red")
        plt.title(f"Linear Regression for {factor}")
        plt.xlabel(factor)
        plt.ylabel("target")
        plt.show()



def Cau17():
    # Load the breast cancer dataset
    data = load_breast_cancer()

    # Print the keys of the dictionary
    print(data.keys())


def Cau18():
    cancer = load_breast_cancer()
    cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    cancer_df['target'] = cancer.target

    print("Shape of the data: ", cancer_df.shape)
    benign_malignant_counts = cancer_df['target'].value_counts()
    print("Number of benign tumors: ", benign_malignant_counts[0])
    print("Number of malignant tumors: ", benign_malignant_counts[1])

def Cau19():
    # Load the breast cancer dataset
    cancer = load_breast_cancer()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.3, random_state=42)

    # Initialize an array to store the accuracy scores
    scores = np.zeros(10)

    # Loop over different numbers of neighbors
    for i in range(1, 11):
        # Create a KNN classifier with i neighbors
        knn = KNeighborsClassifier(n_neighbors=i)

        # Fit the classifier to the training set
        knn.fit(X_train, y_train)

        # Predict the labels of the test set
        y_pred = knn.predict(X_test)

        # Calculate the accuracy score
        score = accuracy_score(y_test, y_pred)

        # Store the accuracy score
        scores[i - 1] = score

    # Plot the accuracy scores as a line chart
    plt.plot(range(1, 11), scores)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy score')
    plt.title('KNN classifier performance on breast cancer dataset')
    plt.show()


def Cau20():
    # Generate a binary classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

    # Create a Logistic Regression model
    log_reg = LogisticRegression()

    # Create a Linear SVC model
    lin_svc = LinearSVC()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit both models to the training set
    log_reg.fit(X_train, y_train)
    lin_svc.fit(X_train, y_train)

    # Evaluate both models on the test set
    y_pred_log_reg = log_reg.predict(X_test)
    y_pred_lin_svc = lin_svc.predict(X_test)

    # Print the accuracy scores
    print("Logistic Regression accuracy:", accuracy_score(y_test, y_pred_log_reg))
    print("Linear SVC accuracy:", accuracy_score(y_test, y_pred_lin_svc))

def Cau21():
   faces = fetch_olivetti_faces()
   print(faces.DESCR)

def Cau22():
    faces = fetch_olivetti_faces()
    # Accessing the images
    images = faces.images
    print("Shape of images array:", images.shape)  # This will give you the shape of the array containing the images

    # Accessing the flattened data
    data = faces.data
    print("Shape of data array:", data.shape)  # This will give you the shape of the flattened data array

    # Accessing the target labels
    target = faces.target
    print("Shape of target array:", target.shape)  # This will give you the shape of the array containing the labels
    print("Unique labels:", np.unique(target))  # This will give you the unique labels in the target array

def Cau23(n_faces):
    faces = fetch_olivetti_faces()
    n_rows, n_cols = int(np.sqrt(n_faces)), int(np.sqrt(n_faces))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    for i in range(n_faces):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].imshow(faces.images[i], cmap="gray")
        axes[row, col].axis("off")
    plt.show()



def Cau24():
    X, y = mglearn.datasets.make_forge()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # Create an SVC object with a linear kernel
    svm_linear = SVC(kernel='linear')

    # Fit the model with your data and labels
    svm_linear.fit(X_train, y_train)

    # Predict the labels of your test data
    y_pred_linear = svm_linear.predict(X_test)

    # Evaluate the performance of the linear SVM
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    print(f"Accuracy (linear SVM): {accuracy_linear}")


def Cau25():
    X, y = mglearn.datasets.make_forge()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    # Create an SVC object with a linear kernel
    svm_linear = SVC(kernel='linear')

    # Fit the model with your data and labels
    svm_linear.fit(X_train, y_train)

    # Predict the labels of your test data
    y_pred_linear = svm_linear.predict(X_test)

    # Evaluate the performance of the linear SVM
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    print(f"Accuracy (linear SVM): {accuracy_linear}")


def k_fold_cross_validation(X, y, k=5, model=None):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if model is not None:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
        else:
            # Fit any model here and replace this with the appropriate scoring metric
            score = 0

    scores.append(score)

    return np.mean(scores)

def Cau26():
    X, y = mglearn.datasets.make_forge()
    # Create an SVC object with a linear kernel
    svm_linear = SVC(kernel='linear')

    # Perform k-fold cross-validation
    k_fold_accuracy = k_fold_cross_validation(X, y, k=5, model=svm_linear)
    print(f"K-fold cross-validation accuracy (linear SVM): {k_fold_accuracy}")


def train_and_evaluate_svm(x_train, y_train, x_test, y_test):
    # Create an SVM object with a linear kernel
    svm_model = SVC(kernel='linear')

    # Train the model with the training data
    svm_model.fit(x_train, y_train)

    # Predict the labels of the test data
    y_pred = svm_model.predict(x_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    print(f"The model is {accuracy * 100}% accurate")

    return svm_model

def Cau27():
    X, y = mglearn.datasets.make_forge()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_model = train_and_evaluate_svm(x_train, y_train, x_test, y_test)
    svm_model = SVC(kernel='linear')
    svm_model.fit(x_train, y_train)

    # Print the SVM model
    print(svm_model)

def Cau28():
    clf = LogisticRegression()

    # Generate a random dataset
    X, y = mglearn.datasets.make_forge()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train the classifier on the training set
    clf.fit(X_train, y_train)

    # Evaluate the performance of the classifier
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')


    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

def has_glasses(glasses, faces):
    target = [0] * len(faces)
    for start1, end1 in glasses:
        for i, (start2, end2) in enumerate(faces):
            if start1 <= start2 <= end1 or start1 <= end2 <= end1:
                target[i] = 1
    return target

def Cau29(glasses, y):
    glasses = [(10, 19), (30, 32), (37, 38), (50, 59), (63, 64), (69, 69), (120, 121), (124, 129), (130, 139),
               (160, 161), (164, 169), (180, 182), (185, 185), (189, 189), (190, 192), (194, 194), (196, 199),
               (260, 269), (270, 279), (300, 309), (330, 339), (358, 359), (360, 369)]
    faces = [(15, 17), (20, 22), (35, 36), (55, 56), (65, 66), (125, 126), (135, 136), (165, 166), (183, 184),
             (191, 193), (265, 266), (275, 276), (305, 306), (335, 336), (365, 366)]

    target = has_glasses(glasses, faces)
    print(target)

def Cau30():
    def Cau29():
        glasses = [(10, 19), (30, 32), (37, 38), (50, 59), (63, 64), (69, 69), (120, 121), (124, 129), (130, 139),
                   (160, 161), (164, 169), (180, 182), (185, 185), (189, 189), (190, 192), (194, 194), (196, 199),
                   (260, 269), (270, 279), (300, 309), (330, 339), (358, 359), (360, 369)]

    faces = [(15, 17), (20, 22), (35, 36), (55, 56), (65, 66), (125, 126), (135, 136), (165, 166), (183, 184),
             (191, 193), (265, 266), (275, 276), (305, 306), (335, 336), (365, 366)]

    target = has_glasses(glasses, faces)
    print(target)

    def Cau30():
        glasses = [(10, 19), (30, 32), (37, 38), (50, 59), (63, 64), (69, 69), (120, 121), (124, 129), (130, 139),
                   (160, 161), (164, 169), (180, 182), (185, 185), (189, 189), (190, 192), (194, 194), (196, 199),
                   (260, 269), (270, 279), (300, 309), (330, 339), (358, 359), (360, 369)]

    faces = [(15, 17), (20, 22), (35, 36), (55, 56), (65, 66), (125, 126), (135, 136), (165, 166), (183, 184),
             (191, 193), (265, 266), (275, 276), (305, 306), (335, 336), (365, 366)]

    # Create random features for each sample in `faces`
    X = np.random.rand(len(faces), 10)

    # Create a new SVC classifier
    clf = svm.SVC()
    target = has_glasses(glasses, faces)

    # Train the classifier with the new target vector
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the classifier on the test set
    accuracy = clf.score(X_test, y_test)
    print("Accuracy:", accuracy)

def Cau31():
    X, y = mglearn.datasets.make_forge()

    glasses = [(10, 19), (30, 32), (37, 38), (50, 59), (63, 64), (69, 69), (120, 121), (124, 129), (130, 139),
               (160, 161), (164, 169), (180, 182), (185, 185), (189, 189), (190, 192), (194, 194), (196, 199),
               (260, 269), (270, 279), (300, 309), (330, 339), (358, 359), (360, 369)]
    faces = [(15, 17), (20, 22), (35, 36), (55, 56), (65, 66), (125, 126), (135, 136), (165, 166), (183, 184),
             (191, 193), (265, 266), (275, 276), (305, 306), (335, 336), (365, 366)]

    # Create a random target vector with length equal to the number of samples in X
    target = np.random.randint(0, 2, size=len(X))

    # Set the target values for the faces with glasses
    for i, face in enumerate(faces):
        start, end = face
    target[start - 1:end] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

    # Create a new SVC classifier with linear kernel
    svc_2 = svm.SVC(kernel='linear')

    # Perform 5-fold cross-validation
    scores = cross_val_score(svc_2, X_train, y_train, cv=5)

    # Print the mean accuracy
    print("Mean accuracy:", scores.mean())

def load_data():
    X, y = mglearn.datasets.make_forge()

    groups = {}
    for i, img in enumerate(X):
        key = y[i]
    if key not in groups:
        groups[key] = []
    groups[key].append(img)
    return groups

def Cau32():

    # Load the data
    data = load_data()

    # Extract the images of the person with indexes from 30 to 39
    person_data = []
    for i in range(100, 1001):
        person_data.extend(data[str(i)])

    # Split the data into training and testing sets

def print_faces(faces, labels, n):
    plt.figure(figsize=(10, 4))
    for i in range(n):
    plt.subplot(2, n, i+1)
    plt.imshow(faces[i], cmap='gray')
    plt.title('Label: {}'.format(labels[i]))
    plt.axis('off')
    plt.show()

def Cau33():
# Load the data
    data = load_data()

# Extract the images of the person with indexes from 100 to 1000
    person_data

def Cau35():
    news = fetch_20newsgroups(subset="all")
    print(type(news.data))
    print(type(news.target))
    print(type(news.target_names))
    print(news.target_names)

def Cau36():


    # Assume X and y are your features and target variables
    X_train, X_test, y_train, y_test = train_test_split(10, 20, test_size=0.25, random_state=0)
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    # Assume X_train and X_test are your training and testing text data

    # Use CountVectorizer to convert text data to a matrix of token counts
    count_vectorizer = CountVectorizer()
    X_train_count = count_vectorizer.fit_transform(X_train)
    X_test_count = count_vectorizer.transform(X_test)

def Cau37():
    X, y = mglearn.datasets.make_forge()


    # Load your dataset (e.g., 20 Newsgroups)
    # ...

    # Create three pipelines with different vectorizers
    pipeline_count = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
    ])

    pipeline_hashing = Pipeline([
    ('vectorizer', HashingVectorizer()),
    ('classifier', MultinomialNB())
    ])

    pipeline_tfidf = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
    ])

    # Train and evaluate each pipeline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    pipelines = [pipeline_count, pipeline_hashing, pipeline_tfidf]
    for pipeline in pipelines:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Vectorizer: {pipeline.named_steps['vectorizer'].__class__.__name__}, Accuracy: {accuracy:.3f}")

def k_fold_cross_validation(classifier, X, y, k=5):
"""
Performs k-fold cross-validation on the specified data and labels.

Parameters:
classifier (Classifier): The classifier to use for cross-validation.
X (array-like): The input data.
y (array-like): The target data.
k (int): The number of folds for cross-validation.

Returns:
scores (list): The cross-validation scores.
"""
    kf = KFold(n_splits=k)
    scores = []

    for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    scores.append(score)

    return scores

def Cau38():
    X, y = mglearn.datasets.make_forge()
    # Assume that `X` and `y` are the input and target data

    clf = LogisticRegression()
    scores = k_fold_cross_validation(clf, X, y)
    print(scores)

def Cau39():
    X, y = mglearn.datasets.make_forge()
    # Assume that `X` and `y` are the input and target data
    classifier = LogisticRegression()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Cau1()
    # Cau2()
    # Cau3()
    # Cau4()
    # Cau5()
    # Cau6()
    # Cau7()
    # Cau8()
    # Cau9()
    # Cau10()
    # Cau11()
    # Cau12()
    # Cau13()
    # Cau14()
    # Cau15()
    # Cau16()
    # Cau17()
    # Cau18()
    # Cau19()
    # Cau20()
    # Cau21()
    # Cau22()
    # Cau23(9)
    # Cau24()
    # Cau25()
    # Cau26()
    # Cau27()
    # Cau28()
    # Cau29()
    # Cau30()
    # Cau31()
    # Cau32()
    # Cau33()
    # Cau34()
    # Cau35()
    # Cau36()
    # Cau37()
    # Cau38()
    # Cau39()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
