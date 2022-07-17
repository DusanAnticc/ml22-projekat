import sys
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def load_data(file_path):
    data = pd.read_csv(file_path, sep=r'\s*,\s*', header=0, encoding='utf-8', engine='python')
    return data


def preprocess(data):

    indexi = (data[data['workclass'] == "?"]).index
    for i in indexi:
        data = data.drop(i)

    indexi = (data[data['education'] == "?"]).index
    for i in indexi:
        data = data.drop(i)

    indexi = (data[data['marital_status'] == "?"]).index
    for i in indexi:
        data = data.drop(i)

    indexi = (data[data['occupation'] == "?"]).index
    for i in indexi:
        data = data.drop(i)

    indexi = (data[data['relationship'] == "?"]).index
    for i in indexi:
        data = data.drop(i)

    indexi = (data[data['race'] == "?"]).index
    for i in indexi:
        data = data.drop(i)

    indexi = (data[data['sex'] == "?"]).index
    for i in indexi:
        data = data.drop(i)

    indexi = (data[data['native_country'] == "?"]).index
    for i in indexi:
        data = data.drop(i)

    data["workclass"].replace(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'], [0, 1, 2, 3, 4, 5, 6, 7], inplace=True)
    data["education"].replace(['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], inplace=True)
    data["marital_status"].replace(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'], [0, 1, 2, 3, 4, 5, 6], inplace=True)
    data["occupation"].replace(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], inplace=True)
    data["relationship"].replace(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'], [0, 1, 2, 3, 4, 5], inplace=True)
    data["race"].replace(['White', 'Asian-Pac-Islander', 'Husband', 'Amer-Indian-Eskimo', 'Other', 'Black'], [0, 1, 2, 3, 4, 5], inplace=True)
    data["sex"].replace(['Female', 'Male'], [0, 1], inplace=True)
    data["native_country"].replace(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], inplace=True)
    data["salary"].replace(['<=50K', '>50K'], [0, 1], inplace=True)
    data["salary"].replace(['<=50K.', '>50K.'], [0, 1], inplace=True)
    return data


def calculate_F1_score(Y_true, Y_predicted):
    tn, fp, fn, tp = confusion_matrix(Y_true, Y_predicted).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)


if __name__ == "__main__":
    train_data = load_data(sys.argv[1])
    test_data = load_data(sys.argv[2])

    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    y_valid = test_data["salary"].to_numpy()
    del test_data['salary']

    svm = SVC()

    X_train = train_data.drop("salary", axis=1).to_numpy()
    Y_train = train_data["salary"].to_numpy()


    svm.fit(X_train, Y_train)
    y_pred = svm.predict(test_data.values)

    f_measure_ = f1_score(y_valid, y_pred, average='micro')
    print(f_measure_)