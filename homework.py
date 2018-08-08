from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools

class Homework(object):
    def __init__(self):
        self.dataset = {}

        self.run()

    def load_data(self):
        with open("abalone_dataset.txt", "r") as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                line = line.split("\t")
                sex, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight, classified = line

                if sex == "M":
                    sex = 1
                elif sex == "F":
                    sex = 2
                elif sex == "I":
                    sex = 3

                self.dataset[i] = {
                    "sex": int(sex),
                    "length": float(length),
                    "diameter": float(diameter),
                    "height": float(height),
                    "whole weight": float(whole_weight),
                    "shucked weight": float(shucked_weight),
                    "viscera weight": float(viscera_weight),
                    "shell weight": float(shell_weight),
                    "classified": int(classified)
                }

    def features(self, *keys):
        temp_dataset = []
        for i, data in self.dataset.items():
            temp_data = []

            if "all" in keys:
                for key, value in data.items():
                    if key == "classified":
                        continue

                    temp_data.append(value)

            else:
                for key in keys:
                    temp_data.append(data[key])

            temp_dataset.append(temp_data)

        return temp_dataset

    def naive_bayes_classification(self, training_sample_numbers, *features):
        gaussian = GaussianNB()
        features = self.features(*features)
        y = np.ravel(self.features("classified"))

        kF = KFold(n_splits= math.ceil(len(self.dataset) / training_sample_numbers))

        scores = cross_val_score(gaussian, features, y, cv= kF)

        y_predicts = cross_val_predict(gaussian, features, y, cv=kF)

        c_matrix = confusion_matrix(y, y_predicts)

        print("Scores:")
        print(scores)
        print("\n")
        print("Predictions:")
        print(y_predicts)
        print("\n")
        print("Confusion Matrix:")
        print(c_matrix)

        print("Total misclassification errors:")
        classification_errors = c_matrix[0][1]  + c_matrix[0][2] + c_matrix[1][0] + c_matrix[1][2] + c_matrix[2][0] + c_matrix[2][1]
        print(classification_errors)

        print("\n")
        print("Accuracy:")
        accuracy = ((c_matrix[0][0] + c_matrix[1][1] + c_matrix[2][2]) / (len(self.dataset))) * 100
        print("%s%%" % accuracy)

        plt.figure()
        self.plot_confusion_matrix(c_matrix, classes=["Young", "Middle Aged", "Old"],
                              title='Confusion Matrix')
        plt.show()
        print("\n*************\n")

    def run(self):
        self.load_data()
        self.naive_bayes_classification(100, "sex", "length", "diameter")
        self.naive_bayes_classification(1000, "sex", "length", "diameter")
        self.naive_bayes_classification(100, "all")
        self.naive_bayes_classification(1000, "all")

    def plot_confusion_matrix(self, cm, classes,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')



Homework()