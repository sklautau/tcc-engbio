import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

csv_train = 'D:/sofia/ufpa/tcc/dataset_updated/train_data.csv'
df_train = pd.read_csv(csv_train, dtype=str)
#df_train = pd.read_csv(csv_train, encoding="latin1", dtype=str,)

csv_test = 'D:/sofia/ufpa/tcc/dataset_updated/test_data.csv'
df_test = pd.read_csv(csv_test, dtype=str)

y = df_train['target'].tolist()

values, counts = np.unique(np.array(y),return_counts=True)
print(y)
print('values=',values)
print('counts=',counts)

neg= counts[0]
pos= counts[1]
total = neg+pos

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


X = y # = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)

# split data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a dummy classifier that always predicts the most frequent class
dummy = DummyClassifier(strategy='most_frequent')

# fit the classifier to the training data
dummy.fit(X, y)

# evaluate the classifier on the train data
accuracy_train = dummy.score(X, y)

print(f"Dummy classifier train accuracy: {accuracy_train:.3f}")


# # evaluate the classifier on the testing data
# accuracy_test = dummy.score(X_test, y_test)

# print(f"Dummy classifier test accuracy: {accuracy_test:.3f}")

