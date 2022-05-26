import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

fnames = [ "j4", "j5" , 's1']
folders = ["rm", "lm", "cm", "sf", "rand"]
# folders = ["sf", "rand"]
temp = []
df_dict = []
df_dict2 = []

for x in folders:
    temp = []
    for v in fnames:
        df = pd.read_csv("New/vids/vids_p/" + x + "/" + v + "_" + x + ".csv")
        temp.append(df)
    df_dict.append(pd.concat(temp, ignore_index=True))

dataset = pd.concat(df_dict, ignore_index=True)

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
#

# fnames2 = ["j2"]
# for x in folders:
#     temp = []
#     for l in fnames2:
#         df = pd.read_csv("vids/" + x + "/" + l + "_" + x + ".csv")
#         temp.append(df)
#     df_dict2.append(pd.concat(temp, ignore_index=True))
#
# dataset2 = pd.concat(df_dict2, ignore_index=True)
#
# X2 = dataset2.iloc[:, 1:-1].values
# y2 = dataset2.iloc[:, -1].values
# y2 = le.fit_transform(y2)
#
#
# X_train, y_train = X,y
# X_test, y_test = X2, y2


#Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=30)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
#
# Feature Scaling
sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])
print(X_train)
# print(X_test)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes)

model = Sequential()
model.add(Dense(50, activation='sigmoid', input_dim=6))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(5, activation='softmax'))

# sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=4, validation_split=0.15)
#
pred_train = model.predict(X_train)
# print(pred_train)
#
scores = model.evaluate(X_train, y_train, verbose=0)
#
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))
#
preds = model.predict(X_test)
print(preds)
#
scores2 = model.evaluate(X_test, y_test, verbose=0)
#
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))
