import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from pickle import dump



fnames = ['r1', 'r2', "j4", "j5" ]
folders = ["rm", "lm", "cm", "rand"]
#folders = ["sf", "rand"]
temp = []
df_dict = []
for x in folders:
    temp = []
    for v in fnames:
        df = pd.read_csv("New/vids/vids_p/" + x + "/" + v + "_" + x + ".csv")
        temp.append(df)
    df_dict.append(pd.concat(temp, ignore_index=True))

dataset = pd.concat(df_dict, ignore_index=True)
lst = [0 for x in range(0,dataset.shape[0])]
dataset["r"] = lst


folders = ["sf"]
temp = []
df_dict = []

for x in folders:
    temp = []
    for v in fnames:
        df = pd.read_csv("New/vids/vids_p/" + x + "/" + v + "_" + x + ".csv")
        temp.append(df)
    df_dict.append(pd.concat(temp, ignore_index=True))

dataset2 = pd.concat(df_dict, ignore_index=True)
lst = [1 for x in range(0,dataset2.shape[0])]
dataset2["r"] = lst
df_dict2 = [dataset2, dataset]
dataset1 = pd.concat(df_dict2, ignore_index=True)
print(dataset1)



X = dataset1.iloc[:, 1:-2].values
y = dataset1.iloc[:, -1].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=20)


sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



model = Sequential()
model.add(Dense(16, activation='sigmoid', input_dim=6))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

# sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=11, batch_size=4, validation_split=0.15)
#
pred_train = model.predict(X_train)
# print(pred_train)
#
scores = model.evaluate(X_train, y_train, verbose=0)
#
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))
#
preds = model.predict(X_test)
# print(preds)
#
scores2 = model.evaluate(X_test, y_test, verbose=0)
#
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))


model.save( "models/" + str(round(scores[1], 5) ) + str(fnames) +'.h5')
print('Model Saved!')


# save the model
# dump(model, open('model.pkl', 'wb'))
# save the scaler
dump(sc, open('scaler_prj.pkl', 'wb'))




