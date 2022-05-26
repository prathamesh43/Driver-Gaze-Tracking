import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from pickle import dump
import os
import copy
import torch
import struct
import codecs
import numpy as np
import torchvision
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
import time
import tqdm.notebook


use_gpu = torch.cuda.is_available()
if use_gpu:
    print('GPU is available!')
    device = "cuda"
    pinMem = True
else:
    print('GPU is not available!')
    device = "cpu"
    pinMem = False


savePath = 'torch_mods/'
if not os.path.isdir(savePath):
    os.makedirs(savePath)


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

trainDataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
testDataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
# Creating dataloader
trainLoader = DataLoader(trainDataset, batch_size=4, shuffle=True,num_workers=4, pin_memory=pinMem)
testLoader = DataLoader(testDataset, batch_size=4, shuffle=False,num_workers=4, pin_memory=pinMem)



class eorp(nn.Module):
    def __init__(self):
        super(eorp, self).__init__()
        self.l1 = nn.Sequential(nn.Linear(6, 16), nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        self.l4 = nn.Sequential(nn.Linear(8, 2), nn.Softmax())

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x


net = eorp()
print(net)

net = net.double().to(device)

init_weights = copy.deepcopy(net.l2[0].weight.data)

def train_model(model, criterion, num_epochs, learning_rate):
    start = time.time()
    train_loss = []  # List for saving the loss per epoch
    totalLoss = 0
    totalPreds = torch.tensor([0])

    for epoch in range(num_epochs):
        epochStartTime = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        running_loss = 0.0
        # Loading data in batches
        batch = 4
        for data in tqdm.notebook.tqdm(trainLoader):
            inputs, labels = data

            inputs, labels = inputs.float().to(device), labels.float().to(device)

            # Initializing model gradients to zero
            model.zero_grad()
            # Data feed-forward through the network
            outputs = model(inputs)
            # print(outputs.data)
            # Predicted class is the one with maximum probability
            _, preds = torch.max(outputs.data, 1)
            # print(preds)
            # Finding the MSE
            loss = criterion(outputs, labels)
            # Accumulating the loss for each batch
            running_loss += loss.item()
            # Backpropaging the error
            if batch == 0:
                totalLoss = loss
                totalPreds = preds
                batch += 1
            else:
                totalLoss += loss
                totalPreds = torch.cat((totalPreds, preds), 0)
                batch += 1

        totalLoss = totalLoss / batch
        # totalLoss.backward()
        totalLoss.backward(retain_graph=True)

        # Updating the model parameters
        for f in model.parameters():
            f.data.sub_(f.grad.data * learning_rate)

        epoch_loss = running_loss / 4920  # Total loss for one epoch
        train_loss.append(epoch_loss)  # Saving the loss over epochs for plotting the graph

        print('Epoch loss: {:.6f}'.format(epoch_loss))
        epochTimeEnd = time.time() - epochStartTime
        print('Epoch complete in {:.0f}m {:.0f}s'.format(
            epochTimeEnd // 60, epochTimeEnd % 60))
        print('-' * 25)
        # Plotting Loss vs Epochs
        fig1 = plt.figure(1)
        plt.plot(range(epoch + 1), train_loss, 'r--', label='train')
        if epoch == 0:
            plt.legend(loc='upper left')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Plot of training loss vs epochs')
        fig1.savefig(savePath + 'mlp_lossPlot.png')

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model

model = eorp().to(device) # Initilaizing the model
criterion = nn.BCELoss()
model = train_model(model, criterion, num_epochs=5, learning_rate=1) # Training the model

test_running_corr = 0
# Loading data in batches
batches = 0

model.eval()  # Testing the model in evaluation mode

for tsData in tqdm.notebook.tqdm(testLoader):
    inputs, _ = tsData

    inputs = inputs.float().to(device)

    with torch.no_grad():  # No back-propagation during testing; gradient computation is not required

        # Feedforward train data batch through model
        output = model(inputs)
        # Predicted class is the one with maximum probability
        _, preds = output.data.max(1)
        if batches == 0:
            totalPreds = preds
            batches = 1
        else:
            totalPreds = torch.cat((totalPreds, preds), 0)

print(preds)

ts_corr = np.sum(np.equal(totalPreds.cpu().numpy(), y_test))
ts_acc = ts_corr / y_test.shape[0]
print('Testing accuracy = ' + str(ts_acc * 100) + '%')

# model = Sequential()
# model.add(Dense(16, activation='sigmoid', input_dim=6))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(2, activation='softmax'))
#
# # sgd = optimizers.SGD(lr=0.1)
# model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model.fit(X_train, y_train, epochs=11, batch_size=4, validation_split=0.15)
# #
# pred_train = model.predict(X_train)
# # print(pred_train)
# #
# scores = model.evaluate(X_train, y_train, verbose=0)
# #
# print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))
# #
# preds = model.predict(X_test)
# # print(preds)
# #
# scores2 = model.evaluate(X_test, y_test, verbose=0)
# #
# print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))


# model.save( "models/" + str(round(scores[1], 5) ) + str(fnames) +'.h5')
# print('Model Saved!')
#
#
# # save the model
# # dump(model, open('model.pkl', 'wb'))
# # save the scaler
# dump(sc, open('scaler_prj.pkl', 'wb'))




