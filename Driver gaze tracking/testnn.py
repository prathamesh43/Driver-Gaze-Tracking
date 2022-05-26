import torch
from torch.autograd import Variable


x_data = Variable(torch.Tensor([[0.0], [15.0] , [40.0], [62.0], [72.0], [82.0]]))
y_data = Variable(torch.Tensor([[16.0], [465.0], [650.0], [890.0], [950.0], [1264.0]]))


class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


# our model
our_model = LinearRegressionModel()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(our_model.parameters(), lr=0.01)

for epoch in range(10):
    # Forward pass: Compute predicted y by passing
    # x to the model
    pred_y = our_model(x_data)
    print(pred_y)

    # Compute and print loss
    loss = criterion(pred_y, y_data)

    # Zero gradients, perform a backward pass,
    # and update the weights.

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print('epoch {}, loss {}'.format(epoch, loss.item()))


new_var = Variable(torch.Tensor([[76]]))
pred_y = our_model(new_var)
print(pred_y)
print("predict (after training)", 76, our_model(new_var).item())

