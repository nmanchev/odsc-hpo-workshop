import torch
import os
import time

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchsummary

def load_mnist(device = "cpu", data_path = "data", train_fraction = 0.8, verbose = True):
    """
    # If the data is not present we can download it using
    
    from keras.datasets import mnist

    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    np.save("data/train_X", train_X)
    np.save("data/train_y", train_y)
    np.save("data/test_X", test_X)
    np.save("data/test_y", test_y)
    """    
    
    assert 0.0 <= train_fraction <= 1.0
    
    train_X = np.load(os.path.join(data_path, "train_X.npy"))
    train_y = np.load(os.path.join(data_path, "train_y.npy"))
    test_X = np.load(os.path.join(data_path, "test_X.npy"))
    test_y = np.load(os.path.join(data_path, "test_y.npy"))
                     
    indices = np.random.permutation(train_X.shape[0])
    training_size = int(train_X.shape[0] * train_fraction)
    training_idx, valid_idx = indices[:training_size], indices[training_size:]
    train_X, valid_X = train_X[training_idx,:], train_X[valid_idx,:]
    train_y, valid_y = train_y[training_idx], train_y[valid_idx]

    train_X = torch.tensor(train_X, device=device).unsqueeze(dim=1).float()
    train_y = torch.tensor(train_y, device=device)
    valid_X = torch.tensor(valid_X, device=device).unsqueeze(dim=1).float()
    valid_y = torch.tensor(valid_y, device=device)    
    test_X = torch.tensor(test_X, device=device).unsqueeze(dim=1).float()
    test_y = torch.tensor(test_y, device=device)
    
    if verbose:
        print("train_X size : ", train_X.shape)
        print("train_y size : ", train_y.shape)
        print("valid_X size : ", valid_X.shape)
        print("valid_y size : ", valid_y.shape)
        print("test_X size  : ", test_X.shape)
        print("test_y size  : ", test_y.shape)
        
    return train_X, train_y, valid_X, valid_y, test_X, test_y

class Net(nn.Module):
    
    def __init__(self, activ):
        
        super().__init__()
        
        self.activ = activ
        
        self.conv1 = nn.Conv2d(in_channels=1,              
                               out_channels=16,            
                               kernel_size=5,             
                               stride=1,                   
                               padding=2)                            
                
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)    
        self.pool2 = nn.MaxPool2d(2)               
        
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
        self.softmax = F.log_softmax

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.activ(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)   
        x = self.fc(x)
    
        output = self.softmax(x, dim=1)
        
        return output
    
    def summary(self, input_size=(1, 28, 28)):
        
        torchsummary.summary(self, input_size = input_size)
        
    def reset(self):
    
        torch.manual_seed(1234)
        np.random.seed(1234)
        
        for layer in self.children():
           if hasattr(layer, "reset_parameters"):
               layer.reset_parameters()

        
def train(net, batch_size, loss_crierion, optimizer, train_X, train_y, max_epochs = 1, verbose = True, iterations = 1000):
    
    loss_hist = []

    samples_per_epoch = train_X.shape[0]
    
    for epoch in range(max_epochs):

        for index in range(0, samples_per_epoch, batch_size):

            images = train_X[index:index + batch_size, :, :]
            labels = train_y[index:index + batch_size,]

            optimizer.zero_grad()

            out = net(images)
            loss = loss_crierion(out, labels)
            loss_hist.append(loss.item())

            loss.backward()
            optimizer.step()
            
            if verbose and (index % iterations == 0):
                 print("Epochs [{:d}/{:d}], Samples[{:d}/{:d}], Loss: {:.4f}".format(epoch + 1, max_epochs, index + iterations, samples_per_epoch, loss))
                    
    return loss_hist

def test(net, test_X, test_y):
    
    with torch.no_grad():
        out = net.forward(test_X)
        pred = torch.argmax(out, 1)
        correct = pred.eq(test_y).sum()

    accuracy = correct / test_X.shape[0]
    
    return accuracy

def main():
    
    # Check for GPU    
    if torch.cuda.is_available():
      # This degrades the GPU performance but enforces reproducibility.
      torch.use_deterministic_algorithms(True)
      os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

      print("CUDA available. Using GPU acceleration.\n")
      device = "cuda"
    else:
      print("CUDA is NOT available. Using CPU for training.\n")
      device = "cpu"
    
    # Load data
    train_X, train_y, valid_X, valid_y, test_X, test_y = load_mnist(device)
    
    # Take the first 20,000 training images only
    # in case the script is running on a small HW tier
    train_X = train_X[:20000]
    train_y = train_y[:20000]
    
    torch.manual_seed(1234)
    np.random.seed(1234)

    # Build network
    activ = F.relu

    net = Net(activ = activ)
    #net.to(device)
    
    # Set Hyperparameters
    batch_size = 100
    learning_rate = 0.001
    momentum = 0.8
    max_epochs = 1
    activ = F.relu
    
    # Train
    print("\nTraining...")
    loss_crierion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    train(net, batch_size, loss_crierion, optimizer, train_X, train_y)
    
    # Test
    print("\nTesting...")
    acc = test(net, test_X, test_y)
    print("Accuracy on test: {:.4f}".format(acc))

if __name__ == "__main__":
    
    main()