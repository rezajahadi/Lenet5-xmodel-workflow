import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
import argparse
import sys
import os
import shutil
from common import *
import matplotlib.pyplot as plt

DIVIDER = '-----------------------------------------'

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

def train_test(build_dir, batchsize, learnrate, epochs):

    dset_dir = build_dir + '/dataset'
    float_model = build_dir + '/float_model'

    if (torch.cuda.device_count() > 0):
        print('You have',torch.cuda.device_count(),'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device',str(i),': ',torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
        device = torch.device('cpu')

    model = LeNet5().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learnrate)

    # Define transformations for training and testing
    train_transform_32x32 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform_32x32 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(dset_dir, 
                                               train=True, 
                                               download=True,
                                               transform=train_transform_32x32)  # Update the transformation
    test_dataset = torchvision.datasets.MNIST(dset_dir,
                                              train=False, 
                                              download=True,
                                              transform=test_transform_32x32)  # Update the transformation

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batchsize, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batchsize, 
                                              shuffle=False)

    # for epoch in range(1, epochs + 1):
    #     train(model, device, train_loader, optimizer, epoch)
    #     test(model, device, test_loader)

    # shutil.rmtree(float_model, ignore_errors=True)    
    # os.makedirs(float_model)   
    # save_path = os.path.join(float_model, 'f_model.pth')
    # torch.save(model.state_dict(), save_path) 
    # print('Trained model written to', save_path)

    # return

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, device, test_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.4f}")

    # Plotting the loss and accuracy graphs
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.savefig(os.path.join(build_dir, 'loss_plot.png'))  # Save the loss plot in the specified directory

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.savefig(os.path.join(build_dir, 'accuracy_plot.png'))  # Save the accuracy plot in the specified directory

    plt.tight_layout()
    plt.show()

    # Saving the trained model
    shutil.rmtree(float_model, ignore_errors=True)    
    os.makedirs(float_model)   
    save_path = os.path.join(float_model, 'f_model.pth')
    torch.save(model.state_dict(), save_path) 
    print('Trained model written to', save_path)

    return

def run_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',   type=str,  default='build',       help='Path to build folder. Default is build')
    ap.add_argument('-b', '--batchsize',   type=int,  default=100,           help='Training batchsize. Must be an integer. Default is 100')
    ap.add_argument('-e', '--epochs',      type=int,  default=3,             help='Number of training epochs. Must be an integer. Default is 3')
    ap.add_argument('-lr','--learnrate',   type=float,default=0.001,         help='Optimizer learning rate. Must be floating-point value. Default is 0.001')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ',torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir    : ',args.build_dir)
    print ('--batchsize    : ',args.batchsize)
    print ('--learnrate    : ',args.learnrate)
    print ('--epochs       : ',args.epochs)
    print(DIVIDER)

    train_test(args.build_dir, args.batchsize, args.learnrate, args.epochs)

    return

if __name__ == '__main__':
    run_main()


# (vitis-ai-pytorch) Vitis-AI /workspace > export BUILD=./build
# (vitis-ai-pytorch) Vitis-AI /workspace > export LOG=${BUILD}/logs
# (vitis-ai-pytorch) Vitis-AI /workspace > mkdir -p ${LOG}
# (vitis-ai-pytorch) Vitis-AI /workspace > python -u train.py -d ${BUILD} 2>&1 | tee ${LOG}/train.logs