
from ucimlrepo import fetch_ucirepo 
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
# fetch dataset 
spambase = fetch_ucirepo(id=94) 
from collections import OrderedDict
# data (as pandas dataframes) 
X = spambase.data.features 
y = spambase.data.targets 
  
# metadata 
print(spambase.metadata) 
  
# variable information 
print(spambase.variables) 

data=torch.tensor(X[X.columns].values).to(torch.float)
label=torch.tensor(y.values).to(torch.float)
if torch.cuda.is_available():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset
idx=np.random.permutation(data.shape[0])
data=data[idx]
label=label[idx]
train = TensorDataset(data[:2065,:],label[:2065,:])
val = TensorDataset(data[2065:3065,:],label[2065:3065,:])
test = TensorDataset(data[3065:,:],label[3065:,:])

from torch.utils.data import DataLoader
train_dataloader = DataLoader(train,batch_size = 8,shuffle = True)
val_dataloader = DataLoader(val,batch_size = 8,shuffle = True)
test_dataloader = DataLoader(test,batch_size = 8,shuffle = True)

class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(57,20)
        self.relu1=nn.ReLU()
        self.linear2=nn.Linear(20,20)
        self.relu2=nn.ReLU()
        self.linear3=nn.Linear(20,20)
        self.relu3=nn.ReLU()
        self.linear4=nn.Linear(20,20)
        self.relu4=nn.ReLU()
        self.linear5=nn.Linear(20,20)
        self.relu5=nn.ReLU()
        self.linear6=nn.Linear(20,20)
        self.relu6=nn.ReLU()
        self.linear7=nn.Linear(20,20)
        self.relu7=nn.ReLU()
        self.linear8=nn.Linear(20,20)
        self.relu8=nn.ReLU()
        self.linear9=nn.Linear(20,2)
        self.softmax=nn.Softmax()
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu1(x)
        x=self.linear2(x)
        x=self.relu2(x)
        x=self.linear3(x)
        x=self.relu3(x)
        x=self.linear4(x)
        x=self.relu4(x)
        x=self.linear5(x)
        x=self.relu5(x)
        x=self.linear6(x)
        x=self.relu6(x)
        x=self.linear7(x)
        x=self.relu7(x)
        x=self.linear8(x)
        x=self.relu8(x)
        x=self.linear9(x)
        x=self.softmax(x)
        return x

def accuracy(loader, model, device):
    score=0
    total=0
    with torch.no_grad():
        for inputs,labels in loader:
            inputs=torch.log(0.1*torch.ones(inputs.shape)+inputs)
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total=total+labels.shape[0]
            accurates=int((predicted == labels[:,0].to(device)).sum().item())
            score=score+accurates
    return score/total

model1=A()
model1=nn.DataParallel(model1)
model1=model1.to(device)
optimizer = optim.SGD(model1.parameters(), lr=1e-2, momentum=0.5)
n_epochs = 100
criterion= nn.CrossEntropyLoss()

train_loss=torch.zeros(n_epochs)

train_accuracy=torch.zeros(n_epochs)
val_accuracy=torch.zeros(n_epochs)
test_accuracy=torch.zeros(n_epochs)

for epoch in range(n_epochs):
    total_loss=0
    size=0
    for inputs,labels in train_dataloader:
        inputs=torch.log(0.1*torch.ones(inputs.shape)+inputs)
        inputs,labels=inputs.to(device),labels.to(torch.long).to(device)
        outputs=model1(inputs)
        loss=criterion(outputs,labels[:,0])
        total_loss=total_loss+loss
        size=size+labels.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#    print(f'At epoch {epoch}, the train accuracy is {accuracy(train_dataloader,model,device)}, the test accuracy is {accuracy(val_dataloader,model,device)}')
    train_loss[epoch]=total_loss/size
    train_accuracy[epoch]=accuracy(train_dataloader,model1,device)
    val_accuracy[epoch]=accuracy(val_dataloader,model1,device)
    test_accuracy[epoch]=accuracy(test_dataloader,model1,device)
    if epoch%10==0:
        print(f'At epoch {epoch}, the train accuracy is {accuracy(train_dataloader,model1,device)}, the test accuracy is {accuracy(val_dataloader,model1,device)}')
   
accuracy(test_dataloader,model1,device)
import matplotlib.pyplot as plt
plt.plot(train_loss.detach().numpy())
plt.savefig('train_loss.jpg')
plt.show()
plt.close()

plt.plot(train_accuracy.detach().numpy())
plt.savefig('train_accuracy.jpg')
plt.show()
plt.close()

plt.plot(val_accuracy.detach().numpy())
plt.savefig('val_accuracy.jpg')
plt.show()
plt.close()

plt.plot(test_accuracy.detach().numpy())
plt.savefig('test_accuracy.jpg')
plt.show()
plt.close()

torch.save(model1.state_dict(),'spam_deeplearning.pth')

def accuracy1(loader, model, device):
    score=0
    total=0
    with torch.no_grad():
        for inputs,labels in loader:
            inputs=torch.log(0.1*torch.ones(inputs.shape)+inputs)
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total=total+labels.shape[0]
            accurates=int((predicted == labels[:,0].to(device)).sum().item())
            score=score+accurates
    return score/total
class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(57,20)
        self.relu1=nn.ReLU()
        self.linear2=nn.Linear(20,20)
        self.relu8=nn.ReLU()
        self.linear9=nn.Linear(20,2)
        self.softmax=nn.Softmax()
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu1(x)
        x=self.linear2(x)
        x=self.relu8(x)
        x=self.linear9(x)
        x=self.softmax(x)
        return x

model=B()
model=nn.DataParallel(model)
model=model.to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.1)
n_epochs = 100
criterion= nn.CrossEntropyLoss()

train_loss1=torch.zeros(n_epochs)

train_accuracy1=torch.zeros(n_epochs)
val_accuracy1=torch.zeros(n_epochs)
test_accuracy1=torch.zeros(n_epochs)

for epoch in range(n_epochs):
    total_loss1=0
    size=0
    for inputs,labels in train_dataloader:
        inputs=torch.log(0.1*torch.ones(inputs.shape)+inputs)
        inputs,labels=inputs.to(device),labels.to(torch.long).to(device)
        outputs=model(inputs)
        loss=criterion(outputs,labels[:,0])
        total_loss1=total_loss+loss
        size=size+labels.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#    print(f'At epoch {epoch}, the train accuracy is {accuracy(train_dataloader,model,device)}, the test accuracy is {accuracy(val_dataloader,model,device)}')
    train_loss1[epoch]=total_loss/size
    train_accuracy1[epoch]=accuracy1(train_dataloader,model,device)
    val_accuracy1[epoch]=accuracy1(val_dataloader,model,device)
    test_accuracy1[epoch]=accuracy1(test_dataloader,model,device)
    if epoch%10==0:
        print(f'At epoch {epoch}, the train accuracy is {accuracy1(train_dataloader,model,device)}, the test accuracy is {accuracy1(val_dataloader,model,device)}')
accuracy1(test_dataloader,model,device)
plt.plot(train_loss1.detach().numpy(),label='Train loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('train_los1.jpg')
plt.show()
plt.close()

plt.plot(train_accuracy1.detach().numpy(),label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('train_accuracy1.jpg')
plt.show()
plt.close()

plt.plot(val_accuracy1.detach().numpy(),label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('val_accuracy1.jpg')
plt.show()
plt.close()

plt.plot(test_accuracy1.detach().numpy(),label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('test_accuracy1.jpg')
plt.show()
plt.close()

torch.save(model.state_dict(),'spam_deeplearning_light.pth')
model_load=B()
model_load=nn.DataParallel(model_load)
model_load=model_load.to(device)
model_load.load_state_dict(torch.load('spam_deeplearning_light.pth'))
accuracy1(test_dataloader,model_load,device)

torch.save(model.state_dict(),'spam_deeplearning_light.pth')


class C(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(57,1824)
        self.relu1=nn.ReLU()
        self.linear2=nn.Linear(1824,912)
        self.relu2=nn.ReLU()
        self.linear3=nn.Linear(912,456)
        self.relu3=nn.ReLU()
        self.linear4=nn.Linear(456,228)
        self.relu4=nn.ReLU()
        self.linear5=nn.Linear(228,114)
        self.relu5=nn.ReLU()
        self.linear6=nn.Linear(114,57)
        self.relu6=nn.ReLU()
        self.linear7=nn.Linear(57,2)
        self.softmax=nn.Softmax()
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu1(x)
        x=self.linear2(x)
        x=self.relu2(x)
        x=self.linear3(x)
        x=self.relu3(x)
        x=self.linear4(x)
        x=self.relu4(x)
        x=self.linear5(x)
        x=self.relu5(x)
        x=self.linear6(x)
        x=self.relu6(x)
        x=self.linear7(x)
        x=self.softmax(x)
        return x
modelC=C()
modelC=nn.DataParallel(modelC)
modelC=modelC.to(device)

optimizer = optim.SGD(modelC.parameters(), lr=1e-3,momentum=0.9)
n_epochs = 100
criterion= nn.CrossEntropyLoss()

train_lossC=torch.zeros(n_epochs)

train_accuracyC=torch.zeros(n_epochs)
val_accuracyC=torch.zeros(n_epochs)
test_accuracyC=torch.zeros(n_epochs)

for epoch in range(n_epochs):
    total_loss1=0
    size=0
    for inputs,labels in train_dataloader:
        inputs=torch.log(0.1*torch.ones(inputs.shape)+inputs)
        inputs,labels=inputs.to(device),labels.to(torch.long).to(device)
        outputs=modelC(inputs)
        loss=criterion(outputs,labels[:,0])
        total_lossC=total_loss+loss
        size=size+labels.shape[0]
        grads = torch.autograd.grad(outputs=loss,inputs=modelC.parameters(),create_graph=True)
        grad_norm = 0
        for grad in grads:
            grad_norm += grad.pow(2).sum()
        
        optimizer.zero_grad()
        loss=loss+grad_norm
        loss.backward()
        optimizer.step()
#    print(f'At epoch {epoch}, the train accuracy is {accuracy(train_dataloader,model,device)}, the test accuracy is {accuracy(val_dataloader,model,device)}')
    train_lossC[epoch]=total_loss/size
    train_accuracyC[epoch]=accuracy1(train_dataloader,modelC,device)
    val_accuracyC[epoch]=accuracy1(val_dataloader,modelC,device)
    test_accuracyC[epoch]=accuracy1(test_dataloader,modelC,device)
    if epoch%10==0:
        print(f'At epoch {epoch}, the train accuracy is {accuracy1(train_dataloader,modelC,device)}, the test accuracy is {accuracy1(val_dataloader,modelC,device)}')
