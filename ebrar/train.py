import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from dataloader import CustomDataset
from model import MyModel
from utils import check_accuracy

BATCH_SIZE = 64
EPOCHS = 21
LEARNING_RATE = 1e-3
NUM_CLASSES = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                      transforms.ToTensor()
                                      ])

# Load data 

train_dataset = CustomDataset(csv_file = "/Users/hcagri/Documents/ComputerVision/projects/ebrar/data/annotations.csv",
                        root_dir = '/Users/hcagri/Documents/ComputerVision/projects/ebrar/data/images', 
                        transform=transform_train
                        )

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MyModel(NUM_CLASSES).to(device)

for param in model.resnet18.parameters():
  param.requires_grad = False

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)


for epoch in tqdm(range(EPOCHS)):
    loop = tqdm(train_loader)
    epoch_loss = []
    
    for data, labels in loop:
        # Get data to cuda if possible
        data = data.to(device=device)
        labels = labels.to(device=device)
        
        # forward
        score = model(data)
        loss = criterion(score, labels)
        
        epoch_loss.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
      

    mean_loss = sum(epoch_loss) / len(epoch_loss)
    print("Mean Loss of Epoch = {:.2f}".format(mean_loss))

    if epoch % 2 == 0:
      train_acc = check_accuracy(train_loader, model, device)
    

torch.save(model.state_dict(), 'mymodel_v1.pth')
