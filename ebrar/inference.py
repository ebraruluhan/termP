import torch 
from dataloader import CustomDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import MyModel
import matplotlib.pyplot as plt
from utils import img_convert
import time




def inference(model, image, device):

    with torch.no_grad():
        
        image = image.to(device=device)
        scores_class = model(image)
        _, prediction_label = scores_class.max(1)
            
    return prediction_label

transform_test = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((224,224)),
                                      transforms.ToTensor()
                                      ])

test_dataset = CustomDataset(csv_file = "data/annotations.csv",
                            root_dir = 'data/images', 
                            transform=transform_test)

test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

model = MyModel(10)
model.load_state_dict(torch.load('mymodel_v1.pth', map_location='cpu'))
model.eval()

imgs, targets = next(iter(test_loader))

label_names = ['blue square', 'green square', 'red square', 
               'blue circle', 'green circle', 'red circle',
               'blue triangle', 'green triangle', 'red triangle', 'background']

for img in imgs:
    img_i = img.unsqueeze(0)
    s = time.time()
    label = inference(model, img_i, 'cpu')
    e = time.time()
    img = img_convert(img)
    plt.imshow(img)
    plt.title(label_names[label])
    plt.show()

