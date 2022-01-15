import torch 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torchvision.transforms import transforms
from model import MyModel
import matplotlib.pyplot as plt
from utils import img_convert
from skimage import io
import os


def inference(model, image, device):

    with torch.no_grad():
        
        image = image.to(device=device)
        scores_class = model(image)
        _, prediction_label = scores_class.max(1)
            
    return prediction_label

transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((224,224)),
                                      transforms.ToTensor()
                                      ])

label_names = ['blue square', 'green square', 'red square', 
               'blue circle', 'green circle', 'red circle',
               'blue triangle', 'green triangle', 'red triangle', 'background']
model = MyModel(10)
model.load_state_dict(torch.load('mymodel_v1.pth', map_location='cpu'))
model.eval()


root_dir = "data/images"
img_name = "78.jpg"

img_path = os.path.join(root_dir,img_name)
img = io.imread(img_path)

img = transform(img)

label = inference(model, img.unsqueeze(0), 'cpu')
img = img_convert(img)
plt.imshow(img)
plt.title(label_names[label])
plt.show()
