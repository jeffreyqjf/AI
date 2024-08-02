from torchvision import models, transforms
from PIL import Image
import torch

resnet = models.resnet101(pretrained=True)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


with open("C:\\Users\\31722\PycharmProjects\AI\本书代码文件\dlwpt-code-master 2\dlwpt-code-master\data\p1ch2\imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines() ]
#  img = Image.open("C:\\Users\\31722\\Desktop\\剪辑\\homework1.jpg")
img = Image.open("C:\\Users\\31722\Downloads\\03a4d2cc9f6ecff013e547e6036a6640.jpeg")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)
resnet.eval()
out = resnet(batch_t)
#  print(out)
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]])
print(percentage[index[0]])

