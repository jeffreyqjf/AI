import torch
import torch.nn as nn
from main import MyNet, data_test
from torchvision import transforms,datasets
from PIL import Image
import matplotlib.pyplot as plt
data_path = r"./data"
pipeline = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=(0.1307,), std = (0.3081,))
                               ])

if __name__ == "__main__":

    model = MyNet()
    model.load_state_dict(torch.load(rf"C:\Users\31722\PycharmProjects\AI\my_pt\120_model.pt"))
    right = 0
    for i in range(1000):
        imgs, labels = data_test[i]
        result = model(imgs)
        fn = nn.Softmax(dim=1)
        result = fn(result)
        #print(result)
        #print(labels, torch.max(result))
        _, predicted = torch.max(result, dim=1)
        # print(predicted)
        if predicted[0] == labels:
            right += 1
        else:
            print(labels, torch.max(result))
    print(f"correct rate:{right/1000}")

    img = Image.open("./pic/4.jpg")
    img_t = pipeline(img)
    print(img_t.shape)
    result = model(img_t)
    result = fn(result)
    print(result)
    _, predicted = torch.max(result, dim=1)
    print(predicted)
"""
    img, labels = data_test[1]
    plt.imshow(img, cmap="gray")
    plt.show()
"""
