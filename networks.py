
import torch.nn as nn
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from plot_loss_fn import plot_loss

device = torch.device("cuda")
data_path = r"./data"
channel = 1
#  归一化
original_datas = datasets.MNIST(data_path, train=False, download=True, transform=transforms.ToTensor())
data_transform = torch.stack([one_img for one_img, _ in original_datas], dim=3)
#  print(data_transform.shape)
data_mean = data_transform.view(channel, -1).mean(dim=1)
data_std = data_transform.view(channel, -1).std(dim=1)
#  print(f"data_mean:{data_mean}")
#  print(f"data_std:{data_std}")
pipeline = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=data_mean, std=data_std)
                               ])
data_train = datasets.MNIST(data_path, train=True, download=True,transform=pipeline)
data_test = datasets.MNIST(data_path, train=False, download=True, transform=pipeline)

#  print(len(data_train)) 60000
#  img, label = data_train[100]
#  print(img.shape, label)  #  size [1, 28, 28]
train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
test_loader = DataLoader(data_test, batch_size=64, shuffle=True)

class MyNet(nn.Module):
    def __init__(self):
        """
        一开始图像是1*28*28
        第一次卷积--> 64*28*28
        第一次池化--> 64*13*13
        第二次卷积--> 32*13*13
        第二次池化--> 32*6*6
        第三次卷积--> 16*6*6


        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()
        self.act3 = nn.Tanh()
        self.act4 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(16 * 6 * 6, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        output = self.pool1(self.act1(self.conv1(x)))
        #  print(output.shape) #torch.Size([1, 64, 13, 13])
        output = self.pool2(self.act2(self.conv2(output)))
        #  print(output.shape) #torch.Size([1, 32, 6, 6])
        output = self.act3(self.conv3(output))
        #  print(output.shape) #torch.Size([1, 16, 6, 6])
        output = output.view(-1, 16 * 6 * 6)
        output = self.act4(self.linear1(output))
        output = self.linear2(output)
        return output


def loop(epochs, optimizer, model, loss_fn, train_loader):
    data_plot = {"loss_train":[], "loss_test":[]}
    for epoch in range(1, epochs+1):
        #  print(f"hello{epoch}")
        loss_train_sum = 0.0
        loss_test_sum = 0.0

        for imgs_test, labels_test in test_loader:
            labels_test = labels_test.to(device=device)
            imgs_test = imgs_test.to(device=device)

            output_test = model(imgs_test)
            #  print(labels_test.shape, output_test.shape, imgs_test.shape)
            loss_test = loss_fn(output_test, labels_test)
            optimizer.zero_grad()

            loss_test_sum += loss_test.item()

        for imgs_train, labels_train in train_loader:
            labels_train = labels_train.to(device=device)
            imgs_train = imgs_train.to(device=device)

            outputs_train = model(imgs_train)
            loss_train = loss_fn(outputs_train, labels_train)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            loss_train_sum += loss_train.item()

        data_plot["loss_train"].append(loss_train_sum)
        data_plot["loss_test"].append(loss_test_sum)

        plot_loss(train_loss=data_plot["loss_train"], test_loss=data_plot["loss_test"], x_epochs=epoch, save_epochs=2)
        if epoch == 1 or epoch % 5 == 0:
            print(f"""{epoch} epoch, \ntraining loss = {loss_train_sum}\ntest loss = {loss_test_sum}\n""")
        if epoch % 40 == 0:
            torch.save(model.state_dict(), rf".\my_pt\{epoch}_model.pt")
            print(data_plot)
            #  model.load_state_dict(torch.load(rf"C:\Users\31722\PycharmProjects\AI\my_pt\{epoch}_model.pt", map_location=device)) mode_load
    print(data_plot)


class Block(nn.Module):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    model = MyNet().to(device=device)
    #  model.load_state_dict(torch.load(rf"C:\Users\31722\PycharmProjects\AI\my_pt\40_model.pt", map_location=device))
    """
    img, label = data_train[0]
    ret = model(img.unsqueeze(0))
    print(ret.shape) #torch.Size([1, 10])
    """
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    loop(epochs=1000,
         optimizer=optimizer,
         model=model,
         loss_fn=loss_fn,
         train_loader=train_loader)