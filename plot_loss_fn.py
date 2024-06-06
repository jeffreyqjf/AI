import matplotlib.pyplot as plt

plt.ion()  # turning interactive mode on


def plot_loss(train_loss, test_loss, x_epochs, save_epochs):
    global graph
    if x_epochs == 1:
        graph = plt.plot([i for i in range(1, x_epochs + 1)], train_loss, color="red")[0]
        graph = plt.plot([i for i in range(1, x_epochs + 1)], test_loss, color="g")[0]
        plt.pause(0.25)
    else:
        graph.remove()
        graph = plt.plot([i for i in range(1, x_epochs + 1)], train_loss, color="red")[0]
        graph = plt.plot([i for i in range(1, x_epochs + 1)], test_loss, color="g")[0]
        plt.pause(0.25)
        if x_epochs % save_epochs == 0:
            plt.savefig(f"{x_epochs}_loss_figure")


if __name__ == "__main__":
    train_loss = [1]
    test_loss = [2]
    x_epochs = 1
    save_epochs = 40
    for i in range(100):
        plot_loss(train_loss=train_loss, test_loss=test_loss, x_epochs=1 + i, save_epochs=save_epochs)
        train_loss.append(5)
        test_loss.append(6)
