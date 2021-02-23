import matplotlib.pyplot as plt

#Expects the data dictionary as input
#
#
def plotAlg(data, algorithm, plotWhat, saveToFile=False):

    if algorithm not in data["alex"].keys():
        print(f"No data found for an algorithm by the name of: {algorithm}")
        return

    if plotWhat not in data["alex"][algorithm]["train"].keys():
        print(f"No data found in algorithm {algorithm} under the name {plotWhat}")
        return

    plt.plot(data["alex"][algorithm]["train"][plotWhat], "-", color="red")
    plt.plot(data["alex"][algorithm]["validation"][plotWhat], "--", color="orange")
    plt.plot(data["vgg"][algorithm]["train"][plotWhat], "-.", color="blue")
    plt.plot(data["vgg"][algorithm]["validation"][plotWhat], ":", color="cyan")

    plt.legend([f"Alex training {plotWhat[6:]}",
                f"Alex val. {plotWhat[6:]}",
                f"VGG training {plotWhat[6:]}",
                f"VGG val. {plotWhat[6:]}",
                ])

    plt.title(f"Optimizing using {algorithm}")
    plt.xlabel("Epochs")
    plt.ylabel(plotWhat[6:].capitalize())


    if (saveToFile):
        plt.savefig(f"plt_{algorithm}_{plotWhat[6:]}.png")
    else:
        plt.show()
