import matplotlib.pyplot as plt

def save_time_compute(time_compute:list[int], path:str= "time_compute.png"):
    """
    Save the time compute in a plot.
    """    
    # Plot the data
    plt.plot(time_compute, label='time compute', color='blue')
    plt.xlabel("iteration")
    plt.ylabel("time compute")
    plt.legend()
    plt.savefig(path)
    plt.close()
    
def save_evolued_loss(train_loss:list[list], path:str= "evolued_loss.png"):
    """
    Save the evolution of training loss over epochs for multiple iterations.
    """
    for i, loss_list in enumerate(train_loss):
        plt.plot(loss_list, label=f'Iteration {i}', color=plt.cm.viridis(i / len(train_loss)))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)
    plt.close()
    
def save_evolued_acc(acc:list, path:str= "evolued_acc.png"):
    """
    Save the evolution of training loss over epochs for multiple iterations.
    """
    plt.plot(acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(path)
    plt.close()


def save_monitor_life_agent(predictExplor, by_10_bad_inter, by_10_good_inter, pourcent_by_10, mean_val, path:str= "test02.png"):
    """
    Compare this snippet from scriptPython/test02.py:
    """
    def int_to_color(value):
        if value == 0:
            return "yellow"
        return "gray"

    # Create a figure and axis
    fig, ax = plt.subplots()

    for i, value in enumerate(predictExplor):
        ax.add_patch(plt.Rectangle((i, 0), width=0, height=10, color=int_to_color(value), alpha=0.4))
    plt.plot(by_10_bad_inter, label='bad inter', color='red')
    plt.plot(by_10_good_inter, label='good inter', color='green')
    plt.plot(pourcent_by_10, label='global', color='blue')
    plt.plot(mean_val, label='mean valence', color='black')
    plt.legend()
    plt.savefig(path)
    plt.close()