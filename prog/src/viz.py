# -*- coding:utf-8 -*-

from matplotlib import pyplot as plt
from TrainTestManager import TrainTestManager


def plot_query_strategy_metrics(manager: TrainTestManager, save_path):
    """
    Function that plots loss and accuracy of active learning process for single query strategy
    Args:
        manager(object): trained TrainTestManager object
        save_path(str): path where to save figure
    """

    num_data = manager.metric_values['number_of_data']

    f = plt.figure(figsize=(10, 5))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    # loss plot
    ax1.plot(num_data, manager.metric_values['global_train_loss'], '-o', label='Training loss')
    ax1.plot(num_data, manager.metric_values['global_test_loss'], '-o', label='Test loss')
    ax1.set_title('Training and test loss')
    ax1.set_xlabel('Number of data used')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # accuracy plot
    ax2.plot(num_data, manager.metric_values['global_train_accuracy'], '-o', label='Training accuracy')
    ax2.plot(num_data, manager.metric_values['global_test_accuracy'], '-o', label='Test accuracy')
    ax2.set_title('Training and test accuracy')
    ax2.set_xlabel('Number of data used')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    f.savefig(save_path + 'fig1.png')
    plt.show()
