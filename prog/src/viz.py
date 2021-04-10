# -*- coding:utf-8 -*-

from matplotlib import pyplot as plt
from TrainTestManager import TrainTestManager


def plot_query_strategy_metrics(manager: TrainTestManager, save_path):
    """
    Function that plots loss and accuracy of active learning process for single query strategy
    Args:
        manager(TrainTestManager): trained TrainTestManager object
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


def plot_compare_to_random_metrics(query_manager: TrainTestManager, random_manager: TrainTestManager, save_path):
    """
    Function that plots loss and accuracy of active learning process for 2 query strategies
    Usually used to compare query strategy to random query strategy
    Args:
        query_manager(TrainTestManager): TrainTestManager objects used with query strategy
        random_manager(TrainTestManager)
        save_path(str): path where to save figure
    """
    num_data = query_manager.metric_values['number_of_data']
    query_strat = query_manager.querier.__class__.__name__

    f = plt.figure(figsize=(10, 5))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    # loss plot
    ax1.plot(num_data, query_manager.metric_values['global_test_loss'], '-o', label=f'{query_strat} querying Test loss')
    ax1.plot(num_data, random_manager.metric_values['global_test_loss'], '-o', label='Random querying test loss')
    ax1.set_title('Test Loss')
    ax1.set_xlabel('Number of data used')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # accuracy plot
    ax2.plot(num_data, query_manager.metric_values['global_test_accuracy'], '-o',
             label=f'{query_strat} querying accuracy')
    ax2.plot(num_data, random_manager.metric_values['global_test_accuracy'], '-o', label='Random querying accuracy')
    ax2.set_title('Test accuracy')
    ax2.set_xlabel('Number of data used')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    f.savefig(save_path + 'fig1.png')
    plt.show()


def plot_all_metrics(random_manager: TrainTestManager, entropy_manager: TrainTestManager,
                     lc_manager: TrainTestManager, margin_manager: TrainTestManager,
                     save_path):
    """
    Function that plots loss and accuracy of active learning process for 2 query strategies
    Usually used to compare query strategy to random query strategy
    Args:
        random_manager(TrainTestManager): TrainTestManager objects used with Random query strategy
        entropy_manager(TrainTestManager): TrainTestManager objects used with Entropy query strategy
        lc_manager(TrainTestManager): TrainTestManager objects used with Uncertainty query strategy
        margin_manager(TrainTestManager): TrainTestManager objects used with Margin query strategy
        save_path(str): path where to save figure
    """

    num_data = random_manager.metric_values['number_of_data']

    f = plt.figure(figsize=(10, 5))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    # loss plot
    ax1.plot(num_data, random_manager.metric_values['global_test_loss'], '-o', label='Random querying Test loss')
    ax1.plot(num_data, entropy_manager.metric_values['global_test_loss'], '-o', label='Entropy querying test loss')
    ax1.plot(num_data, lc_manager.metric_values['global_test_loss'], '-o',
             label='Least Conf. querying test loss')
    ax1.plot(num_data, margin_manager.metric_values['global_test_loss'], '-o', label='Margin querying test loss')
    ax1.set_title('Test Loss')
    ax1.set_xlabel('Number of data used')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # accuracy plot
    ax2.plot(num_data, random_manager.metric_values['global_test_accuracy'], '-o', label='Random querying Test loss')
    ax2.plot(num_data, entropy_manager.metric_values['global_test_accuracy'], '-o', label='Entropy querying test loss')
    ax2.plot(num_data, lc_manager.metric_values['global_test_accuracy'], '-o',
             label='Uncertainty querying test loss')
    ax2.plot(num_data, margin_manager.metric_values['global_test_accuracy'], '-o', label='Margin querying test loss')
    ax2.set_title('Test accuracy')
    ax2.set_xlabel('Number of data used')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    f.savefig(save_path + 'fig1.png')
    plt.show()
