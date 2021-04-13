# Active Learning Project

This project has been developed as part of the class assignment in the Neural Network course IFT 780 at the 
University of Sherbooke.

It consists of training deep learning models by using only a subset of the training dataset. For that purpose, 
three uncertainty based strategy of selection are implemented:

## Query strategies
    - Entropy based strategy
    - Margin Sampling (MS) strategy
    - Least Confidence (LC) strategy
    - Random (baseline)
Further details on the aforementioned can be found here: http://burrsettles.com/pub/settles.activelearning.pdf 

For the experiment, two datasets are used:

## Datasets
    - Fashion Mnist
    - Cifar100

Regarding models, the following are implemented here:

## Model
    - SENet : Sqeeze and Excite Net
    - ResNet : A version of the Residual Network
    - DenseNet : The densely convolutional network
    - SimpleCNN : A baseline CNN model of type `Conv-ReLu` -> `Conv-ReLu-Pooling`.... 

## Run the script
Using a version of python 3.5 or higher, the code can be run as follows:
```
python -u train.py --initial_data_ratio=0.2 --dataset=mnistfashion  --query_size=2000 --model=SENet --mode=All
```
This will run the training of the `Squeeze and Excite network` with different pooling strategies, then will output 
a graph that compares different results(accuracy and loss).

## Available Parameters
* `--train_set_threshold` : the allowed proportion of the training set that can be used in active learning process
* `--initial_data_ratio` : the proportion of training data to start with. This proportion is drawn uniformly at random
* `--num-epochs` : the number of epoch for each pooling
* `--dataset`: the dataset to use
* `--query_size`: the size of new sample to pool and label
* `--model`: the model to use
* `--query_strategy`: which query strategy to use. This will be ignored if `--mode=All`
* `--mode` : which mode to use. `Single` for just one strategy, `Compare` to compare with `Random`, or `All` 
  to compare all the strategies.
* `--batch_size`: the batch size
* `--save_path`: the path to where should the outpout be saved.
* `--optimizer`: the optimizer, `Adam` or `SGD`
*  `--lr` : the learning rate
*  `--data_aug`: boolean value for data augmentation

## Hyper parameters search

An intensive search of hyper parameters can be run using the command below:
```
python -u hpsearch.py --dataset=cifar100 --model=ResNet --data_aug --query_strategy=Margin --save_path=./runs
```

This command will run a search of hyper parameters for the `ResNet` model on `cifar100` dataset with data augmentation,
and the pooling will be done using the `Margin Sampling` strategy.

The progress of the search can be monitored using `tensor board` by running the following commands:
To start tensorboard run,
```
tensorboard --logdir=runs
```
Then by default, tensor board starts at this address: `http://localhost:6006/`.

If the code is run from a remote machine that can be accessed via `ssh`, then use the command below
to create a `ssh tunnel`:

```
ssh -N -f -L localhost:[local port]:localhost:[remote port] [uname]@[remote address]
```

