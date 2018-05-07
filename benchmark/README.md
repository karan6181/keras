# Keras Benchmarks

## Overview
The benchmark module aims to provide a performance comparison on different Keras backends using various models and 
dataset on CPU, 1 GPU and multi-GPU machines.
Currently supported backends: TensorFlow, Apache MXNet 

## Setup
To install MXNet backend refer to 
[Installation](https://github.com/awslabs/keras-apache-mxnet/wiki/Installation#1-install-keras-with-apache-mxnet-backend)

To switch between different backends refer to 
[configure Keras backend](https://github.com/awslabs/keras-apache-mxnet/wiki/Installation#2-configure-keras-backend)

## CNN Benchmarks
We provide benchmark scripts to run on CIFAR-10, ImageNet and Synthetic Dataset(randomly generated)

### CIFAR-10 Dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset has 60000 32x32 color images in 10 classes.
The [training scripts](https://github.com/awslabs/keras-apache-mxnet/blob/master/benchmark/image-classification/benchmark_resnet.py)
 will automatically download the dataset, you need to provide dataset name, resnet version 
(1 or 2), number of layers (20, 56, or 110), number of GPUs to use, and number of epoch to use(optional, Default:200). 

Example Usage:

`python benchmark_resnet.py --dataset cifar10 --version 1 --layers 56 --gpus 4 --epoch 20`


### ImageNet Dataset
First, download ImageNet Dataset from [here](http://image-net.org/download), there are total 1.4 million images 
with 1000 classes, each class is in a subfolder. In this script, each image is processed to size 256x256

Since ImageNet Dataset is too large, there are two training mode for data that does not fit into memory: 
[`train_on_batch`](https://keras.io/models/sequential/#train_on_batch) and 
[`fit_generator`](https://keras.io/models/sequential/#fit_generator), 
we recommend train_on_batch since it's more efficient on multi_gpu.
(Refer to [Keras Document](https://keras.io/getting-started/faq/#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory) 
and Keras Issue [#9502](https://github.com/keras-team/keras/issues/9502), 
[#9204](https://github.com/keras-team/keras/issues/9204), [#9647](https://github.com/keras-team/keras/issues/9647))

Compare to CIFAR-10, you need to provide additional params: training mode and path to imagenet dataset.

Example usage:

`python benchmark_resnet.py --dataset imagenet --mxnet_backend_training_speed.pngversion 1 -layers 56 --gpus 4 --epoch 20 --train_mode train_on_batch --data_path home/ubuntu/imagenet/train/`

### Synthetic Dataset
We used benchmark scripts from 
[TensorFlow Benchmark](https://github.com/tensorflow/benchmarks/tree/keras-benchmarks/scripts/keras_benchmarks) 
official repo, and modified slightly for our use case.

Directly run the shell script to launch the benchmark, provide one of the configurations in config.json and whether 
you want to benchmark inference speed (True or False), and number of epoch to use(optional, Default:200). 

Example Usage:

`sh run_<backend-type>_backend.sh gpu_config False 20`

### CNN Benchmark Results
Here we list the result of MXNet backend training speed on CIFAR-10, ImageNet and Synthetic Data using 
ResNet50V1 model, on CPU, 1, 4, 8 GPUs using AWS instances. 
Hardware specifications of the instances can be found [here](https://aws.amazon.com/ec2/instance-types/)

For more detailed benchmark results, please refer to [CNN results](https://github.com/awslabs/keras-apache-mxnet/tree/keras2_mxnet_backend/benchmark/benchmark_result/CNN_result.md). 

|||
|  ------ | ------ |
|  Keras Version | 2.1.5 |
|  MXNet Version | 1.1.0 |
|  Data Format | Channel first |

|  Instance | GPU used | Package | CIFAR-10 | ImageNet | Synthetic Data |
|  ------ | ------ | ------ | ------ | ------ | ------ |
|  C5.18xLarge | 0  | mxnet-mkl | 87 | N/A | 9 |
|  P3.8xLarge | 1 | mxnet-cu90 | N/A | 165 | 229 |
|  P3.8xLarge | 4 | mxnet-cu90 | 1792 | 538 | 728 |
|  P3.16xLarge | 8 | mxnet-cu90 | 1618 | 728 | 963 |

![MXNet backend training speed](https://github.com/roywei/keras/blob/benchmark_result/benchmark/benchmark_result/mxnet_backend_training_speed.png)

Note: X-axis is number of GPUs used, Y-axis is training speed(images/second)

## RNN Benchmarks

We provide benchmark scripts to run on Synthetic(randomly generated), Nietzsche, and WikiText-2 character level Dataset.

Directly run the shell script to launch the benchmark, provide one of the configurations in config.json and whether you want to benchmark inference speed (True or False), and number of epoch to use(optional, Default:20). 

Example Usage:

`sh run_<backend-type>_backend.sh gpu_config False 20`

### Synthetic Dataset

We used benchmark scripts from [TensorFlow Benchmark](https://github.com/tensorflow/benchmarks/tree/keras-benchmarks/scripts/keras_benchmarks) official repo, and modified slightly for our use case.

Put `lstm_synthetic` as models parameter in `run_<backend-type>_backend.sh`

### Nietzsche Dataset

We have used an official Keras LSTM example scripts [lstm_text_generation.py](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py), and modified slightly for our use case.

Put `lstm_nietzsche` as models parameter in `run_<backend-type>_backend.sh`

### WikiText-2 Dataset

We have used an official WikiText-2 character level Dataset from this [link](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset).

The `lstm_text_generation.py` includes a dataset that is hosted on S3 bucket from this [link](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip) (This is a WikiText-2 raw character level data).

Put `lstm_wikitext2` as models parameter in `run_<backend-type>_backend.sh`

### RNN Benchmark Results

Here, we list the result on Synthetic, Nietzsche, and WikiText-2 dataset using Sequential model(LSTM) on Amazon AWS C5.18xLarge(CPU), C5.xLarge(CPU), and P3.8xLarge(1, 4 GPUs) instance with MXNet and TensorFlow backend. Batch size is 128 and set `unroll=True` in Recurrent layer. For more details about the instance configuration, please refer [P3](https://aws.amazon.com/ec2/instance-types/p3/) and [C5](https://aws.amazon.com/ec2/instance-types/c5/).

For more detailed benchmark results, please refer to [RNN results.](benchmark_result/RNN_result.md)

| Framework/Library | Version |
| :----------------- | :------- |
| Keras             | 2.1.5   |
| MXNet             | 1.1.0   |
| TensorFlow        | 1.7.0   |
| CUDA              | 9.0.176 |

#### 1. Synthetic Dataset

![lstm_Synthetic_128.png](benchmark_result/lstm_Synthetic_128.png)

| Instance    | GPUs | MXNet Backend<br />Speed/Epoch | TensorFlow Backend<br />Speed/Epoch |
| :----------- | :---- | :------------------------------ | :----------------------------------- |
| C5.18xLarge | 0    | 24s 485us/step                 | 14s 284us/step                      |
| P3.8xLarge  | 1    | 13s 261us/step                 | 12s 249us/step                      |
| P3.8xLarge  | 4    | 12s 240us/step                 | 21s 430us/step                      |

#### 2. Nietzsche Dataset

![lstm_Nietzsche_128.png](benchmark_result/lstm_Nietzsche_128.png)

| Instance    | GPUs | MXNet Backend<br />Speed/Epoch | TensorFlow Backend<br />Speed/Epoch |
| :----------- | :---- | :------------------------------ | :----------------------------------- |
| C5.18xLarge | 0    | 78s 389us/step                 | 55s 273us/step                      |
| P3.8xLarge  | 1    | 52s 262us/step                 | 51s 252us/step                      |
| P3.8xLarge  | 4    | 47s 235us/step                 | 87s 435us/step                      |

#### 3. WikiText-2 Dataset

![lstm_Wikitext2_128.png](benchmark_result/lstm_Wikitext2_128.png)

| Instance    | GPUs | MXNet Backend<br />Speed/Epoch | TensorFlow Backend<br />Speed/Epoch |
| :----------- | :---- | :------------------------------ | :----------------------------------- |
| C5.18xLarge | 0    | 1345s 398us/step               | 875s 259us/step                     |
| P3.8xLarge  | 1    | 868s 257us/step                | 817s 242us/step                     |
| P3.8xLarge  | 4    | 775s 229us/step                | 1468s 434us/step                    |
## Credits

Synthetic Data scripts modified from 
[TensorFlow Benchmarks](https://github.com/tensorflow/benchmarks/tree/keras-benchmarks)

## Reference
[1] [TensorFlow Benchmarks](https://github.com/tensorflow/benchmarks/tree/keras-benchmarks)
