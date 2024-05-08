# eve_training

Here you can find trainings scripts for autonomous endovascular controllers trained on [eve_bench](https://github.com/lkarstensen/eve_bench) environments using the [eve_rl](https://github.com/lkarstensen/eve_rl) framework. 

You can find the scripts in the folder *training_scripts*. Currently it contains the scripts utilized to create the results from **insert paper**. 

## Getting Started

1. Clone the repo including the submodules:
```
git clone -recurse-submodules https://github.com/lkarstensen/eve_training.git
```
2. Install [SOFA](https://www.sofa-framework.org) (e.g. with the [EVE instructions](https://github.com/lkarstensen/eve?tab=readme-ov-file#install-sofa-with-sofapython3-and-beamadapter)) 
3. Install and test [eve](https://github.com/lkarstensen/eve)
```
python3 -m pip install -e ./eve
python3 ./eve/examples/function_check.py
```
4. Install and test [eve_bench](https://github.com/lkarstensen/eve_bench)
```
python3 -m pip install -e ./eve_bench
python3 ./eve_bench/examples/function_check.py
```
5. Install and test [eve_rl](https://github.com/lkarstensen/eve_rl)
```
python3 -m pip install -e ./eve_rl
python3 ./eve_rl/examples/function_check.py
```
6. Now you are ready to use this framework and start one of the training scripts. 

## How to use

Best way to start the training scripts is via console. You have to give the trainer device, amount of workers, learning rate and neural network structure as arguments. Training device and amount of workers depend on your available hardware. Additionally you can give each training a individual name for logging. 

Here are examples to recreate the results from **insert paper**. Here we have an input embedder of 1 LSTM-layer with 500 nodes and policy- and q-networks with the structur [900 900 900 900]. In this example we train with 29 workers and training on a cuda GPU. You should leva 2-3 of your available threads for other tasks than workers. 

Start your script and wait 1-2 days, depending on your setup. The scripts will create a *results* folder in your current working directory and log the training progress and save checkpoints for each evaluation. 

### BasicWireNav

```
python3 ./training_scripts/BasicWireNav_train.py -d cuda -nw 29 -lr 0.00021989352630306626 --hidden 900 900 900 900 -en 500 -el 1 -n BasicWireNav
```

### ArchVariety

```
python3 ./training_scripts/ArchVariety_train.py -d cuda -nw 29 -lr 0.0003218 --hidden 400 400 400 -en 900 -el 1 -n ArchVariety
```

### DualDeviceNav

```
python3 ./training_scripts/DualDeviceNav_train.py -d cuda -nw 29 -lr 0.00021989352630306626 --hidden 900 900 900 900 -en 500 -el 1 -n DualDeviceNav
```

### Hyperparameter optimization

Additionally we have implemented a script to optimize the reinforcement learning hyperparameters:

```
python3 ./training_scripts/ArchVariety_optimize.py -d cuda -nw 29 -n Hyperparameter_opti
```

Start this script and wait 1-2 weeks for enough trainings to be performed for good results. This script will optimize the learning rate, amount of hidden layers, size of hidden layers, amount of input embedder layers and size of input embedder layers. 

## Best practice

A good practice is to run the training in a docker container. This enables easy rollout on several machines, as well as easy pausing and stopping of the container. 

You can find a dockerfile in this repo. It will start from a cuda enabled ubuntu, install SOFA dependencies, pull and configure SOFA, copy this folder and install eve, eve_bench, eve_rl and eve_training. 

You have to come up with a <tag> for the docker image.

1. Build the docker image:
```
docker buildx build --platform=linux/amd64 -t <tag> -f ./dockerfile .
```

2. (optionally) Push image to a docker repository:
```
docker push <tag>
```

3. (optionally) Pull image from a docker repository
```
docker pull <tag>
```

4. Run the docker image (first create a *results* folder <path/to/results>):
```
docker run --gpus all --mount type=bind,source=<path/to/results>,target=/opt/eve_training/results --shm-size 15G -d <tag> python3 ./training_scripts/BasicWireNav_train.py -d cuda -nw 29 -lr 0.00021989352630306626 --hidden 900 900 900 900 -en 500 -el 1 -n BasicWireNav
```

This will run the training in a docker container and save training results to the *results* folder. 
