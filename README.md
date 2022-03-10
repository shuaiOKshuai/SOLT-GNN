# On Size-Oriented Long-Tailed Graph Classification of Graph Neural Networks

We provide the code (in PyTorch) and datasets for our paper "[On Size-Oriented Long-Tailed Graph Classification of Graph Neural Networks](https://zemin-liu.github.io/papers/SOLT-GNN-WWW-22.pdf)" (SOLT-GNN for short), which is published in WWW-2022.


## 1. Descriptions
The repository is organised as follows:

- dataset/: the original data and sampled subgraphs of the five benchmark datasets.
- main.py: the main entry of tail graph classificaiton for SOLT-GIN.
- gin.py: base GIN model.
- PatternMemory.py: the module of pattern memory.
- utils.py: contains tool functions for loading the data and data split.
- subgraph_sample.py: contains codes for subgraph sampling.


## 2. Requirements

- Python-3.8.5
- Pytorch-1.8.1
- Networkx-2.4
- numpy-1.18.1


## 3. Running experiments

Our experimental environment is Ubuntu 20.04.1 LTS (GNU/Linux 5.8.0-55-generic x86_64), and we train our model using NVIDIA GeForce RTX 1080 GPU with CUDA 11.0.

(1) First run subgraph_sample.py to complete the step of subgraph sampling before running the main.py, and the sampled subgraph data will occupy a certain storage space:

- python subgraph_sample.py

(2) Tail graph classification with five different seeds(0,1,2,3,4) for each dataset with hyper-parameter settings in our experiments:

- python main.py --dataset PTC  --alpha 0.3 --mu1 1.5 --mu2 1.5
- python main.py --dataset PROTEINS  --alpha 0.15 --mu1 2 --mu2 2 
- python main.py --dataset DD    --alpha 0.1 --mu1 0.5 --mu2 0.5
- python main.py --dataset FRANK --alpha 0.1 --mu1 2 --mu2 0
- python main.py --dataset IMDBBINARY --alpha 0.15 --mu1 1 --mu2 1

Due to the different experimental environment, the experimental results will fluctuate when you use the codes. For reproducing our results in the paper, you need to tune the values of key hyper-parameters like $\mu_1,\mu_2, \alpha, d_m$,  in your experimental environment. The recommended search space of $\mu_1,\mu_2, \alpha, d_m$, are {0.1, 0.5, 1, 1.5, 2}, {0.1, 0.5, 1, 1.5, 2}, {0.05, 0.1, 0.15, 0.2, 0.25, 0.3}, {16, 32, 64, 128},  respectively. As the performance of the SOLT-GIN highly relates to GIN, so the tuning of hyper-parameters of GIN is encouraged. Please firstly fix the configuration of GIN before exploration of SOLT-GIN for efficiency.

## 4. Note
- The implementation of SOLT-GNN is based on the official implementation of GIN (https://github.com/weihua916/powerful-gnns).

## 5. Citation
