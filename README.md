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

### Experimental environment
Our experimental environment is Ubuntu 20.04.1 LTS (GNU/Linux 5.8.0-55-generic x86_64), and we train our model using NVIDIA GeForce RTX 1080 GPU with CUDA 11.0.

### How to run

(1) First run subgraph_sample.py to complete the step of subgraph sampling before running the main.py. Note that, the sampled subgraph data may occupy some storage space.

- python subgraph_sample.py

(2) Tail graph classification:

- python main.py --dataset PTC  --alpha 0.3 --mu1 1.5 --mu2 1.5
- python main.py --dataset PROTEINS  --alpha 0.15 --mu1 2 --mu2 2 
- python main.py --dataset DD    --alpha 0.1 --mu1 0.5 --mu2 0.5
- python main.py --dataset FRANK --alpha 0.1 --mu1 2 --mu2 0
- python main.py --dataset IMDBBINARY --alpha 0.15 --mu1 1 --mu2 1

### Note
- We repeat the experiments for five times and average the results for report (with standard deviation). Note that, for the five runs, we employ seeds {0, 1, 2, 3, 4} for parameters initialization, respectively.
- The change of experimental environment may result in performance fluctuation for both the baselines and our SOLT-GNN. To reproduce the results in the paper, please set the experimental environment as illustrated above as much as possible. The utilized parameter settings are illustrated in the python commands. Note that, for the possible case of SOLT-GNN performing a bit worse which originates from environment change, the readers can further tune the parameters, including $\mu_1$, $\mu_2$, $\alpha$ and $d_m$. In particular, for these four hyper-parameters, we recommend the authors to tune them in {0.1, 0.5, 1, 1.5, 2}, {0.1, 0.5, 1, 1.5, 2}, {0.05, 0.1, 0.15, 0.2, 0.25, 0.3}, {16, 32, 64, 128}, respectively. As the performance of SOLT-GIN highly relates to GIN, so the tuning of hyper-parameters for GIN is encouraged. When tuning the hyper-parameters for SOLT-GNN, please first fix the configuration of GIN for efficiency.
- To run the model on your own datasets, please refer to the following part (Input Data Format) for the dataset format.
- The implementation of SOLT-GNN is based on the official implementation of GIN (https://github.com/weihua916/powerful-gnns).
- To tune the other hyper-parameters, please refer to main.py for more details. In particular, for K


## 4. Input Data Format
In order to run SOLT-GNN on your own datasets, here we provide the input data format for SOLT-GNN as follows.

Each dataset XXX only contains one file, named as XXX.txt. Note that, in each dataset, we have a number of graphs. In particular, for each XXX.txt, 

- The first line only has one column, which is the number of graphs (marked as N) contained in this dataset; and the following part of this XXX.txt file is the data of each graph, including a total of N graphs.
- In the data of each graph, the first line has two columns, which denote the number of nodes (marked as n) in this graph and the label of this graph, respectively. Following this line, there are n lines, with the i-th line corresponding to the information of node i in this graph (index i starts from 0). In each of these n lines (n nodes), the first column is the node label, the second column is the number of its neighbors (marked as m), and the following m columns correspond to the indeces (ids) of its neighbors.
- Therefore, each graph has n+1 lines.


## 5. Cite

	@inproceedings{liu2022onsize,
	  title={On Size-Oriented Long-Tailed Graph Classification of Graph Neural Networks},
	  author={Liu, Zemin and Mao, Qiheng and Liu, Chenghao and Fang, Yuan and Sun, Jianling},
	  booktitle={Proceedings of the ACM Web Conference 2022},
	  year={2022}
	}

## 6. Contact
If you have any questions for the data and code, please contact Qiheng Mao (22021184@zju.edu.cn).
