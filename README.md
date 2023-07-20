# Dynamic Graph Evolution Learning for Recommendation
This repository contains PyTorch codes and datasets for the paper:
>Haoran Tang, Shiqing Wu, Guandong Xu, and Qing Li. 2023. Dynamic Graph Evolution Learning for Recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23). Association for Computing Machinery, New York, NY, USA, 1589–1598. https://doi.org/10.1145/3539618.3591674.

## Introduction
Dynamic Graph Evolution Learning (DEGL) aims to solve the graph-based real-time evolution problem for multi-round recommendation scenario by three efficient learning modules and also bridges the normalization with model learning for better capturing the dynamics.

## Citation
If you want to use our codes or refer our paper, please cite it:
```
@inproceedings{10.1145/3539618.3591674,
author = {Tang, Haoran and Wu, Shiqing and Xu, Guandong and Li, Qing},
title = {Dynamic Graph Evolution Learning for Recommendation},
year = {2023},
isbn = {9781450394086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539618.3591674},
doi = {10.1145/3539618.3591674},
pages = {1589–1598},
numpages = {10},
location = {Taipei, Taiwan},
series = {SIGIR '23}
}
```

## Datasets
We utilize three public datasets: Wikipedia, Reddit and Foursquare.
* The Wikipedia and Reddit dataset
<br> https://snap.stanford.edu/jodie/#datasets
* The Foursquare dataset
<br> https://sites.google.com/site/yangdingqi/home/foursquare-dataset

## Run the codes
Please unzip the datasets first. Also you need to create the `saved_models/` and the `results/` directories.
### Train model
To train DGEL, please run the following basic commands:
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset wikipedia --embedding_dim 32 --sample_length 100 --epochs 30
```
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset reddit --embedding_dim 64 --sample_length 150 --epochs 30
```
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset foursquare --embedding_dim 32 --sample_length 200 --epochs 30
```
`--dataset:` indicating the data name you select
<br>`--embedding_dim:` diminsion of each dynamic sub-embedding
<br>`--sample_length:` the number of neighbors
<br>`--epochs`: the number of training epochs
<br>`Note:` There are some other hyper-parameters in codes but we have set the optimal values.
### Evaluate model
To evaluate DGEL, please run the following basic commands:
```
CUDA_VISIBLE_DEVICES=0 python evaluate_all.py --dataset wikipedia --embedding_dim 32 --sample_length 100 --epochs 30
```
```
CUDA_VISIBLE_DEVICES=0 python evaluate_all.py --dataset reddit --embedding_dim 64 --sample_length 150 --epochs 30
```
```
CUDA_VISIBLE_DEVICES=0 python evaluate_all.py --dataset foursquare --embedding_dim 32 --sample_length 200 --epochs 30
```

## Acknowledgements
This research is supported by the dual degree PhD program from the Department of Computing at the Hong Kong Polytechnic University (PolyU) & the School of Computer Science at the University of Technology Sydney (UTS).
