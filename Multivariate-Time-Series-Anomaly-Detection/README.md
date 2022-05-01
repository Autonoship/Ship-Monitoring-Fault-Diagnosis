# Multivariate Time Series Anomaly Detection based on SMIS datasets in Pytorch

## Introduction
This repository replicates the result in [MTAD-GAT](https://github.com/ML4ITS/mtad-gat-pytorch). 

## Data Preprocessing

run preprocessing.ipynb

## Training

- Training ASC SIMS datasets for 10 epochs, using standard GAT instead of GATv2 (which is the default), and a validation split of 0.2:
```bash 
python train.py --dataset asc --epochs 10 --use_gatv2 False --val_split 0.2
```

See more training details in [MTAD-GAT](https://github.com/ML4ITS/mtad-gat-pytorch).

## Output and visualization results

```result_visualizer.ipynb``` provides a jupyter notebook for visualizing results. 
To launch notebook:
```bash 
jupyter notebook result_visualizer.ipynb
```












