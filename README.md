# LiteHAR
LiteHAR: Lightweight Human Activity Recognition from WiFi Signals with Random Convolution Kernels

Implementation of the LiteHAR model by Hojjat Salehinejad and Shahrokh Valaee. 

The corresponding paper has been accepted for presentation at IEEE ICASSP 2022. Paper on ArXiv. 


## Data
Here the link to the dataset used in the paper:
https://github.com/ermongroup/Wifi_Activity_Recognition


## Prerequisite
Python >= 3.6
numpy
pandas
scikit-learn
numba
joblib

## How to Run
Run the bash script provided as: ./runner.sh

## Parameters
Setup parameters in the runner.sh:

python3.6 main.py -m rigRocket -k 10000 -cv 1 -e 20 -i ../Dataset/Data/

where

- i: path to the data
- e: number of epochs (if necessary)
- m: model 
- k: number of kernels
- cv: number of cross-validation
