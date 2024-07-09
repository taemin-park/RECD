This file provides instructions for implementing the source code of the paper entitled 'Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch'.
-------------------------------------------------------------------------------------------------------------------------------------------------------
Dependencies

python3.8 

torch 1.8.1 (python3.8 -m pip install torch==1.8.1)
torchvision 0.9.1 (python3.8 -m pip install torchvision==0.9.1)
numpy 1.20.3 (python3.8 -m pip install numpy==1.20.3)
randAugment (python3.8 -m pip install git+https://github.com/ildoonet/pytorch-randaugment)
matplotlib (python3.8 -m pip install matplotlib)
progress (python3.8 -m pip install progress)

-------------------------------------------------------------------------------------------------------------------------------------------------------

If you want to run RECD_fix.py with 
0th gpu, imbalance ratio of labeled data 100, imbalance ratio of unlabeled data 1,  Number of samples in the maximal class of labeled data 1500, Number of samples in the maximal class of unlabeled data 3000, 
500 epochs with each epoch 500 iterations, manualseed as 0, dataset as CIFAR-10:

python3.8 RECD_fix.py --gpu 0 --imb_ratio 100 --imb_ratio_u 1 --num_max 1500 --num_max_u 3000 --epochs 500 --val-iteration 500 --manualSeed 0 --dataset cifar10


If you want to run RECD_remix.py with 
1st gpu , imbalance ratio of labeled data 20, Number of samples in the maximal class of labeled data 450, 
500 epochs with each epoch 500 iterations, manualseed as 0, dataset as STL-10:

python3.8 RECD_remix.py --gpu 1 --imb_ratio 20 --num_max 450 --epochs 500 --val-iteration 500 --manualSeed 0 --dataset stl10

-------------------------------------------------------------------------------------------------------------------------------------------------------

These codes measure classification peformance of RECD on testset after each epoch of training.
After 500 epochs of training, codes show the average performane of the last 20 epochs. 

-------------------------------------------------------------------------------------------------------------------------------------------------------

Section 5 of the main paper presents the performances of RECD. 