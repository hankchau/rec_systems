# Recommender Systems for Sparse Data
- Most recommender systems use collaborative filtering (CF), which relies on calculating similarity, to capture accurate user-item relations. With better computing power, and the rising popularity of machine learning, many researchers are applying deep neural networks (DNNs) to recommender systems. However, since raw data collected from the real world is often incomplete, sparsity is a huge and prevalent problem. CF methods and DNNs have both shown to perform poorly without sufficient data. 

- Many methods, such as matrix factorization (MF) and L1 Regularization, have been proposed to solve the data sparsity problem. MF aims to find latent features that approximate the behavior for each user and each item. "Blank spaces", can then be filled by mapping each latent user-item pair. I will only focus on MF here, and leave other methods for future exploration. 

## Dataset
- The same dataset is used for the different models presented. For anonymity, the dataset presented captures customer-to-hedgefund relationships in a binary format.

## General Matrix Factorization 
- GMF:
    - Neural network based approach.
    - Embedding layers to find latent features for users and items.
    - Matrix dot product to find latent features of dimension k for recommendation.
    - Fragile, very sensitive to class weights.
    - Fast convergence. 
    - Low accuracy with low precision and very high recall.
## Multilayer Perceptrons 
- MLP:
    - Deep neural network based approach.
    - Embedding layers to find latent features for users and items.
    - Multilayer perceptron to find latent features of dimension k for recommendation.
    - More robust than GMF. 
    - Slow convergence. Hard to find early stopping criteria. 
    - Decent accuracy with ~50% precision and decent recall. 
## Neural Matrix Factorization 
- NeuMF:
    - Combines MLP and GMF models into a hybrid neural network for recommendation.
    - Weights non-linear mappings (DNN in MLP) and linear mappings (dot product in GMF) for better estimation.
    - Model concept proposed in *Neural Collaborative Filtering* by Xiangnan He, Lizi Liao, et al, 2017. 
    - Link to Paper: https://arxiv.org/pdf/1708.05031.pdf

## Collaborative Filtering
- Jaccard CF:
    - Jaccard similarity as metric for a KNN-based collaborative filtering model.
    - Compared to other sim. metrics, Jaccard is relatively insensitive to sparse data. 
    - Item-based CF. 
