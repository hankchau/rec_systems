# Recommender Systems for Sparse Data
Most recommender systems use collaborative filtering (CF), which relies on calculating similarity, to capture accurate user-item relations. With better computing power, and the rising popularity of machine learning, many researchers are applying deep neural networks (DNNs) to recommender systems. However, since raw data collected from the real world is often incomplete, sparsity is a huge and prevalent problem. CF methods and DNNs have both shown to perform poorly without sufficient data. 

Many methods, such as matrix factorization (MF) and L1 Regularization, have been proposed to solve the data sparsity problem. MF aims to find latent features that approximate the behavior for each user and each item. "Blank spaces", can then be filled by mapping each latent user-item pair to a rating value. By approximating this value on . I will only focus on MF here, and leave other methods for future exploration. 

The implementations of the models here are my own work, but the neural-network-based CF concepts (especially NeuMF) are based on *"Neural Collaborative Filtering"* by Xiangnan He, Lizi Liao et al, 2017. 
- Link to Paper: https://arxiv.org/pdf/1708.05031.pdf

## Dataset
The same dataset is used for the different models presented. For anonymity, I have only presented a fake table that captures customer to item relations in a binary format, where '1' records the existence of a relation between a customer-item pair, and '0' represents a blank space where no data is available. The matrix has a sparsity of 77%. 

Note that each '0' in the table simply corresponds to a **_lack of data_, and not a negative relationship**. To trian a neural network, we would need labeled training data in both classes ('1' and '0'). To account for training data with label '0', random negative sampling is used with varying ratios of one positive instance to few negative instances. Since negative samples are assumed, and there is an imbalance between positive and negative samples, class weights favoring positive samples can be incorporated in training.

## Metrics
Since we are mostly interested in positive predictions, accuracy no longer determines the model's performance. The important metrics to track are **Validation Loss, Precision, and Recall**. Validation loss helps prevent overfitting, and recall measures the portion of true positives the model predicts. 

A low precision is also expected and acceptable, since most of our training and validation sets are labeled as negative, and we wouldn't want to discourage our model in predicting '1's.

## General Matrix Factorization 
- GMF:
    - Neural network based approach.
    - Embedding layers to find latent features of dimension k for users and items.
    - Matrix dot product to map latent features to a probability P('1' | latent_features).
    - Adam with fast learn rate. Binary crossentropy as loss function.
    - Fragile, very sensitive to class weights.
    - Fast convergence. Fixed epochs. 
    - Low accuracy with low precision and very high recall.
    
## Multilayer Perceptrons 
- MLP:
    - Deep neural network based approach.
    - Embedding layers to find latent features of dimension k for users and items.
    - Multilayer perceptron to map latent features to a probability P('1' | latent_features).
    - Adam with slow learn rate, Binary crossentropy as loss function.
    - More robust than GMF. 
    - Slow convergence. Hard to find early stopping criteria. 
    - Decent accuracy with ~50% precision and decent recall. 
    
## Neural Matrix Factorization 
- NeuMF:
    - Combines MLP and GMF models into a hybrid neural network for recommendation.
    - Uses pretrained MLP and GMF parts for faster convergence. 
    - Weights non-linear mappings (MLP) and linear mappings (GMF) for better estimation.
    - Stochastic GD with slow learn rate. Binary crossentropy as loss function.
    - Robust. Behavior similar to MLP.
    - Slow convergence. Hard to find early stopping criteria.
    - Better precision, recall. 
    
## Collaborative Filtering
- Jaccard CF:
    - Jaccard similarity as metric for a KNN-based collaborative filtering model.
    - Compared to other sim. metrics, Jaccard is relatively insensitive to sparse data. 
    - Item-based CF. 
