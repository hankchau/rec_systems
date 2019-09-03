# Recommender Systems for Sparse Data
Most recommender systems use collaborative filtering (CF), which relies on calculating similarity, to capture accurate user-item relations. With better computing power, and the rising popularity of machine learning, many researchers are applying deep neural networks (DNNs) to recommender systems. However, since raw data collected from the real world is often incomplete, sparsity is a huge and prevalent problem. CF methods and DNNs have both shown to perform poorly without sufficient data. 

Many methods, such as matrix factorization (MF) and L1 Regularization, have been proposed to solve the data sparsity problem. MF aims to find latent features that approximate the behavior for each user and each item. "Blank spaces", or missing data, can then be filled by mapping each latent user-item pair to a rating value. By approximating this value, a recommender can provide accurate suggestions to corresponding users. The methods on this page will only focus on MF here, and leave other methods for future exploration. 

The implementations of the models here are my own work, but the neural-network-based CF concepts (especially NeuMF) are based on *"Neural Collaborative Filtering"* by Xiangnan He, Lizi Liao et al, 2017. 
- Link to Paper: https://arxiv.org/pdf/1708.05031.pdf

## Dataset
The same dataset is used for the different models presented. The dataset is presented as a table that captures customer to item relations in a binary format, where '1' records the existence of a relation between a customer-item pair, and '0' represents a blank space where no data is available. The matrix has a sparsity of 99.5%. For anonymity, I have not presented a sample for the dataset. I will consider putting up a fake dataset in the same format for visualization in the future. 

Note that each '0' in the table simply corresponds to a **_lack of data_, and not a negative relationship**. To trian a neural network, we would need labeled training data in both classes ('1' and '0'). To account for training data with label '0', random negative sampling is used with varying ratios of one positive instance to a few negative instances. Since negative samples are assumed, and there is an imbalance between positive and negative samples, class weights favoring positive samples can be incorporated in training.

## Metrics
Since we are only interested in positive interactions ('1's), we will not be using accuracy as am important measurement. Accuracy will only be used to determine whether the model is learning at all. Instead **Precision** and **Recall** (hit rate and capture rate) will be used to measure our model's performance. All NN models are set up with binary crossentropy as loss function and Early Stopping using "val_loss".


## General Matrix Factorization 
- GMF:
    - Neural network based approach.
    - Embedding layers to find latent features of dimension k for users and items.
    - Matrix dot product to map latent features to a probability P('1' | latent_features).
    - Adam with fast learn rate. Binary crossentropy as loss function.
    - Fragile, very sensitive to class weights.
    - Fast convergence. Fixed epochs. 
    - Low precision (25%) and very high recall (100%), if class weights are used. 
    - "Shotgun approach". 
    
## Multilayer Perceptrons 
- MLP:
    - Deep neural network based approach.
    - Embedding layers to find latent features of dimension k for users and items.
    - Multilayer perceptron to map latent features to a probability P('1' | latent_features).
    - Adam with slow learn rate, Binary crossentropy as loss function.
    - More robust than GMF. 
    - Slow convergence. Hard to find early stopping criteria. 
    - Decent precision (~50%) and bad recall (~45%). 
    
## Neural Matrix Factorization 
- NeuMF:
    - Combines MLP and GMF models into a hybrid neural network for recommendation.
    - Uses pretrained MLP and GMF parts for faster convergence. 
    - Weights non-linear mappings (MLP) and linear mappings (GMF) for better estimation.
    - Stochastic GD with slow learn rate. Binary crossentropy as loss function.
    - Robust. Behavior similar to MLP.
    - Slow convergence. Hard to find early stopping criteria.
    - Better precision (~80%), better recall (~60%). 
    
## Collaborative Filtering
- Jaccard CF:
    - Jaccard similarity as metric for a KNN-based collaborative filtering model.
    - Compared to other sim. metrics, Jaccard is relatively insensitive to sparse data. 
    - Item-based CF to save runtime when computing similarity. (There are 4180 items and > 60,000 customers)
    
## Auto Encoders
- Autoencoder:
    - Unsupervised Learning.
    - Encoder for latent feature extraction.
    - Decoder for reconstruction from latent features.
    - User-based feature extraction. 
    - Results TBD.
    
## Deep and Wide NN:
- TBD.
