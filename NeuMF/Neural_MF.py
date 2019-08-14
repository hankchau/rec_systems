import sys
import csv
import numpy as np
import keras_metrics as km

import Data
import GMF
import Neural_CF as MLP

from keras import callbacks
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Concatenate, Dense

# weighting for MLP
ALPHA = 0.5
LEARN_RATE = 0.0001

BATCH_SIZE = 256
EPOCHS = 10
ITERATIONS = 3


class Recommender:

    def __init__(self):
        print('building Neural MF model ...')

    def build_model(self, pretrain):

        # setting up gmf
        gmf = pretrain.gmf
        gmf_weights = gmf.gmf_model.layers[-1].get_weights()
        gmf_weights = [w * (1 - ALPHA) for w in gmf_weights]
        gmf.gmf_model.layers.pop()

        # setting up mlp
        mlp = pretrain.ncf
        mlp_weights = mlp.ncf_model.layers[-1].get_weights()
        mlp_weights = [w * ALPHA for w in mlp_weights]
        mlp.ncf_model.layers.pop()

        predict_weights = np.concatenate((gmf_weights[0], mlp_weights[0]), axis=0)
        predict_biases = gmf_weights[1] + gmf_weights[1]

        # build model concat
        self.NeuMFLayer = Concatenate(axis=-1, name='NeuMF_Layer')([gmf.mul, mlp.hiddenLayer3])
        self.predictLayer = Dense(1, activation='sigmoid', name='Prediction')(self.NeuMFLayer)

        # build
        self.NeuMF = Model(
            inputs=[gmf.GMF_custInputLayer, gmf.GMF_fundInputLayer,
                    mlp.MLP_custInputLayer, mlp.MLP_fundInputLayer],
            outputs=[self.predictLayer]
        )
        self.NeuMF.get_layer('Prediction').set_weights([predict_weights, predict_biases])

        self.NeuMF.compile(
            optimizer=SGD(lr=LEARN_RATE),
            loss='binary_crossentropy',
            metrics=['acc', km.binary_precision(0), km.binary_recall(0)]
        )
        print('compiling NeuMF Model ...')
        print(self.NeuMF.summary())

    def fit(self, data, outpath, perm, batch_size=BATCH_SIZE, epochs=EPOCHS):
        cust_train = np.array(data.train_data['CST_ID'])[perm]
        fund_train = np.array(data.train_data['FND_ID'])[perm]
        pred_train = np.array(data.train_data['Rating'])[perm]

        print('fitting NeuMF model on train data ...')
        inputs = {
            'GMF_cust_Input': cust_train,
            'GMF_fund_Input': fund_train,
            'MLP_cust_Input': cust_train,
            'MLP_fund_Input': fund_train
        }

        self.history = self.NeuMF.fit(
            x=inputs,
            y=pred_train,
            validation_split=0.2,
            class_weight={1: 0.75, 0: 0.25},
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss', mode='min', patience=3),
                callbacks.ModelCheckpoint(
                    filepath=(outpath + '/NeuMF_model_save.hdf5'),
                    monitor='val_loss',
                    save_best_only=True, mode='min', period=1)
            ]
        )

        # self.NeuMF.save_weights(outpath + '/NeuMF_model_weights.hdf5')

        return self.history

    def predict(self, data):
        pred_custs = data.neg_data['CST_ID']
        pred_funds = data.neg_data['FND_ID']

        self.rates = self.NeuMF.predict(
            x=[np.array(pred_custs), np.array(pred_funds)],
            batch_size=256
        )

        self.predictions = {}
        for indx in range(len(pred_custs)):
            custID = data.train_custs[pred_custs[indx]]
            fundID = data.funds[pred_funds[indx]]
            if custID not in self.predictions:
                self.predictions[custID] = []

            rate = self.rates[indx]
            # condition
            if rate >= 0.5:
                    self.predictions[custID].append(fundID)

    def save_predict(self, outpath):
        outpath += '/NeuMF_predictions.csv'
        with open(outpath, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['CST_ID', 'Fund_ID'])

            for key, value in self.predictions.items():
                line = [key] + value
                writer.writerow(line)


class PreTrainer:
    def __init__(self, data):
        print('preparing for pre-train ...')
        print('using default hyperparameters ...')
        self.build_MLP(data)
        self.build_GMF(data)

    # pretrain entire model
    def get_pretrain(self, data, outpath, perm):
        self.train_MLP(data, outpath)
        self.train_GMF(data, outpath)
        self.perm = perm

    def load_weights(self, mlp_file, gmf_file):
        print('loading pretrained model weights ...')

        self.ncf.ncf_model.load_weights(mlp_file)
        self.gmf.gmf_model.load_weights(gmf_file)

    def build_MLP(self, data):
        print('pretraining MLP model ...')
        self.ncf = MLP.NeuralCF()

        self.ncf.build_model(
            cust_dim=data.train_custs_len,
            fund_dim=data.funds_len,
            latent_dim=MLP.LATENT_DIM,
            lr=MLP.LEARN_RATE
        )

    def build_GMF(self, data):
        print('pretraining GMF model')
        self.gmf = GMF.GMF()

        self.gmf.build_model(
            cust_dim=data.train_custs_len,
            fund_dim=data.funds_len,
            latent_dim=GMF.LATENT_DIM,
            lr=GMF.LEARN_RATE,
            mul=True
        )

    def train_MLP(self, data, outpath, iteration=MLP.ITERATIONS):
        for i in range(iteration):
            self.ncf.fit(data=data, outpath=outpath, perm=self.perm)
            data.get_train_data()

    def train_GMF(self, data, outpath, iteration=GMF.ITERATIONS):
        for i in range(iteration):
            self.gmf.fit(data=data, outpath=outpath, perm=self.perm)
            data.get_train_data()


def main(datafile, outpath, gmf_weights=None, mlp_weights=None):
    data = Data.DataParse()
    data.read_train_file(datafile)
    data.get_train_data()

    pt = PreTrainer(data)
    perm = np.random.permutation(len(data.train_data['CST_ID']))

    # load pretrained model
    if len(sys.argv) > 3:
        pt.load_weights(mlp_file=mlp_weights, gmf_file=gmf_weights)
    else:
        pt.get_pretrain(data, outpath, perm)

    rc = Recommender()
    rc.build_model(pretrain=pt)

    for i in range(ITERATIONS):
        rc.fit(data, outpath, perm=perm, batch_size=BATCH_SIZE, epochs=EPOCHS)
        data.get_train_data()

    rc.predict(data)
    rc.save_predict(outpath)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], gmf_weights=sys.argv[3], mlp_weights=sys.argv[4])
