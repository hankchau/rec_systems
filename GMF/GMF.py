import sys
import numpy as np
import keras_metrics as km

import Data

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Embedding, Flatten, Dot, Multiply, Dense
from keras import Input, initializers, regularizers, callbacks

LATENT_DIM = 64

BATCH_SIZE = 256
LEARN_RATE = 0.0005
EPOCHS = 10

REG_SCALE = 0.01
ITERATIONS = 3

class GMF:
    def __init__(self):
        print('pre-training GMF ...')

    # default method uses dot product for combining layer
    def build_model(self, cust_dim, fund_dim, latent_dim, lr, dot=False, mul=False):
        print('building GMF model ...')

        # customer input embedding layer
        self.GMF_custInputLayer = Input(shape=(1,), dtype='float', name='GMF_cust_Input')
        cust_embedding = Embedding(
            trainable=True,
            input_dim=cust_dim,
            output_dim=latent_dim,
            embeddings_initializer=initializers.RandomNormal(mean=0, stddev=0.01),
            embeddings_regularizer=regularizers.l2(REG_SCALE),
            name='GMF_cust_Embedding'
        )
        # fund input embedding layer
        self.GMF_fundInputLayer = Input(shape=(1,), dtype='float', name='GMF_fund_Input')
        fund_embedding = Embedding(
            trainable=True,
            input_dim=fund_dim,
            output_dim=latent_dim,
            embeddings_initializer=initializers.RandomNormal(mean=0, stddev=0.01),
            embeddings_regularizer=regularizers.l2(REG_SCALE),
            name='GMF_fund_Embedding'
        )
        # Flatten input latent layers
        self.userLatentLayer = Flatten(name='GMF_cust_Latent')(cust_embedding(self.GMF_custInputLayer))
        self.fundLatentLayer = Flatten(name='GMF_fund_Latent')(fund_embedding(self.GMF_fundInputLayer))

        # check which combination to use
        if dot:
            self.dot = Dot(axes=-1, name='Dot_Layer')([self.userLatentLayer, self.fundLatentLayer])
            self.predictions = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(self.dot)
        elif mul:
            # element-wise multiplication
            self.mul = Multiply(name='Mul_Layer')([self.userLatentLayer, self.fundLatentLayer])
            self.predictions = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(self.mul)

        # build model
        self.gmf_model = Model(
            inputs=[self.GMF_custInputLayer, self.GMF_fundInputLayer],
            output=self.predictions
        )
        print('GMF Model Architecture: ')
        print(self.gmf_model.summary())

        print('compiling GMF model ...')
        # compile GMF model
        self.gmf_model.compile(
            optimizer=Adam(lr=lr),
            loss='binary_crossentropy',
            metrics=['acc', km.binary_precision(0), km.binary_recall(0)]
        )

    def fit(self, data, outpath, perm, batch_size=BATCH_SIZE, epochs=EPOCHS):
        cust_train = np.array(data.train_data['CST_ID'])[perm]
        fund_train = np.array(data.train_data['FND_ID'])[perm]
        pred_train = np.array(data.train_data['Rating'])[perm]

        print('fitting GMF model on train data ...')
        inputs = {
            'cust_Input': cust_train,
            'fund_Input': fund_train
        }

        self.gmf_model.fit(
            x=inputs,
            y=pred_train,
            validation_split=0.2,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3),
                callbacks.ModelCheckpoint(
                    filepath=(outpath + '/gmf_model_weights.hdf5'),
                    monitor='val_loss',
                    save_best_only=True, mode='min', period=1)
            ]
        )

    def evaluate(self, test_data, batch_size):
        print('\nevaluating GMF model on test data...')

        cust_test = np.array(test_data['CST_ID'])
        fund_test = np.array(test_data['FND_ID'])
        pred_test = np.array(test_data['Rating'])

        return self.gmf_model.evaluate(
            x=[cust_test, fund_test],
            y=pred_test,
            batch_size=batch_size
        )

    def predict(self, data):
        pred_custs = np.zeros(shape=(13,))
        pred_funds = np.arange(13)

        rate = self.gmf_model.predict(x=[pred_custs, pred_funds], batch_size=1)
        print(rate)


def main(datafile, outpath):
    # get data
    data = Data.DataParse()
    data.read_train_file(datafile)
    data.get_train_data()
    # get testing data for eval
    # data.get_test_data(testfile)
    perm = np.random.permutation(len(data.train_data['CST_ID']))

    gmf = GMF()
    gmf.build_model(
        cust_dim=data.train_custs_len,
        fund_dim=data.funds_len,
        latent_dim=LATENT_DIM,
        lr=LEARN_RATE,
        mul=True
    )

    for i in range(ITERATIONS):
        gmf.fit(data, outpath=outpath, perm=perm, batch_size=BATCH_SIZE, epochs=EPOCHS)
        data.get_train_data()
    # res = gmf.evaluate(data.test_data, BATCH_SIZE)

    gmf.predict(data)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
