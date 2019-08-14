import sys
import random
import pandas as pd

"""
Using all positive instances
"""

NEG_RATIO = 3

class DataParse:

    def get_test_data(self, testfile):
        print('\nreading testing data ...')

        df = pd.read_csv(testfile, sep=',')
        df = df.transpose()

        self.test_data = {'CST_ID': [], 'FND_ID': [], 'Rating': []}
        self.test_custs = {}

        fund_indx = -1

        for rindex, row in df.iterrows():
            for cindex, f in row.iteritems():
                # get customer ID
                if fund_indx == -1:
                    self.test_custs[cindex] = f
                    continue
                # get info
                self.test_data['CST_ID'].append(cindex)
                self.test_data['FND_ID'].append(fund_indx)
                self.test_data['Rating'].append(f)
            fund_indx += 1

        self.test_custs_len = len(self.test_custs)
        self.test_len = len(self.test_data['CST_ID'])

        print('testing data: ' + str(self.test_len))
        print('finished reading test samples\n')

    def get_train_data(self):
        self.get_negative_sample()

        self.train_data = {'CST_ID': [], 'FND_ID': [], 'Rating': []}
        self.train_data['CST_ID'] = self.pos_train['CST_ID'] + (self.neg_train['CST_ID'])
        self.train_data['FND_ID'] = self.pos_train['FND_ID'] + (self.neg_train['FND_ID'])
        self.train_data['Rating'] = self.pos_train['Rating'] + (self.neg_train['Rating'])

        print('finished generating training samples')

    def read_train_file(self, trainfile):
        print('reading training data ...')

        # customer rating matrix
        df = pd.read_csv(trainfile, sep=',')
        df = df.transpose()

        # positive samples
        self.pos_train = {'CST_ID': [], 'FND_ID': []}
        # negative samples
        self.neg_data = {'CST_ID': [], 'FND_ID': []}
        # index to customer dictionary
        self.train_custs = {}
        self.funds = {}

        fund_indx = -1
        # get data
        for rindex, row in df.iterrows():
            if not fund_indx == -1:
                self.funds[fund_indx] = rindex

            for cindex, f in row.iteritems():
                if fund_indx == -1:
                    self.train_custs[cindex] = f
                    continue
                if f == 1:
                    self.pos_train['CST_ID'].append(cindex)
                    self.pos_train['FND_ID'].append(fund_indx)
                elif f == 0:
                    self.neg_data['CST_ID'].append(cindex)
                    self.neg_data['FND_ID'].append(fund_indx)
            fund_indx += 1

        self.funds_len = len(self.funds)
        self.train_custs_len = len(self.train_custs)
        # get lengths
        self.neg_len = len(self.neg_data['CST_ID'])
        self.pos_len = len(self.pos_train['CST_ID'])
        # self.NEG_SAMPLE_RATIO = int(self.neg_len/self.pos_len)
        self.NEG_SAMPLE_RATIO = NEG_RATIO

        # get ratings
        self.pos_train['Rating'] = [1] * self.pos_len
        self.neg_data['Rating'] = [0] * self.neg_len

        return self.pos_train, self.neg_data

    # get negative instances from sampling
    def get_negative_sample(self):

        print('training data: ' + str(self.pos_len + self.neg_len) +
              '\n   positive: '+ str(self.pos_len) +
              '\n   negative: ' + str(self.neg_len) +
              '\n   sparsity: ' + str(self.neg_len/(self.neg_len + self.pos_len)) +
              '\n   negative sample ratio: ' + str(self.NEG_SAMPLE_RATIO))

        # negative instance index
        neg_trainIndx = random.sample(range(0, self.neg_len - 1),
                                      int(self.pos_len * self.NEG_SAMPLE_RATIO))
        self.neg_train = {'CST_ID': [], 'FND_ID': []}

        # negative instance samples
        for indx in neg_trainIndx:
            self.neg_train['CST_ID'].append(self.neg_data['CST_ID'][indx])
            self.neg_train['FND_ID'].append(self.neg_data['FND_ID'][indx])

        self.neg_train_len = len(self.neg_train['CST_ID'])
        # get ratings
        self.neg_train['Rating'] = [0] * self.neg_train_len

        print('training sampled: ' + str(self.pos_len + self.neg_train_len) +
              '\n   positive: ' + str(self.pos_len) +
              '\n   negative: ' + str(self.neg_train_len))


def main(trainfile, testfile):
    data = DataParse()
    data.read_train_file(trainfile)
    data.get_train_data()
    data.get_test_data(testfile)

    print('finished reading data')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
