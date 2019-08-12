"""
Collaborative Filtering Recommender System
Using Jaccard Similarity

By Hank Chau

Item-Based Customer/Fund Recommendation
"""

import sys
import csv
import numpy as np
import pandas as pd

# num nearest neighbors
NUM_NEIGHBORS = 3

class CollabFilter:
    def __init__(self, num):
        print('initializing collaborative filter model')
        self.num_neighbors = num

    # read input data
    def read_csv(self, filename):
        df = pd.read_csv(filename, sep=',')

        # item - item
        df = df.transpose()

        # get column names ( excluding CST_ID )
        self.features = (df.iloc[0]).tolist()
        # get cst_ids
        users = df.index.values.tolist()
        self.users = users[1:]
        # numpy 2d array
        self.data = df.to_numpy()

    # compute jaccard similarity ( adj. matrix )
    def get_jaccard_similarity(self):
        feature_length = len(self.features)

        self.similarity = np.empty(shape=(len(self.users), len(self.users)), dtype='float')
        for u1 in range(len(self.users)):
            for u2 in range(len(self.users)-u1):
                u2 += u1
                union_count = np.count_nonzero(np.add(self.data[u1], self.data[u2]))

                intersect = np.multiply(self.data[u1], self.data[u2])
                try:
                    sim_val = np.count_nonzero(intersect) / union_count
                except ZeroDivisionError:
                    sim_val = 0.0

                if sim_val == 0:
                    sim_val = 1 / feature_length

                self.similarity[u1][u2] = sim_val
                self.similarity[u2][u1] = sim_val

    # get rating predictions
    def predict_purchases(self):
        self.predictions = {
            '1.0': [], '2.0': [], '3.0': [],
            '4.0': [], '5.0': [], '6.0': [],
            '7.0': [], '8.0': [], '9.0': [],
            '10.0': [], '11.0': [], '12.0': [],
            '13.0': []
        }

        for f in range(len(self.features)):
            target_users = np.where(self.data[1:,f] == 0)[0]
            # neighbors = np.flatnonzero(self.data[1:,f])

            for tu in target_users:
                user = self.users[tu]
                # get prediction
                prediction = self.predict(tu)
                if prediction == 1:
                    self.predictions[user].append(self.features[f])

    # calculate score nearest neighbor
    def get_closest_neighbors(self, target):
        # sort by desc
        sorted_indices= np.argsort(self.similarity[target])[::-1]
        # extract knn
        closest_neighbors = sorted_indices[1:self.num_neighbors + 1]

        # count num of 0s / 1s
        count0 = 0
        count1 = 0
        for cn in closest_neighbors:
            dt = self.data[target][cn]
            if dt == 0:
                count0 += 1
            else:
                count1 += 1

        return count0, count1

    def predict(self, target):
        score0, score1 = self.get_closest_neighbors(target)
        if score0 > score1:
            return 0
        else:
            return 1

    # evaluate scores
    def to_csv(self, filename):
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Fund_ID', 'CST_ID'])
            for key, value in self.predictions.items():
                line = [key] + value
                writer.writerow(line)

def main(infile, outfile):
    cf = CollabFilter(NUM_NEIGHBORS)
    # infile = 'path to input file'
    cf.read_csv(infile)

    cf.get_jaccard_similarity()
    cf.predict_purchases()

    # outfile = 'path to output file'
    cf.to_csv(outfile)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
