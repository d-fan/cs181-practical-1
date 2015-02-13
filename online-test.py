import csv
import gzip
import numpy as np
# from sklearn import svm
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn import neighbors

TRAIN_SIZE = 100000

train_filename = 'train.csv.gz'
test_filename  = 'test.csv.gz'
pred_filename  = 'example_mean.csv'

def build_svm(train_data):
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    y_rbf = svr_rbf.fit(train_data['features'], train_data['gap'])
    return y_rbf

# Load the training file.
train_data2 = {'smiles': [], 'features': [], 'gap': []}
with gzip.open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(train_csv, None)

    # Load the data.
    i = 0
    for row in train_csv:
        i += 1
        if (i % 1000 == 0):
            print i
        if i >= TRAIN_SIZE:
            break
        #smiles   = row[0]
        features = np.array([int(float(x)) for x in row[1:257]])
        gap      = float(row[257])
        
#        train_data2['smiles'].append(smiles)
        train_data2['features'].append(features)
        train_data2['gap'].append(gap)   

# Compute the mean of the gaps in the training data.

# Load the test file.
"""
test_data = []
test_data2 = []
with gzip.open(test_filename, 'r') as test_fh:

    # Parse it as a CSV file.
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(test_csv, None)

    # Load the data.
    for i, row in enumerate(test_csv):
        # if (i > 1000):
        #     break
        # if (i % 10 == 0):
        #     print "Finished ", i, "tests"
        id       = row[0]
        smiles   = row[1]
        features = np.array([float(x) for x in row[2:258]])
        
        test_data.append({ 'id':       id,
                           'smiles':   smiles,
                           'features': features })
        # test_data2['id'].append(id)
        # test_data2['smiles'].append(smiles)
        # test_data2['features'].append(features)
"""

# Write a prediction file.
with open(pred_filename, 'w') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')

    # Write the header row.
    pred_csv.writerow(['Id', 'Prediction'])

    print "Building svm"
    rbf = build_svm(train_data2)

    # for datum in test_data:
    #     # pred_csv.writerow([datum['id'], mean_gap])
    #     pred_csv.writerow([datum['id'], rbf.predict(datum['features'])[0]])

    print "Calculating RMSE"
    rmse = 0
    for i in xrange(TRAIN_SIZE):
        rmse += (rbf.predict(test_data2['features'])[0] - test_data2['gap'])**2

    rmse = np.sqrt(rmse/TRAIN_SIZE)
    print "LOL gg: ", rmse

# def predict(test_data):
#     predictions = []
#     for datum in test_data:
#         predictions.append({'id': datum['id'],
#                             'pred': })
#     return predictions

# def predict(clf, features):
#     clf.predict(features)

