import csv
import gzip
import numpy as np
# from sklearn import svm
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn import neighbors

train_filename = 'train.csv.gz'
test_filename  = 'test.csv.gz'
pred_filename  = 'example_mean.csv'

def build_svm(train_data):
    # clf = svm.SVC(gamma = 0.001)
    # clf.fit(train_data['features'], train_data['gap'])
    # return clf
    #####################
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    # ############################
    y_rbf = svr_rbf.fit(train_data['features'], train_data['gap'])
    y_lin = svr_lin.fit(train_data['features'], train_data['gap'])
    y_poly = svr_poly.fit(train_data['features'], train_data['gap'])
    # #############################
    # clf = Ridge(alpha=1.0)
    # clf.fit(train_data['features'], train_data['gap'])
    ##########################

    # num_neighbors = 10
    # knn = neighbors.KNeighborsRegressor(num_neighbors, weights='uniform')
    # y_ = knn.fit(train_data['features'], train_data['gap'])

    return y_rbf

# Load the training file.
train_data = []
train_data2 = {'smiles': [], 'features': [], 'gap': []}
with gzip.open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(train_csv, None)

    # Load the data.
    for i, row in enumerate(train_csv):
        if (i > 1000):
            break
        smiles   = row[0]
        features = np.array([float(x) for x in row[1:257]])
        gap      = float(row[257])
        
        train_data.append({ 'smiles':   smiles,
                            'features': features,
                            'gap':      gap })
        train_data2['smiles'].append(smiles)
        train_data2['features'].append(features)
        train_data2['gap'].append(gap)   

# Compute the mean of the gaps in the training data.
gaps = np.array([datum['gap'] for datum in train_data])
mean_gap = np.mean(gaps)

# Load the test file.
test_data = []
test_data2 = []
with gzip.open(test_filename, 'r') as test_fh:

    # Parse it as a CSV file.
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(test_csv, None)

    # Load the data.
    for i, row in enumerate(test_csv):
        if (i > 1000):
            break
        id       = row[0]
        smiles   = row[1]
        features = np.array([float(x) for x in row[2:258]])
        
        test_data.append({ 'id':       id,
                           'smiles':   smiles,
                           'features': features })
        # test_data2['id'].append(id)
        # test_data2['smiles'].append(smiles)
        # test_data2['features'].append(features)

# Write a prediction file.
with open(pred_filename, 'w') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')

    # Write the header row.
    pred_csv.writerow(['Id', 'Prediction'])

    rbf = build_svm(train_data2)

    # for datum in test_data:
    #     # pred_csv.writerow([datum['id'], mean_gap])
    #     pred_csv.writerow([datum['id'], rbf.predict(datum['features'])[0]])

    rmse = 0
    for datum in train_data:
        rmse += (rbf.predict(datum['features'])[0] - datum['gap'])**2

    rmse = np.sqrt(rmse/1000)
    print "LOL gg: ", rmse

# def predict(test_data):
#     predictions = []
#     for datum in test_data:
#         predictions.append({'id': datum['id'],
#                             'pred': })
#     return predictions

# def predict(clf, features):
#     clf.predict(features)

