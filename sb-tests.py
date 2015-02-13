import csv
import gzip
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import train_test_split
from sklearn import neighbors

train_filename = 'train.csv.gz'
test_filename  = 'test.csv.gz'
pred_filename  = 'example_mean.csv'
samples = 5000

def pred_avg(arr, datum):
    total_pred = 0
    for model in arr:
        pred = model.predict(datum)
        if isinstance(pred, list):
            pred = pred[0]
        total_pred += pred
    return total_pred/len(arr)

def build_svm(train_data):
    # X_train, X_test, y_train, y_test = train_test_split(train_data['features'], train_data['gap'], test_size=0.8)
    # clf = svm.SVC(gamma = 0.001)
    # clf.fit(train_data['features'], train_data['gap'])
    # return clf
    #####################
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    # ############################
    # y_rbf = svr_rbf.fit(train_data['features'], train_data['gap'])
    # y_lin = svr_lin.fit(train_data['features'], train_data['gap'])
    # y_poly = svr_poly.fit(train_data['features'], train_data['gap'])
    # #############################
    r_clf = Ridge(alpha = 1.0)
    l_clf = Lasso(alpha=0.001)
    en_clf = ElasticNet(alpha=0.0001, l1_ratio = 0.7)
    r_clf.fit(train_data['features'], train_data['gap'])
    # l_clf.fit(train_data['features'], train_data['gap'])
    # en_clf.fit(train_data['features'], train_data['gap'])
    ##########################

    num_neighbors = 10
    knn = neighbors.KNeighborsRegressor(num_neighbors, weights='uniform')
    # y_knn = knn.fit(train_data['features'], train_data['gap'])

    return [r_clf]

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
        # if (i > samples):
        #     break

        # if (i % 10 == 0):
        #     print "Finished ", i, "trainings"
        smiles   = row[0]
        # features1 = np.array([float(x) for x in row[1:257]])
        features2 = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),2,nBits=4096))
        gap      = float(row[257])
        if (i < samples):
            train_data.append({ 'smiles':   smiles,
                            'features': features2,
                            'gap':      gap })
        else:
            train_data2['smiles'].append(smiles)
            train_data2['features'].append(features2)
            train_data2['gap'].append(gap)  

        

        if (i > samples*2):
            break 

# # Compute the mean of the gaps in the training data.
# gaps = np.array([datum['gap'] for datum in train_data])
# mean_gap = np.mean(gaps)

# Load the test file.
# test_data = []
# test_data2 = []
# with gzip.open(test_filename, 'r') as test_fh:

#     # Parse it as a CSV file.
#     test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    
#     # Skip the header row.
#     next(test_csv, None)

#     # Load the data.
#     for i, row in enumerate(test_csv):
#         if (i > samples):
#             break
#         # if (i % 10 == 0):
#         #     print "Finished ", i, "tests"
#         id       = row[0]
#         smiles   = row[1]
#         features = np.array([float(x) for x in row[2:258]])
        
#         test_data.append({ 'id':       id,
#                            'smiles':   smiles,
#                            'features': features })
        # test_data2['id'].append(id)
        # test_data2['smiles'].append(smiles)
        # test_data2['features'].append(features)

# Write a prediction file.
with open(pred_filename, 'w') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')

    # Write the header row.
    pred_csv.writerow(['Id', 'Prediction'])

    pred_arr = build_svm(train_data2)

    # for datum in test_data:
    #     # pred_csv.writerow([datum['id'], mean_gap])
    #     pred_csv.writerow([datum['id'], pred_avg(pred_arr, datum['features'])[0]])

    rmse = 0
    for datum in train_data:
        rmse += (pred_avg(pred_arr, datum['features']) - datum['gap'])**2

    rmse = np.sqrt(rmse/samples)
    print "LOL gg: ", rmse



