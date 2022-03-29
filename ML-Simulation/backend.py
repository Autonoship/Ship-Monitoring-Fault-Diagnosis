import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import dump
from sklearn import preprocessing
import copy
from joblib import load

# we need to normilize our paramenters based on the below ranges
parameter_range = {
    'GAS OUT': [10, 600],
    'SCAV AIR': [20, 80],
    'LOAD': [0, 100],
    'RPM': [-70, 130]
}

# Load directory paths for persisting model

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)

def min_max_normalize(df):
    normalizad_df = df.copy()
    for col in normalizad_df.columns:
        for key in parameter_range.keys():
            if key in col:
                normalizad_df[col] = (normalizad_df[col] - parameter_range[key][0]) / (parameter_range[key][1] - parameter_range[key][0])
                break
    return normalizad_df

def train():
    # Load, read and normalize training data
    training = "./train_ASC.csv"
    data_train = pd.read_csv(training)
    y_train = data_train['label'].values
    X_train = data_train.drop('label', axis = 1)

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)
        
    # Data normalization -> (0,1)
    X_train = min_max_normalize(X_train)
    # X_train.to_csv('./X_train_norm.csv')
    
    # Models training
    print('training in progress...')
    # Linear Discrimant Analysis (Default parameters)
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_train, y_train)
    
    # Serialize model
    from joblib import dump
    dump(clf_lda, MODEL_PATH_LDA)
        
    # Neural Networks multi-layer perceptron (MLP) algorithm
    clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=1000)
    clf_NN.fit(X_train, y_train)
       
    # Serialize model
    from joblib import dump, load
    dump(clf_NN, MODEL_PATH_NN)

# Return classification score for both Neural Network and LDA inference model from the all dataset provided
def test():

    # Load, read and normalize testing data
    testing = "./test_ASC.csv"
    data_test = pd.read_csv(testing)
    y_test = data_test['label'].values
    X_test = data_test.drop('label', axis = 1)

    print("Shape of the testing data")
    print(X_test.shape)
    print(y_test.shape)
    
    # Data normalization -> (0,1)
    X_test = min_max_normalize(X_test)

    clf_lda = load(MODEL_PATH_LDA)
    score_lda = clf_lda.score(X_test, y_test)
    
    clf_nn = load(MODEL_PATH_NN)
    score_nn = clf_nn.score(X_test, y_test)
    
    print('LDA score: {}, Neural Network score: {}'.format(score_lda, score_nn))
        
if __name__ == '__main__':
    train()
    test()