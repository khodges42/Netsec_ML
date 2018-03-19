from __future__ import division
from pyfiglet import Figlet
prog = "Lambda"
descr = "Regressive Covariance Neural Network"
print(Figlet(font='katakana').renderText(prog)+"\n"+descr+"\n"+"-"*100)
import ijson
import pandas as pd
import argparse
import json
import os
import random
import numpy

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.regularizers import L1L2

#stfu
import logging
logging.getLogger('tensorflow').disabled = True
#quickmaths
pd.options.display.float_format = '{:.4f}'.format


def frange(start, stop, step):
    if start > stop:
        i = start
        while i > stop:
            yield i
            i -= step
    else:
        i = start
        while i < stop:
            yield i
            i += step

def calculate_cm_stats(binary_y, df, x):
    cm = confusion_matrix(binary_y, (df > x)).ravel()
    _tn, _fp, _fn, _tp = cm
    _totals = _tn + _fp + _fn + _tp
    _accuracy = float(_tp+_tn)/_totals
    _fn_inverse = _fn/_totals
    _percent_fn = (100-(100*_fn_inverse))/100
    _fn_to_accuracy = (_accuracy + _percent_fn)/2
    _weighted_fn_to_accuracy = float(((_percent_fn * 2) + _accuracy)/3)
    _percent_detected = (_tp)/(_tp+_fn)
    _weighted_detection_to_accuracy = ((_percent_detected + _accuracy) / 2)
    _lambda = (_weighted_detection_to_accuracy + _weighted_fn_to_accuracy) / 2
    return {
                "tn":_tn,"fp":_fp,"fn":_fn,"tp":_tp,
                "total":_totals,
                "accuracy":_accuracy,
                "fn_inverse":_fn_inverse,
                "percent_fn":_percent_fn,
                "fn_to_accuracy": _fn_to_accuracy,
                "weighted_fn_to_accuracy":_weighted_fn_to_accuracy,
                "percent_detected":_percent_detected,
                "weighted_detection_to_accuracy":_weighted_detection_to_accuracy,
                "lambda":_lambda,
            }



def gd_threshold(df, binary_y, best = 0, variable = "accuracy", verbose = False):
    best_x = 0
    for x in frange(1.0, 0, 0.01):
        stats = calculate_cm_stats(binary_y, df, x)
        if stats["weighted_fn_to_accuracy"] > best and variable == "fn_weighted":
            best = stats["weighted_fn_to_accuracy"]
            best_x = x
        if stats["fn_to_accuracy"] > best and variable == "fn_to_accuracy":
            best = stats["fn_to_accuracy"]
            best_x = x
        if stats["accuracy"] > best and variable == "accuracy":
            best = stats["accuracy"]
            best_x = x
        if  stats["weighted_detection_to_accuracy"] > best and variable == "detection":
            best = stats["weighted_detection_to_accuracy"]
            best_x = x
        if  stats["lambda"] > best and variable == "lambda":
            best = stats["lambda"]
            best_x = x

    if verbose:
        stats = calculate_cm_stats(binary_y, df, best_x)
        print("Best Threshold for {}: {}".format(variable, best_x))
        print confusion_matrix(binary_y, (df > best_x)).ravel()
        print(stats)
    return best_x, best


def test(args, df, config):
    print(list(df))
    df = df[df[' Label'] != '0']
    df = df.reset_index(drop=True)
    df = df.fillna(0)
    df_y = df[[' Label']]
    df = df[['Mirage_Anomaly', 'Deus_Malicious']]

    df_preds = predict(args, df, config)
    binary_y = numpy.where(df_y[" Label"] == "BENIGN", 0, 1)

    threshold, acc = gd_threshold(df_preds["Lambda_Correlation"], binary_y, variable = 'fn_to_accuracy')
    tn, fp, fn, tp = confusion_matrix(binary_y, (df_preds["Lambda_Correlation"] > threshold)).ravel()

    print("==Calculated Threshold for Minimized False Negatives/Accuracy")
    print(("FP: {}, FN: {},".format(fp,fn))+'\n'+("TP: {}. TN: {},".format(tp,tn)))
    print("Total Accuracy:{}%".format(float(tp+tn)/(tn+fp+fn+tp)))
    print("Weighted FN Accuracy:{}%".format(acc))

    print("==Describe")
    print(df_preds.describe())

    print("==Covariance")
    print(df_preds.cov())

    return pd.concat([df_preds, df_y[" Label"]], axis=1)


def predict(args, df, config):

    df = df[['Mirage_Anomaly', 'Deus_Malicious']]
    df_height, df_width = df.shape # I Changed this and hopefully it wont break anything
    df = df.reset_index(drop=True)
    df = df.fillna(0)
    df_np_d = df.as_matrix()
    scaler = MinMaxScaler()
    df_np_d = scaler.fit_transform(df_np_d)
    X = df_np_d

    json_file = open(args.model_store+"model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(args.model_store+"model.h5")
    print("Loaded model from disk")

    prediction = model.predict(X)
    df_preds = pd.DataFrame(columns=["Lambda_Correlation"])
    df_preds["Lambda_Correlation"] = [x[0] for x in prediction]

    # Normalize
    x = df_preds[["Lambda_Correlation"]].values.astype(float) #returns a numpy array
    normalize_scaler = MinMaxScaler()
    x_scaled = normalize_scaler.fit_transform(x)
    df_preds["Lambda_Correlation"] = pd.DataFrame(x_scaled)

    return pd.concat([df, df_preds["Lambda_Correlation"]], axis=1)


def LR_model():
    model = Sequential()
    reg = L1L2(l1=0.01, l2=0.01)
    model.add(Dense(1, activation='sigmoid', W_regularizer=reg, input_dim=2))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def train(args, df, config):
    df = df[df[' Label'] != '0']
    df = df.reset_index(drop=True)
    df = df.fillna(0)
    df_y = df[[' Label']]
    df = df[['Mirage_Anomaly', 'Deus_Malicious']]
    df_height, df_width = df.shape # I Changed this and hopefully it wont break anything
    df = df.reset_index(drop=True)
    df = df.fillna(0)
    df_np_d = df.as_matrix()
    scaler = MinMaxScaler()
    df_np_d = scaler.fit_transform(df_np_d)
    X = df_np_d
    Y = numpy.where(df_y[" Label"]=="BENIGN", 0, 1)
    # Evaluate model using standardized dataset.
    model = LR_model()
    model.fit(X,Y, epochs=100, batch_size=32, verbose=1)
    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # serialize model to JSON
    model_json = model.to_json()
    with open(args.model_store+"model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(args.model_store+"model.h5")
    print("Saved model to disk")

def test_model(args, df, config):
    df_y = df[[' Label']]
    df = df[[ ' Protocol', ' Destination Port', ' Source Port', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward']]
    df_height, df_width = df.shape # I Changed this and hopefully it wont break anything
    df = df.reset_index(drop=True)
    df = df.fillna(0)
    df_np_d = df.as_matrix()
    scaler = MinMaxScaler()
    df_np_d = scaler.fit_transform(df_np_d)
    X = df_np_d
    Y = numpy.where(df_y[" Label"]=="BENIGN", 0, 1)
    # Evaluate model using standardized dataset.
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=base_model, epochs=100, batch_size=5, verbose=1)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def load_data(args):
    print("Loading input data...")
    input_name,input_extension = os.path.splitext(args.input)
    with open(args.input, 'rb') as f:
        if input_extension.upper() == ".JSON": df = pd.read_json(args.input)
        if input_extension.upper() == ".CSV": df = pd.read_csv(args.input, engine='python', sep='\t', nrows=args.rows, index_col=0)
        print("Data Loaded.")
    if args.random_sample:
        sample_size = int(round(df.shape[0]*args.random_sample))
        print("Returning Random Sample: {}".format(sample_size))
        rows = numpy.random.choice(df.index.values, sample_size, replace=False)
        df = df.ix[rows].reset_index(drop=True)
        return df.fillna(0)
    else:
        return df

def run(config):
    df = load_data(args)

    if args.train: train(args, df, config)
    if args.test: df = test(args, df, config)
    if args.predict: df = predict(args, df, config)

    if args.output:
        output_name,output_extension = os.path.splitext(args.output)
        if output_extension.upper() == ".CSV":
            df.to_csv(args.output, sep='\t')
        elif output_extension.upper() == ".JSON":
            with open(args.output, 'w') as f:
                f.write(df.to_json())
    elif not args.train:
        print(df.head())
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("--input", type=str, default="../data/categorical2.csv", help="Input Dataset")
    parser.add_argument("--output", type=str, default=False, help="Data output directory")
    parser.add_argument("--model_store", type=str, default="../../../models/GAN/lambda/", help="Model output directory")
    parser.add_argument("--config", type=str, default="config.json", help="Config file")
    parser.add_argument("--test", action='store_true', help="Test Labeled Data")
    parser.add_argument("--predict", action='store_true', help="Generate Predictions on Input")
    parser.add_argument("--train", action='store_true', help="Train new Model")
    parser.add_argument("--rows", type=int, default=None, help="Max Rows, default is all")
    parser.add_argument("--random_sample", type=float, default=None, help="Use random sample of data, format is 0.5")

    args = parser.parse_args()
    config = json.load(open(args.config)) if args.config else {"columns": ""}


    run(config)
