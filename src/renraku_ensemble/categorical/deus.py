from pyfiglet import Figlet
prog = "Deus"
descr = "Intrusion Detection Neural Network"
print(Figlet(font='block').renderText(prog)+"\n"+descr+"\n"+"-"*100)
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
pd.options.display.float_format = '{:.4f}'.format



def test(args, df, config):
    df = df[df[' Label'] != '0']
    df = df.reset_index(drop=True)
    df = df.fillna(0)
    df_y = df[[' Label']]
    df = df[[ ' Protocol', ' Destination Port', ' Source Port', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', 'Mirage_Anomaly']]

    df_preds = predict(args, df, config)

    threshold = df_preds["Deus_Malicious"].mean()
    tn, fp, fn, tp = confusion_matrix(numpy.where(df_y[" Label"] == "BENIGN", 0, 1), (df_preds["Deus_Malicious"] > threshold)).ravel()
    print(("FP: {}, FN: {},".format(fp,fn))+'\n'+("TP: {}. TN: {},".format(tp,tn)))
    totals = float(fp+tp+tn+fn)
    print("Total Accuracy:{}%".format(float(tp+tn)/totals))
    print("Missed Positives:{}%".format(float(fn)/totals))


    return pd.concat([df_preds, df_y[" Label"]], axis=1)


def predict(args, df, config):
    json_file = open(args.model_store+"model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(args.model_store+"model.h5")
    print("Loaded model from disk")

    df = df[[ ' Protocol', ' Destination Port', ' Source Port', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', 'Mirage_Anomaly']]
    df_height, df_width = df.shape # I Changed this and hopefully it wont break anything

    df_np_d = df.as_matrix()
    scaler = MinMaxScaler()
    df_np_d = scaler.fit_transform(df_np_d)
    X = df_np_d

    prediction = model.predict(X)
    df_preds = pd.DataFrame(columns=["Deus_Malicious"])
    df_preds["Deus_Malicious"] = [x[0] for x in prediction]


        # Normalize
    x = df_preds[["Deus_Malicious"]].values.astype(float) #returns a numpy array
    normalize_scaler = MinMaxScaler()
    x_scaled = normalize_scaler.fit_transform(x)
    df_preds["Deus_Malicious"] = pd.DataFrame(x_scaled)
    print(df_preds.describe())

    #df_preds["Deus_Malicious"] = numpy.array(prediction)
    return pd.concat([df, df_preds["Deus_Malicious"]], axis=1)


def base_model():
    model = Sequential()
    model.add(Dense(10, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train(args, df, config):
    df = df[df[' Label'] != '0']
    df = df.reset_index(drop=True)
    df = df.fillna(0)
    df_y = df[[' Label']]
    df = df[[ ' Protocol', ' Destination Port', ' Source Port', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', 'Mirage_Anomaly']]
    df_height, df_width = df.shape # I Changed this and hopefully it wont break anything
    df = df.reset_index(drop=True)
    df = df.fillna(0)
    df_np_d = df.as_matrix()
    scaler = MinMaxScaler()
    df_np_d = scaler.fit_transform(df_np_d)
    X = df_np_d
    Y = numpy.where(df_y[" Label"]=="BENIGN", 0, 1)
    # Evaluate model using standardized dataset.
    model = base_model()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("--input", type=str, default="../data/anomalous.csv", help="Input Dataset")
    parser.add_argument("--output", type=str, default=False, help="Data output directory")
    parser.add_argument("--model_store", type=str, default="../../../models/GAN/categorical/", help="Model output directory")
    parser.add_argument("--config", type=str, default="config.json", help="Config file")
    parser.add_argument("--test", action='store_true', help="Test Labeled Data")
    parser.add_argument("--predict", action='store_true', help="Generate Predictions on Input")
    parser.add_argument("--train", action='store_true', help="Train new Model")
    parser.add_argument("--rows", type=int, default=None, help="Max Rows, default is all")
    parser.add_argument("--random_sample", type=float, default=None, help="Use random sample of data, format is 0.5")

    args = parser.parse_args()
    config = json.load(open(args.config)) if args.config else {"columns": ""}


    run(config)
