from pyfiglet import Figlet
prog = "Mirage"
descr = "Anomaly Detection Neural Network"
print(Figlet(font='ticksslant').renderText(prog)+"\n"+descr+"\n"+"-"*100)
import ijson
import pandas as pd
import argparse
import json
import os
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from anomalous_lib.library.convolutional import Conv1DAutoEncoder
from sklearn.metrics import confusion_matrix


def test(args, df, config):
    df_y = df[[' Label']]
    df = df[[ ' Protocol', ' Destination Port', ' Source Port', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward']]


    df_preds = predict(args, df, config)



    threshold = df_preds["Mirage_Anomaly"].mean()
    tn, fp, fn, tp = confusion_matrix(np.where(df_y[" Label"]=="BENIGN", 0, 1), (df_preds["Mirage_Anomaly"] > threshold)).ravel()
    print(("FP: {}, FN: {},".format(fp,fn))+'\n'+("TP: {}. TN: {},".format(tp,tn)))
    print("Accuracy:{}%".format(float(tp+tn)/float(fp+tp+tn+fn)))
    return pd.concat([df_preds, df_y[" Label"]], axis=1)

def predict(args, df, config):
    ae = Conv1DAutoEncoder()
    df = df[[ ' Protocol', ' Destination Port', ' Source Port', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward']]
    df_height, df_width = df.shape
    print("Running on {} Rows".format(df_height))
    df = df.reset_index(drop=True)
    df = df.fillna(0)
    df_np_d = df.as_matrix()
    scaler = MinMaxScaler()
    df_np_d = scaler.fit_transform(df_np_d)
    ae.load_model(args.model_store)
    anomaly_information = ae.anomaly(df_np_d[:df_height, :]) #Should this be height?

    df_preds = pd.DataFrame(columns=["Mirage_Anomaly"])
    df_preds["Mirage_Anomaly"] = [x[1] for x in anomaly_information]
    # Normalize
    x = df_preds[["Mirage_Anomaly"]].values.astype(float) #returns a numpy array
    normalize_scaler = MinMaxScaler()
    x_scaled = normalize_scaler.fit_transform(x)
    df_preds["Mirage_Anomaly"] = pd.DataFrame(x_scaled)

    return pd.concat([df, df_preds["Mirage_Anomaly"]], axis=1)


def train(args, df, config):
    df = df[[ ' Protocol', ' Destination Port', ' Source Port', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward']]
    df_height, df_width = df.shape # I Changed this and hopefully it wont break anything
    df_np_d = df.as_matrix()
    scaler = MinMaxScaler()
    df_np_d = scaler.fit_transform(df_np_d)

    ae = Conv1DAutoEncoder()

    ae.fit(df_np_d, model_dir_path=args.model_store, estimated_negative_sample_ratio=0.95, batch_size=10000, epochs=100, validation_split=0.1)

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
        rows = np.random.choice(df.index.values, sample_size, replace=False)
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
        print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("--input", type=str, default="../data/testout.csv", help="Input Dataset")
    parser.add_argument("--output", type=str, default=False, help="Data output directory")
    parser.add_argument("--model_store", type=str, default="../../../models/GAN/anomalous/", help="Model output directory")
    parser.add_argument("--config", type=str, default="config.json", help="Config file")
    parser.add_argument("--test", action='store_true', help="Test Labeled Data")
    parser.add_argument("--predict", action='store_true', help="Generate Predictions on Input")
    parser.add_argument("--train", action='store_true', help="Train new Model")
    parser.add_argument("--rows", type=int, default=None, help="Max Rows, default is all")
    parser.add_argument("--random_sample", type=float, default=None, help="Use random sample of data, format is 0.5")

    args = parser.parse_args()
    config = json.load(open(args.config)) if args.config else {"columns": ""}


    run(config)
