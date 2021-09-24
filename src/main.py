import numpy as np
import pandas as pd
import argparse
import sys
sys.path.insert(0, '../models/')
import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import utils
from ED import ED_Model
from DTW import DTW_Model
from FLSTM import FLSTM_Model
from SCNN import SCNN_Model
from TL import Transfer_Learning_Model
from TL2 import Transfer_Learning_Model2
import config
import warnings
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6  
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='Few Shot Learning Task Sampling and Models')

    parser.add_argument('-m', '--Model', type=str, required = True, help= 'Which model to use? TL1 uses ECG5000, TL2 uses ECG200', 
                                                                    choices= ['SCNN', 'ED', 'DTW', 'FLSTM', 'TL1', 'TL2'])
    parser.add_argument('-kl', '--Kmin', type=int, default= 1, required = False, help= 'Minimum value of k (default: 1)')
    parser.add_argument('-ku', '--Kmax', type=int, default= 50, required = False, help=' Maximum value of k (default: 50)')
    parser.add_argument('-q', '--Queries', type = int, default = 20, required= False, help = 'Number of queries per class (default: 20)')
    parser.add_argument('-t', '--Tasks', type = int, default = 20, required=False, help = 'Number of tasks (default: 20)')
    parser.add_argument('-rt', '--Retrain', type = int, default= 0, required= False, choices=[0,1], \
                                            help = 'Retrain SCNN or Transfer Learning Model? If 0, uses existing weights for task adaptation. \
                                                    If 1, trains from scratch, following arguments can be specified for training')
    parser.add_argument('-bs', '--BatchSize', type = int, default = 32, required=False, help = 'Batch size for training SCNN/ LSTM-FCN/ TL, or\
                                                                                                finetuning transfer learning model if not retrained(default: 32)')
    parser.add_argument('-me', '--MaxEpochs', type= int, default = 100, required= False, help = 'Max number of epochs for training SCNN/ LSTM-FCN/ TL, \
                                                                                                or finetuning transfer learning model if not retrained(default: 100)')
    parser.add_argument('-od', '--OutputDir', type = str, default = '../weights/', required=False, help = 'Directory to store model weights of SCNN/ LSTM-FCN/ TL model')
    parser.add_argument('-vb', '--Verbose', type = int, default = 0, required=False, help = 'Verbosity level for training or finetuning', choices= [0, 1, 2])

    args = parser.parse_args()

    if args.Model == 'SCNN':
        SCNN_model = SCNN_Model(retrain = args.Retrain, output_dir = args.OutputDir, batch_size = args.BatchSize, epochs = args.MaxEpochs, verbose = args.Verbose)

        if (args.Retrain):
            print('Creating pairs for SCNN...\n')
            X_train, y_train, X_val, y_val = utils.create_supervised_task(config.ECG_TRAIN_DATASETS, config.ECG_VAL_DATASETS, config.DATASETS_PATH, config.PAD_SIZE)
            print('Done creating training and validation pairs, now training Siamese Network\n')
            SCNN_hist = SCNN_model.fit(X_train, y_train, X_val, y_val)
            print('Done training Siamese Network, now testing...\n')
            del X_train, y_train, X_val, y_val

        X_test, y_test = utils.load_test_data()
        print('\nLoaded test data, running tasks for SCNN...\n')
        SCNN_model.print_metrics_SCNN(X_test, y_test, k_min=args.Kmin, k_max=args.Kmax, num_tasks=args.Tasks, q=args.Queries)
            

    elif args.Model == 'ED':
        X_test, y_test = utils.load_test_data()
        print('Loaded test data, running tasks for ED...\n')
        ED_model = ED_Model()
        ED_model.print_metrics_ED(X_test, y_test, k_min=args.Kmin, k_max=args.Kmax, num_tasks=args.Tasks, q=args.Queries)

    elif args.Model == 'DTW':
        X_test, y_test = utils.load_test_data()
        print('Loaded test data, running tasks for DTW...\n')
        DTW_model = DTW_Model()
        DTW_model.print_metrics_DTW(X_test, y_test, k_min=args.Kmin, k_max=args.Kmax, num_tasks=args.Tasks, q=args.Queries)
    
    elif args.Model == 'FLSTM':
        X_test, y_test = utils.load_test_data()
        print('Loaded test data, running tasks for FLSTM...\n')
        FLSTM_model = FLSTM_Model(batch_size=args.BatchSize, epochs=args.MaxEpochs, verbose=args.Verbose, output_dir = args.OutputDir)
        FLSTM_model.print_metrics_FLSTM(X_test, y_test, k_min=args.Kmin, k_max=args.Kmax, num_tasks=args.Tasks, q=args.Queries)

    elif args.Model == 'TL1':
        TL_model = Transfer_Learning_Model(batch_size=args.BatchSize, epochs=args.MaxEpochs, verbose=args.Verbose, output_dir = args.OutputDir)

        if(args.Retrain):
            X_train, y_train, X_val, y_val = utils.load_TL_data()
            print("Pre training model on source dataset...\n")
            # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
            TL_hist = TL_model.pretrain_model(X_train, y_train, X_val, y_val)
            # print('Loaded test data, running tasks for TL...\n')
            del X_train, y_train, X_val, y_val

            
        X_test, y_test = utils.load_test_data()
        print('\nLoaded test data, finetuning model on target dataset...\n')
        TL_model.print_metrics_TL(X_test, y_test, k_min=args.Kmin, k_max=args.Kmax, num_tasks=args.Tasks, q=args.Queries)

    elif args.Model == 'TL2':
        TL_model = Transfer_Learning_Model2(batch_size=args.BatchSize, epochs=args.MaxEpochs, verbose=args.Verbose, output_dir = args.OutputDir)

        if(args.Retrain):
            X_train, y_train, X_val, y_val = utils.load_TL_data_2()
            print("Pre training model on source dataset...\n")
            TL2_hist = TL_model.pretrain_model(X_train, y_train, X_val, y_val)
            print('Loaded test data, running tasks for TL...\n')
            del X_train, y_train, X_val, y_val

            
        X_test, y_test = utils.load_test_data()
        print('\nLoaded test data, finetuning model on target dataset...\n')
        TL_model.print_metrics_TL(X_test, y_test, k_min=args.Kmin, k_max=args.Kmax, num_tasks=args.Tasks, q=args.Queries)

    else:
        print('Invalid Option!')
    

if __name__ == "__main__": 
    main()

