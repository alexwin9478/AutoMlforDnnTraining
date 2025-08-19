import os
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
import random
import time
import datetime
import json
import psutil
import gc
import ast
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from optuna.pruners import HyperbandPruner
from optuna.storages import RetryFailedTrialCallback
from optuna.exceptions import StorageInternalError
from optuna.integration import KerasPruningCallback
from optuna.trial import TrialState
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from keras import backend as be    

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#  Level | Level for Humans | Level Description                  
#  -------|------------------|------------------------------------ 
#   0     | DEBUG            | [Default] Print all messages       
#   1     | INFO             | Filter out INFO messages           
#   2     | WARNING          | Filter out INFO & WARNING messages 
#   3     | ERROR            | Filter out all messages 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

start_time = time.time()

print("Tensorflow version: ", tf.__version__)
tf.config.run_functions_eagerly(False)

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# Einstellungen
journal_storage = 1
url = 'sqlite:///BO_Search_GE_1.sqlite'
url_journal = "./journal_GE_1.log"
studyname = "BO_Search_GE"
timeout =float(2 * 60 *60)
epochs = 3500
epochs_start = 500
trials = 20
jobs = 1 
folds = 5
batch_sizes = ['512']
l2 = 0.01
learningrate_drop = [500, 1000, 1500, 2000, 2500, 3000]
opti_alg = 'Adam'
filename = 'Test_480_to_507_conc_norm.csv'

x_train, y_train, x_val, y_val =[],[],[],[]
# Load the CSV file
cwd = os.getcwd()
filename = os.path.join(cwd, 'data', filename)
df = pd.read_csv(filename)

# Convert the entire DataFrame to a string
data_str = df.to_string()

# Find the relevant portion of the string that represents the dictionary
length_ = len(data_str)
data_str_ = 'dic_data=' + data_str[26:length_-11]

# Extract the dictionary string
start_index = data_str_.find('=') + 1
dict_str = data_str_[start_index:].strip()

# Convert the string to a dictionary
dic_data = ast.literal_eval(dict_str)

print(dic_data.keys())
# Convert the relevant dictionary entries to NumPy arrays
utrain_array = np.array(dic_data['utrain_load']).astype(np.float32)
ytrain_array = np.array(dic_data['ytrain_load']).astype(np.float32)
uval_array = np.array(dic_data['uval_load']).astype(np.float32)
yval_array = np.array(dic_data['yval_load']).astype(np.float32)
utest_array = np.array(dic_data['utest_load']).astype(np.float32)
ytest_array = np.array(dic_data['ytest_load']).astype(np.float32)


# Ensure all arrays have compatible shapes for concatenation
if utrain_array.shape[0] == uval_array.shape[0] == utest_array.shape[0]:
    X_normalized = np.concatenate((utrain_array, uval_array, utest_array), axis=1).T
else:
    raise ValueError("Shape mismatch: utrain, uval, and utest arrays have different number of rows")

if ytrain_array.shape[0] == yval_array.shape[0] == ytest_array.shape[0]:
    y_normalized = np.concatenate((ytrain_array, yval_array, ytest_array), axis=1).T
else:
    raise ValueError("Shape mismatch: ytrain, yval, and ytest arrays have different number of rows")

def data_splitter(x_tensor, y_tensor, split=0.85):
    total_samples = len(x_tensor)
    no_train = round(total_samples * split)

    x_train = tf.convert_to_tensor(x_tensor[:no_train, :], dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_tensor[:no_train, :], dtype=tf.float32)
    x_val = tf.convert_to_tensor(x_tensor[no_train:, :], dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_tensor[no_train:, :], dtype=tf.float32)
    
    return x_train, y_train, x_val, y_val

def dataset_creator(x_tensor, y_tensor, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor)).batch(batch_size)
    return dataset

# Function to calculate value counts in ranges
def calculate_value_counts(data, ranges):
    # Ensure data is a numpy array
    if isinstance(data, tf.Tensor):
        data = data.numpy()
        
    counts = {f"{r[0]}-{r[1]}": 0 for r in ranges}
    data_flat = data.flatten()
    
    for value in data_flat:
        for r in ranges:
            if r[0] <= value < r[1]:
                counts[f"{r[0]}-{r[1]}"] += 1
                break
                
    total = len(data_flat)
    for key in counts:
        counts[key] = counts[key] / total
    
    return counts

# Define the ranges
ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

index_ = X_normalized.shape
val_sec_= int(np.floor(0.75*index_[0]))

# Convert normalized data to tensors once
X_normalized = tf.convert_to_tensor(X_normalized[val_sec_:,:], dtype=tf.float32)
y_normalized = tf.convert_to_tensor(y_normalized[val_sec_:,:], dtype=tf.float32)


def hyperparameter_space(trial, sampler_choice):
    batch_size = trial.suggest_categorical('batch_size', batch_sizes)
    network_param_dict = {}
    n_layers_before_lstm = trial.suggest_int("n_layers_before_lstm", 1, 4)
    hidden_units_before_lstm = []
    
    for i in range(n_layers_before_lstm):
        units_key = f"n_units_before_lstm_{i}"
        if units_key in network_param_dict:
            n_units = network_param_dict[units_key]
        else:
            n_units = trial.suggest_categorical(f"n_units_before_lstm_{i}", [8, 16, 32, 64])
            network_param_dict[units_key] = n_units
        hidden_units_before_lstm.append(n_units)

    n_layers_after_lstm = trial.suggest_int("n_layers_after_lstm", 1, 4)  # Fixing the typo here
    hidden_units_after_lstm = []
    
    for i in range(n_layers_after_lstm):
        units_key = f"n_units_after_lstm_{i}"
        if units_key in network_param_dict:
            n_units = network_param_dict[units_key]
        else:
            n_units = trial.suggest_categorical(f"n_units_after_lstm_{i}", [4, 8, 16, 32, 64])
            network_param_dict[units_key] = n_units
        hidden_units_after_lstm.append(n_units)

    network_param_dict.update({
        "n_layers_before_lstm": n_layers_before_lstm,
        "hidden_units_before_lstm": hidden_units_before_lstm,
        "n_layers_after_lstm": n_layers_after_lstm,
        "hidden_units_after_lstm": hidden_units_after_lstm,
        "l2": l2,
        "lstm_hs": trial.suggest_categorical('lstm_hs', [2, 4, 6, 8])
    })
    
    optimizer_param_dict = {}
    optimizer_param_dict['kwargs'] = {}
    optimizer_param_dict['name'] = opti_alg
    
    # Optimizer parameters with learning rate scheduling
    initial_lr = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    boundaries = learningrate_drop  # Define your boundaries based on the number of epochs
    values = [initial_lr, initial_lr * 0.5, initial_lr * 0.5**2, initial_lr * 0.5**3, initial_lr * 0.5**4, initial_lr * 0.5**5, initial_lr * 0.5**6]
    lr_schedule_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer_param_dict['kwargs']['learning_rate'] = lr_schedule_fn
    optimizer_param_dict['kwargs']['epsilon'] = 1e-8
    optimizer_param_dict['kwargs']['amsgrad'] = True
    optimizer_param_dict['kwargs']['weight_decay'] = 0.1
    optimizer_param_dict['kwargs']['clipnorm'] = 1
    optimizer_param_dict['kwargs']['use_ema'] = True
    
    return network_param_dict, optimizer_param_dict, batch_size

def generate_model(network_param_dict, verbosity=1):
    # tf.keras.backend.clear_session()

    model = Sequential()
    model.add(tf.keras.Input(shape=(5,), name="input_layer"))
    
    for i in range(network_param_dict['n_layers_before_lstm']):
        units_key = f"n_units_before_lstm_{i}"
        units = network_param_dict[units_key]
        model.add(Dense(1 * units, activation="relu", bias_regularizer=tf.keras.regularizers.L2(network_param_dict["l2"]), name=f"dense_before_lstm_{i+1}"))
        #model.add(tf.keras.layers.Dropout(0.2))  # Adding dropout for regularization
    
    model.add(Reshape((1, 1 * network_param_dict['hidden_units_before_lstm'][-1]), name="reshape_layer"))
    model.add(LSTM(network_param_dict["lstm_hs"], return_sequences=True, name="lstm_layer"))
    #model.add(tf.keras.layers.Dropout(0.2))  # Adding dropout for regularization
    
    for i in range(network_param_dict['n_layers_after_lstm']):
        units_key = f"n_units_after_lstm_{i}"
        units = network_param_dict[units_key]
        model.add(Dense(1 * units, activation="relu", bias_regularizer=tf.keras.regularizers.L2(network_param_dict["l2"]), name=f"dense_after_lstm_{i+1}"))
        #model.add(tf.keras.layers.Dropout(0.2))  # Adding dropout for regularization

    model.add(Dense(4, name="dense_output"))
    model.add(Reshape((4,), name="reshape_output_layer"))
    
    if verbosity == 1:
        print(model.summary())

    return model

def generate_optimizer(optimizer_param_dict, verbosity=1):
    kwargs = optimizer_param_dict['kwargs']
    optimizer = getattr(tf.keras.optimizers, optimizer_param_dict['name'])(**kwargs)
    if verbosity == 1:
        print(f"Name of optimizer:  {optimizer_param_dict['kwargs']['learning_rate']}")
    return optimizer

def objective_fn(trial, X_normalized, y_normalized, sampler_choice):
    process = psutil.Process(os.getpid())  # Prozess definieren
    network_params, optimizer_params, batch_size = hyperparameter_space(trial, sampler_choice)
    print("Starting with new hyperparameter combination")
    delay_seconds = random.uniform(0, 20)
    time.sleep(delay_seconds)

    # Generiere das Modell und den Optimizer
    model = generate_model(network_params)
    optimizer = generate_optimizer(optimizer_params)
    
    # Zähle die lernbaren Parameter
    learnable_params = sum(np.prod(p.shape) for p in model.trainable_weights)

    # Speichere die Anzahl der lernbaren Parameter im Trial
    trial.set_user_attr("learnable_params", int(learnable_params))
    print(f"Learnable Params for trial {trial.number}: {learnable_params}")  # Debug-Ausgabe

    GE = learn(network_params, optimizer_params, X_normalized, y_normalized, batch_size, trial)
    
    trial.set_user_attr("GE", GE)

    # Kombiniere GE und die Anzahl der lernbaren Parameter in eine Gesamtmetrik
    combined_metric = GE 

    # clean up
    #tf.keras.backend.clear_session()
    #be.clear_session()
    #gc.collect()
    del model, optimizer, network_params, optimizer_params, batch_size, process, trial, GE, learnable_params, delay_seconds
 
    return combined_metric

def learn(network_params, optimizer_params, X_normalized, y_normalized, batch_size, trial):

    bs = int(batch_size)
    cv_outer = TimeSeriesSplit(n_splits = folds)
    fold_out = 1

    for train_out_ix, test_ix in cv_outer.split(X_normalized):
            
        tensorboard_callback = []
        history = []
        GE_list = []
        Networkmodel_hcci = []
        optimizer = []
        callbacks_inner = []
        early_stop = []
        checkpoint = []
        history = []
        test_loss = []
        min_test = []
        
        time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Convert indices to TensorFlow tensors
        train_out_ix = tf.convert_to_tensor(train_out_ix, dtype=tf.int32)
        test_ix = tf.convert_to_tensor(test_ix, dtype=tf.int32)
        
        X_train_out, X_test = tf.gather(X_normalized, train_out_ix), tf.gather(X_normalized, test_ix)
        y_train_out, y_test = tf.gather(y_normalized, train_out_ix), tf.gather(y_normalized, test_ix)
        
        verbosity = 1 if fold_out == 1 else 0
        Networkmodel_hcci = generate_model(network_params, verbosity)
        optimizer = generate_optimizer(optimizer_params, verbosity)
        
        if verbosity == 1:
            print(f'Batch Size: {batch_size} and trial {trial.number}')
            print(f'Regularization factor: {network_params["l2"]} and trial {trial.number}')
        
        # Experiment with different loss functions
        loss_function = tf.keras.losses.MeanSquaredError()

        delay_seconds = random.uniform(0, 2)
        time.sleep(delay_seconds)
        Networkmodel_hcci.compile(optimizer=optimizer, loss=loss_function, metrics=[tf.keras.metrics.RootMeanSquaredError()])

        # Split data into training and validation sets
        x_train, y_train, x_val, y_val = data_splitter(X_train_out, y_train_out, split=0.85)
        
        # Create datasets
        train_dataset_in = dataset_creator(x_train, y_train, batch_size=bs)
        val_dataset_in = dataset_creator(x_val, y_val, batch_size=bs)
    
        log_dir_in = f'./experiment_logs/model_dir_BO_GE/val_{fold_out}_{time_stamp}_trial{trial.number}'
        model_dir_in = f"./experiment_logs/log_dir_BO_GE/val_{fold_out}_{time_stamp}_trial{trial.number}.weights.h5"
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_in, histogram_freq=0, update_freq='epoch')
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, start_from_epoch=epochs_start, verbose=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_dir_in, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=0)
        callbacks_inner = [checkpoint, early_stop, tensorboard_callback]        

        delay_seconds = random.uniform(0, 2)
        time.sleep(delay_seconds)    
        history = Networkmodel_hcci.fit(train_dataset_in, verbose=0, callbacks=callbacks_inner, epochs=epochs, validation_data=val_dataset_in, validation_freq=1)
        
        val_loss = history.history['val_loss']
        min_val = min(val_loss)
        
        print(f"Fold {fold_out} and trial {trial.number} gives best validation loss of {min_val} in epoch {val_loss.index(min_val)}")
        
        print(f"Starting with GE training and evaluation on the test set")
        
        Networkmodel_hcci.load_weights(model_dir_in)
        
        model_dir_out = f'./experiment_logs/model_dir_BO_GE/test_{fold_out}_{time_stamp}_trial{trial.number}.weights.h5'
        log_dir_out = f"./experiment_logs/log_dir_BO_GE/test_{fold_out}_{time_stamp}_trial{trial.number}"
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, start_from_epoch=epochs_start, verbose=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_dir_out, monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True, verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_out, histogram_freq=0, update_freq='epoch')
        callbacks_outer = [tensorboard_callback, early_stop, checkpoint]
                
        train_dataset_out = dataset_creator(X_train_out, y_train_out, batch_size=bs)
        test_dataset_out = dataset_creator(X_test, y_test, batch_size=bs)
        
        delay_seconds = random.uniform(0, 2)
        time.sleep(delay_seconds)
        history = Networkmodel_hcci.fit(train_dataset_out, validation_data=test_dataset_out, callbacks=callbacks_outer, epochs=epochs, verbose=0, validation_freq=1)
        
        test_loss = history.history['val_loss']
        min_test = min(test_loss)
        
        GE_list.append(min_test)
        print(f"Fold {fold_out} and trial {trial.number} gives the best test loss of {min_test} in epoch {test_loss.index(min_test)}")        

        fold_out += 1   

        # garbage collection
         # Split data into training and validation sets
        del x_train, y_train, x_val, y_val, train_dataset_in, val_dataset_in, history, train_dataset_out, test_dataset_out, Networkmodel_hcci, callbacks_outer, tensorboard_callback, early_stop, checkpoint

        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")
        
    average_GE = np.average(GE_list)
    print(f"Average generalization loss: {average_GE}\n")
   
    print("Done with the hyperparameter combination")
    print("---------------------------------------------------------------------------------------------------------------------------------------\n")
    print("---------------------------------------------------------------------------------------------------------------------------------------\n")
    
    return average_GE

def save_optuna_results(study, filename):
    """
    Saves Optuna-Study in CSV-File.
    """
    trials = study.trials_dataframe()
    # Füge eine Spalte für die lernbaren Parameter hinzu
    learnable_params_list = [t.user_attrs.get("learnable_params", None) for t in study.trials]
    print(f"Learnable Params List: {learnable_params_list}")  # Debug-Ausgabe
    trials['learnable_params'] = learnable_params_list
    trials.to_csv(filename, index=False)
    print(f"Ergebnisse in {filename} gespeichert.")

def save_trials_json(study, filename):
    trials_data = [{"number": trial.number, "value": trial.value, "params": trial.params} for trial in study.trials]
    with open(filename, 'w') as f:
        json.dump(trials_data, f)
    print(f"Trials in {filename} gespeichert.")

def save_best_trial_json(study, filename):
    best_trial = study.best_trial
    best_trial_data = {"number": best_trial.number, "value": best_trial.value, "params": best_trial.params}
    with open(filename, 'w') as f:
        json.dump(best_trial_data, f)
    print(f"Best trial in {filename} gespeichert.")


def search_BO(n_trials=trials, filename=filename, url=url, studyname=studyname):
    best_dict = {}
    best_dict['hyperparams'] = {}

    delay_seconds = random.uniform(0, 30)
    time.sleep(delay_seconds)
    
    sampler_choice = 'BO'   
          
    if journal_storage == 1:
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(url_journal),
        )
    else:
        storage = optuna.storages.RDBStorage(
            url=url,
            heartbeat_interval=30,
            grace_period=60,
            #failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
            engine_kwargs={"pool_size": 96*4,"max_overflow" : 0, "connect_args": {"timeout": 360}},
            #skip_table_creation={True}        
        )

    
    print("----Starting sampler: BO-TPE----")
    
    # Hinzufügen des Hyperband Pruners
    pruner = HyperbandPruner()
    
    delay_seconds = random.uniform(0, 10)
    time.sleep(delay_seconds)

    objective = lambda trial: objective_fn(trial, X_normalized, y_normalized, sampler_choice)
    
    study = optuna.load_study(storage=storage, study_name=studyname, sampler=optuna.samplers.TPESampler(constant_liar=True), pruner=pruner)
    
    study.optimize(objective,  timeout=timeout, n_trials=trials, n_jobs=jobs, gc_after_trial=True, catch=[ValueError,RuntimeError,StorageInternalError,BlockingIOError], callbacks=[lambda study, trial: gc.collect()])
    
    #from optuna.study import MaxTrialsCallback
    #from optuna.trial import TrialState
    # study.optimize(
    #     objective, n_trials=50, callbacks=[MaxTrialsCallback(10, states=(TrialState.COMPLETE,))]
    # )
    # trials = study.trials_dataframe()
    # print("Number of completed trials: {}".format(len(trials[trials.state == "COMPLETE"])))

    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    print("Number of finished trials: ", len(study.trials))
    print("Best trial: ")
    best_trial = study.best_trial
    best_dict['best_trial'] = best_trial
    print("Value: ", best_trial.value)
    best_dict['best_value'] = best_trial.value
    print("Params: ")
    for key, value in best_trial.params.items():
        best_dict['hyperparams'][key] = value
        print(" {}: {}".format(key, value))

    save_optuna_results(study, "optuna_results_nop"+studyname +".csv")
    save_trials_json(study, "all_trials_nop"+studyname +".json")
    save_best_trial_json(study, "best_trial_nop"+studyname +".json")

    return best_dict


best_dict = search_BO(n_trials=trials, filename=filename, url=url, studyname=studyname)

end_time = time.time()
process = psutil.Process(os.getpid())
print(f"Runtime: {end_time - start_time} seconds")
print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")