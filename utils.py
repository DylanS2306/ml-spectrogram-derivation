import os
import numpy as np
import pandas as pd
import librosa
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import integrate
import tensorflow as tf

import cupy as cp



def read_spectrograms(file_name, path=None):
    if path == None:
        path = "./"

    #For some reason mathematica's Export creates 2 sheets, concatenate them both
    spectro1 = pd.read_excel(path + file_name, "Sheet1", header=None, index_col=None)
    spectro2 = pd.read_excel(path + file_name, "Sheet1", header=None, index_col=None)
    
    concat = np.array(sorted(np.concatenate((spectro1, spectro2)), key = lambda x : x[0]))

    return concat[:,0], concat[:,1]


def get_randomised_parameters():
    sgn = [-1,1]

    f = np.random.uniform(low=np.pi/4, high=8*np.pi) #Frequencies
    w_sgn = np.random.uniform(low=-2, high=2)
    w_trig = np.random.randint(low=0, high=3) #Trig/Linear/Constant
    w_period = np.random.uniform(low=0, high=0.1)
    w_ps = np.random.randint(low=-1, high=2)*(np.pi/np.random.randint(low=1,high=10))
    intsgn = sgn[np.random.randint(low=0, high=2)]


    return f, w_period, w_sgn, w_trig, w_ps, intsgn  

def compare_spectrograms(curr_path, new_path):
    print("Comparing Spectrograms...")
    _, target_y = read_spectrograms("mathematica_spectrogram.xlsx")
    _ , curr_y = read_spectrograms(curr_path)
    _ , new_y = read_spectrograms(new_path)
    
    min_length = min(len(curr_y), len(new_y))
    
    return np.linalg.norm(new_y[0:min_length] - target_y[0:min_length]) - np.linalg.norm(curr_y[0:min_length] - target_y[0:min_length]) 

def signal_to_excel(y):

    print("...Generating excel from signal")
    signal = np.array(y)
    frame = pd.DataFrame(signal, columns=['signal_val'])
    frame.to_excel("initial_signal.xlsx", index=False, header=False)
    
    return

def read_intiial_state(path):
    file_name = "initial_guesses.xlsx"
    arr = np.array(pd.read_excel(path + file_name, "Sheet1", index_col=None))
    
    return arr

def get_initial_state(path):

    param = np.array(pd.read_excel(path + "initial_guesses.xlsx", index_col=None))
    spec = read_spectrograms("mathematica_spectrogram.xlsx", path)
    spec_list = []
    point = spec[0]
    state = tuple((param, point))    
    
    return param

def preprocess_signals(sample_direc=None):
    if sample_direc is None:
        sample_direc = "./"
    sample_files = os.listdir(sample_direc)
    for s in sample_files:
        sample = pd.read_excel(sample_direc + s, index=False)
        
    stft = librosa.stft(signal)
    power = librosa.power_to_db(np.abs(stft)**2)
    input_features = power.reshape((power.shape[0], -1)).T
    n_components = 100
    
    pca = PCA(n_components=n_components)
    pca.fit(input_features)

    input_features = pca.transform(input_features)
    
    # return input_features
def construct_labels(n=None, samples=1000):

    if n is None:
        n = 28
    labelled_set = np.empty((n,6,0))
    for i in range(samples):
        sample = np.empty((0,6))
        for _ in range(n):
            f, w_period, w_sgn, w_trig, w_ps, intsgn = get_randomised_parameters()
            sample = np.vstack((sample, [f, w_period, w_sgn, w_trig, w_ps, intsgn]))
        df = pd.DataFrame(sample, columns=["freq", "period", "sgn", "trig", "ps", "intsgn"])
        # df.to_excel(f"./excel_labels/sample_{i}.xlsx")
        sample = np.expand_dims(sample, axis=2)
        labelled_set = np.concatenate((labelled_set, sample), axis=2)
    np.save("labels", labelled_set)

#Testing Purposes

def param_to_func(param):
    #Given (28,6) return the signal
    integs = []
    for k in range(param.shape[0]):
        f, w_sgn, w_trig, w_period, w_ps, intsgn = param[k]
        func = lambda t, f=f, w_sgn=w_sgn, w_period=w_period, w_ps=w_ps : f + w_sgn*np.cos(t*w_period + w_ps)
        func_int = lambda x, intsgn=intsgn, func=func : intsgn*np.cos(integrate.quad(func, 0, x)[0]) #if x >= 0 else intsgn*cp.cos(integrate.quad(func, x,0 )[0])
        integs.append(func_int)

    signal_sum = np.vectorize(lambda t : np.sum([func(t) for func in integs]))
    # @cp.vectorize()
    # def signal_sum(t):
    #     return cp.reduce([func(t) for func in integs])
    x = np.linspace(-500,500, 3600)
    # x_1, x_2, x_3, x_4, x_5, x_6 = np.split(x,6)
    # with cp.cuda.Device(0):
    #     y_1 = signal_sum(x_1)
    # with cp.cuda.Device(1):
    #     y_2 = signal_sum(x_2)
    # with cp.cuda.Device(2):
    #     y_3 = signal_sum(x_3)
    # with cp.cuda.Device(3):
    #     y_4 = signal_sum(x_4)
    # with cp.cuda.Device(4):
    #     y_5 = signal_sum(x_5)
    # with cp.cuda.Device(5):
    #     y_6 = signal_sum(x_6)
    # y = cp.concatenate((y_1, y_2, y_3, y_4, y_5, y_6))
    y = signal_sum(x)
    return np.array(y)

def construct_signals():

    parameters = np.load("labels.npy") #28,6,1000
    input_data = np.empty((0,3600))
    print("Begin Constructing Signal Data...\n")
    for i in range(parameters.shape[-1]):
        param = parameters[:,:,i]
        signal_data = param_to_func(param)
        input_data = np.vstack((input_data, signal_data))
        np.save("signals.npy", input_data)
        print(f"Saved signal sample index : ===== {i}/{range(parameters.shape[-1])}", end="\r")
    return
def test_model():
    saved_model = tf.keras.models.load_model("test_model.h5")
    index = 321
    
    signal = np.load("signals.npy")[index:index+32]
    labels = np.load("labels.npy")[:,:,index:index+32]

    print(f"Running Inference on {signal.shape[0]} signals\n")
    print(f"Label dimensions : {labels.shape}\n")
    pred = saved_model.predict(signal)
    print(f"Prediction Shape: {pred.shape}")
    parameters = pred[0]
    label = labels[:,:,0]
    x = np.linspace(12,60, 100)

    for vals in parameters:
        curve = lambda t : vals[0] + vals[2]*np.cos(vals[1]*t + vals[4])
        y = [curve(i) for i in x]
        plt.plot(x,y)
    plt.savefig("inference.png")
    plt.clf()
    for vals in label:
        curve = lambda t : vals[0] + vals[2]*np.cos(vals[1]*t + vals[4])
        y = [curve(i) for i in x]
        plt.plot(x,y)
    plt.savefig("label.png")

def convert_labels():
    labels = np.load("labels.npy")
    labels = np.transpose(labels, (2,0,1))
    x = np.linspace(12,60, 100)
    for i in range(labels.shape[0]):
        print(f"Processing sample: {i}/{labels.shape[0]}", end="\r")
        label = labels[i]
        for vals in label:
            curve = lambda t : vals[0] + vals[2]*np.cos(vals[1]*t + vals[4])
            y = [curve(i) for i in x]
            plt.plot(x,y)
        plt.ylim(0,25.5)
        plt.savefig(f"./label_png/label_{i}.png")
        plt.clf()
def generate_labels_from_png():
    label_dir = "./label_png/"
    labels = os.listdir(label_dir)
    label_array = np.empty((356,474,0))
    for file_name in labels:
        label = cv2.imread(os.path.join(label_dir, file_name), 0)
        cropped = (label[62:418, 86:560] != 255).astype(int)
        cropped = np.expand_dims(cropped, axis=2)
        label_array = np.concatenate((label_array, cropped), axis=2)
        print(f"Current Iteration: {label_array.shape[-1]}", end="\r")
    # np.save("spectrogram_arr.npy", label_array)
def get_data():
    
    signal_data, labels = np.load("signals.npy"), np.load("spectrogram_arr.npy")

    # norm_labels = np.transpose(labels, (2,0,1))
    # norm_data = normalize(signal_data)

    x_train, x_test, y_train, y_test = train_test_split(signal_data, norm_labels, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test
def test_CNN(model_name, idx):
    model = tf.keras.models.load_model("./models/" + model_name)

    signal = np.load("signals.npy")
    label = np.load("spectrogram")[idx]

    pred = model.predict(signal)
    print(pred.shape)




    
