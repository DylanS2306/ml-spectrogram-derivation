import tensorflow as tf
import numpy as np
import argparse
from neural_net import normalize
import sys
import os
import matplotlib.pyplot as plt
import  cv2

def parse_args(argv):
    parser = argparse.ArgumentParser(description="test cnn model perfomance")
    
    parser.add_argument('-m', '--model', type=str, required=True, help="model_name to pass")
    parser.add_argument('-i', "--idx", type=int, required=True, help="index of label to check")
    args = parser.parse_args(argv)
    
    return args
def main(argv=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    if argv == None:
        args = parse_args(sys.argv[1:])
    
    model_path = "./models/" + args.model + ".h5"
    idx = args.idx
    signals = np.load("signals.npy")
    label = np.load("spectrogram_arr.npy")

    signals_norm = normalize(signals)
    signal = signals_norm[idx]

    model = tf.keras.models.load_model(model_path)
    norm_signal = np.reshape(signal, (1,3600,1))


    pred = model.predict(norm_signal)

    im = np.reshape(pred, (356,474))
    thresh = np.mean(im.flatten())
    # thresh = np.median(im.flatten())

    im = np.where(im > thresh,1, 0)
    plt.imshow(im, cmap='binary')
    plt.savefig(args.model + "-" + str(idx) + "-" + "threshMean" ".png")

    plt.clf()
    plt.imshow(label[:,:,idx], cmap='binary')
    plt.savefig(str(idx) + "-label.png")
if __name__ == "__main__":
    main()