import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pywt

class DataGenerator:
    def __init__(self, batch_size, config, samples=1000):
        self.batch_size=batch_size
        self.config = config
        self.samples=samples
    def generate_data(self):
        if self.config =='train':
            start, end = int(0*self.samples), int(0.8*self.samples)        
        elif self.config == 'test':
            start, end = int(0.8*self.samples),int(0.9*self.samples)
        while True:
    
            signals = np.load('signals.npy')[start:end]
            spectrograms = np.load('spectrogram_arr.npy')[:,:,start:end]
            indices = np.arange(signals.shape[0])
            np.random.shuffle(indices)
            for i in range(0,signals.shape[0], self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_signals = signals[batch_indices]
                batch_spectrograms = spectrograms[:,:,batch_indices]
                # Preprocess the data if necessary (e.g., normalize, standardize)
                batch_signals = normalize(batch_signals)
                # print(f"{self.config} Batch Spectrogram Shape: {batch_spectrograms.shape}")
                batch_spectrograms = np.transpose(batch_spectrograms, (2,0,1))
                # Yield the batch data

                yield batch_signals, batch_spectrograms
class ValidationGenerator:
    def __init__(self, batch_size, samples=1000):
        self.batch_size = batch_size
        self.samples = samples
    
    def generate_data(self):
        start, end = 0.8, 0.9
        start = int(start*self.samples)
        end = int(end*self.samples)
        signals = np.load('signals.npy')[start:end]
        spectrograms = np.load('spectrogram_arr.npy')[:,:,start:end]
        indices = np.arange(signals.shape[0])
        while True:
            for i in range(0, signals.shape[0], self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_signals = signals[batch_indices]
                batch_spectrograms = spectrograms[:,:,batch_indices]
                # Preprocess the data if necessary (e.g., normalize, standardize)
                batch_signals = normalize(batch_signals)
                # print(f"Validation Batch Spectrogram Shape: {batch_spectrograms.shape}")
                batch_spectrograms = np.transpose(batch_spectrograms, (2,0,1))
                # Yield the batch data
                yield batch_signals, batch_spectrograms         

def normalize(data):
    mean  = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    
    normalized_data = (data - mean)/std
    
    return normalized_data

def baseline_CNN(input_shape=None, output_shape=None):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=356*474, activation='sigmoid'),
        tf.keras.layers.Reshape((356,474))
    ])
    return model
def larger_CNN(input_shape=None, output_shape = None):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=output_shape, activation='sigmoid'),
        tf.keras.layers.Reshape((356,474))
        ])
        return model
def generate_scaleogram(signal):
    wavelet_name = 'mexh'
    C = 6
    fmin = 2*np.pi/(8*np.pi)
    fmax = 2*np.pi/(np.pi/4)
    scales = np.logspace(0.099, 15.915, 256)

    coeff, freq = pywt.cwt(signal, scales, wavelet_name, sampling_period=1)
    coeff_mesh, freqs_mesh = np.meshgrid(coeff, freq)
    scaleogram = np.stack([coeff_mesh, freqs_mesh], axis=2)
    print(scaleogram.shape)
    exit()

    return freqs

def wavelet_CNN(input_shape=None, output_shape=None):
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1024, activation='relu'),
            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=output_shape, activation='sigmoid'),
            tf.keras.layers.Reshape((356,474))
            ])
        return model


def train():
    mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:2"])
    num_epochs=100
    batch_size = 32

    with mirrored_strategy.scope():
        train_gen = DataGenerator(batch_size=batch_size, config='train')
        val_gen = ValidationGenerator(batch_size=batch_size)

        model = larger_CNN(input_shape=(3600,1), output_shape=356*474)
        model.compile(loss='binary_crossentropy', optimizer='adam')

        model.fit(train_gen.generate_data(), epochs=num_epochs, steps_per_epoch=800 // batch_size,validation_data=val_gen.generate_data(), validation_steps=100//batch_size, verbose=2)

    save_file = f"./models/large-cnn-dropout-{num_epochs}e-{batch_size}b.h5"
    model.save(save_file)
    return

def create_scaleogram():
    #mexh works kinda well
    wavelet_name = 'mexh'
    cmap = plt.cm.seismic
    dt = 1
    fmin = 2*np.pi/(8*np.pi)
    fmax = 2*np.pi/(np.pi/4)
    
    num_scales = 128
    step_size = 0.1
    C = 6

    scales = C / (2 * np.pi * np.logspace(np.log10(fmax), np.log10(fmin), 256))
    signal = np.load("signals.npy")[0]
    generate_scaleogram(signal)
    discrete_wavelets = ['db5', 'sym5', 'coif5', 'bior2.4']
    continuous_wavelets = ['mexh', 'morl', 'cgau5', 'gaus5']

    coefficients, frequencies = pywt.cwt(signal,scales, wavelet_name, dt)
    print(frequencies.shape)
    print(coefficients.shape)
    exit()
    print(f"CWT Coefficients: {len(coefficients)} \nCWT frequencies: {len(frequencies)}")
    power = (abs(coefficients))**2
    plt.imshow(power)
    plt.savefig("test2.png")
    return
    period = 1./frequencies
    # levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    levels = [1,2,4,8,16,32,64,128]
    levels = [8,16,32,64, 128]
    contourLevels = np.log2(levels)

    plt.contourf(np.arange(0,3600,1,dtype=int), np.log2(period), np.log2(power), contourLevels, extend="both", cmap=cmap)
    plt.savefig("scalogram_" + wavelet_name)
    # coef, freqs = pywt.cwt(signal, scales, 'morl')
    # spectrogram = np.abs(coef)
    # plt.imshow(spectrogram, cmap='viridis', aspect='auto')
    # plt.savefig("spectrogram_mexh.png")

if __name__ == "__main__":
    # convert_labels()
    # generate_labels_from_png()
    # test_model()
    # create_scaleogram()
    train()

# def FFN():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(128, activation="relu", input_shape =(3600,)),
#         tf.keras.layers.Dense(256, activation = "relu"),
#         tf.keras.layers.Dense(512, activation="relu"),
#         tf.keras.layers.Dense(28*6, activation="linear"),
#         tf.keras.layers.Reshape((28,6))
#     ])
#     model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#     return model
