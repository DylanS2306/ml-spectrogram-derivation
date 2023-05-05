# ml-spectrogram-derivation

For generation of models in .h5 format run neural_net.py 
Specify model on line 148 from larger_CNN, baseline_CNN etc.

To test model predictions, run test_cnn.py -m MODEL_NAME -i IDX
and pass the name of the .h5 model file and the index of the signal in the range [0,1000] that you want to test results for
This will output a label image in the format IDX-label.png and MODEL_NAME.png

the .wls file is used as the spectrogram generator for RL framework, however this is too computationally expensive to run for episodes
Wolframengine is a required installation for .wls script to run.

