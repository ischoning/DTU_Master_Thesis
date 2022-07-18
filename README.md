# DTU_Master_Thesis
Predicting motor errors from gaze data using a neural network trained on sequences of detected eye movements.
15 July 2022
Isabella A. Moreno Schøning

This project is the source code for the above-mentioned Master thesis. The thesis has been prepared over five months at the Section for Cognitive Systems,
Department of Applied Mathematics and Computer Science, at the Technical University of Denmark, DTU, in partial fulfillment for the degree Master of Science in Engineering, MSc Eng.

The purpose of the project is determine to what degree eye movements detected from Varjo XR-3 gaze data can predict motor errors. To do so, a novel event detection method is proposed which applies pre-processing, three Gaussian mixture models, two corrective algorithms, and physiological checks following figure 3.3 in Schoning_Master_Thesis_2022.pdf. The detected event sequences and features are then fed into a 2-D convolutional neural network.

## Event Detection
Detects fixations, smooth pursuits, saccades and blinks given positional time-series gaze data from the Varjo XR-3 eye tracker. Output is a csv of concatenated event sequences with features.

1. Gather the data (EVENT_DETECT/data_prep)

2. Pre-process the data (EVENT_DETECT/run_preprocess.ipynb)

3. Run event detection (EVENT_DETECT/run_event_detect_full.ipynb)

## Motor Error Prediction
Supervised bi-classification of motor errors given 18-event sequences and labelled data.

4. Gather the labelled data and experiment start/end times (NN/data_prep)

5. Run CNN on all participants (one at a time) (NN/run_CNN_by_subject.ipynb)

## analysis
Analysis of event detection, motor error data, and the CNN.

## Explanation of Data
The data contains personal information. Due to protection laws, the data is not made public. It can only be accessed via a secure DTU server.
