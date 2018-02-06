# conditiona GAN for medical image semantic segmentation

Requirements

python modules

    numpy==1.13.3
    Keras==2.0.8
    parmap==1.5.1
    matplotlib==2.0.2
    tqdm==4.17.0
    opencv_python==3.3.0.10
    h5py==2.7.0
    theano==0.9.0 or tensorflow==1.3.0

Part 1. Processing the data


Part 2. Running the code
python brain.py

Part 3. Example results


We train and test using a seperate file for every axis. 
The main implementation can be found in shared.py. 
This file exports functions that can be used from each axis module. 
These functions take a class as a parameter that contains every customizable function.
