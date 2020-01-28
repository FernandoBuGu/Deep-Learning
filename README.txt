Project: Predicting breast cancer survival from expression data using deep learning
Laboratorio 19 CIC (Centro de Investigación del Cáncer, Salamanca)
January 2020 
Fernando Bueno Gutierrez
fernando.bueno.gutierrez@usal.es
fernando.bueno.gutie@gmail.com


A_My_Neural_Network.R is the main script and runs a deep learning classifier given supervised data. Use instructions are given in the same script
The classifier was constructed from scratch (without R Deep Learning packages) and works with >90% accuracy for the MNIST data (60000 samples x 784 predictor variables)

data_functions.R, predictions_functions.R and backprop_functions.R contain the functions used by the script, and should be allocated in a place where they can be called by A_My_Neural_Network.R

If the user does not have its own dataset, the MNIST data ready to be loaded and used by A_My_Neural_Network.R is provided in the same repository: fullTrainData.rds. 

This data was downloaded from pjreddie.com and was processed with functions in data_functions.R:
mnist_raw <- read_csv("https://pjreddie.com/media/files/mnist_train.csv", col_names = FALSE)
