#Project: Predicting breast cancer survival from expression data using deep learning
#Laboratorio 19 CIC (Centro de Investigación del Cáncer, Salamanca)
#January 2020 
#Fernando Bueno Gutierrez
#fernando.bueno.gutierrez@usal.es
#fernando.bueno.gutie@gmail.com


#This script runs a deep learning classifier given supervised data

#The classifier was constructed fully from scratch without Deep Learning Packages

#DATA should be in the following format:
  #R data.frame with samples as rows
  #Firts column is the expected categorical output. The remianing columns are the quantitative raw input data

#COMPUTATIONAL TIME is around 5 hours (0.7 seconds per iteration) for the MNIST data with 2 intermediate layers of 16 neurons each (as in https://www.youtube.com/watch?v=aircAruvnKk&t=743s). 
#MNIST data is open access: 60000 samples, 874 input variables, output with 10 categories.
#The program will run in parallel using as many nodes as the user specifies.

#THE NETWORK SETTINGS can be any number of layers and neurons per layer.


################################
####  User specifications  ##### 
################################

#PACKAGES install or call 
library(pracma);library(e1071)#for sigmoid and dsigmoid, respectivelly
library(ddpcr)#for quiet(), silence warnings
library(parallel)#for paralel computation
library(RFmarkerDetector)#for autoscaling the data
library(stringr)#for str_count, that counts number of strings with pattern

#SPECIFY DIRECTORY where functions are and load them to the main script
mypath="/data/dtFernando/NN/mnist_NN"
source(paste0(mypath,"/data_functions.R"))
source_functions(mypath)#load the functions

#LOAD DATA
m_complete_data=readRDS("/data/dtFernando/NN/fullTrainData.rds")#read data

#remove variables with 0 for all samples
m_complete_data=m_complete_data[, sapply(m_complete_data, var) != 0]

#SPECIFY NUMBER OF CORES to be used. By default all cores from the machine will be used. Specify coresLeftOut=X to leave X cores free
cores=detectCores()#detect the number of cores in machine
coresLeftOut=0#number of cores that will ot be used
nsets=cores[1]-coresLeftOut#number of sets among which to split the X training samples. Each set will be run in a core

#SPECIFY BIN SIZE
binsize=100#number of samples per bin. A random bin will be used for each iteration

#SPECIFY NUMBER OF INTERMEDIATE LAYERS AND NEURONS
intermediatelayers<-c(16,16)#for 2 layers of 16 neurons each. The vector length in the number of layers and can be as long as desired

#SELECT MAXIMUM (AVERAGE) COST ALLOWED
max_overall_cost_allowed=0.3#average cost (least squares difference) over all the samples in bin






#####################################################
####  Runnng the classifier after user defined  ##### 
#####################################################
#Take a bin of X random training samples from the complete dataset
mdata=extract_samples_bin(binsize)

#removes variables with 0 in all samples and autoscales the predictors data
mdata=data_autoscaling(mdata)

#Get from the data the number of neurons in the input layer
nneuronsInput=ncol(mdata)-nneuronsL

#Define a vector with the number of neurons in each layer. 1st element is the numbver of neurons in the input layer, 2nd element is number of neurons in the first layer...
n_neurons<-c(nneuronsInput,intermediatelayers,nneuronsL)
n_layers=length(n_neurons)-1#number of layers in the newtork

#compute states and cost. Iteration 0
wba=get_wba(iteration0="Yes",mdata)#wba: a list containing weights, bias and states in each of the layers (i.e. wba$Ws is a list of length equal to the number of layers. Each element in this inner list is a matrix)
cost=compute_average_cost(wba$As[[n_layers]],mdata)#an integer with the average cost across the samples in bin. The input is the state computed with the previous get_wba call, from the last layer

#Update network parameters. Iteration 0. The update is done layer by layer (from right to left) to minimize previous "cost" (least squares differences between expected and predicted outcome)
backpropagation()

#Measure time used to train the network
start_time_999 <- Sys.time()


############################
####  iterative trainnng ###
############################
# iterativelly update weights and bias and compute cost until the cost is below max_overall_cost_allowed
while(cost>max_overall_cost_allowed){
  
  start_time <- Sys.time()#Measure time of each iteration
  
  #1)sample with replacement in binsize for each iteration. For each bin, remove input variables that have 0 for all samples in bin, and update the n_neurons vectors (intermediate and last layers remain same)
  mdata=extract_samples_bin(binsize)
  mdata=data_autoscaling(mdata)
  
  #2)compute states (from left to right) given mdata and the parameters that were updated in the previous backpropagation call
  wba=get_wba(iteration0="No",mdata)#wba: a list containing weights, bias and states in each of the layers (i.e. wba$Ws is a list of length equal to the number of layers. Each element in this inner list is a matrix)
  
  #3)compute cost given the states compute din the last get_wba call. This cost should decrease and eventually stop the while loop
  cost=compute_average_cost(wba$As[[n_layers]],mdata)
  
  #4)update weights and bias no minimize the cost in the last layer (always with respect to the original tags)
  backpropagation()

  #print iteration timespan
  end_time <- Sys.time()
  print(end_time - start_time)
  print(cost)
  
}

#Print total network training  timespan
end_time_999 <- Sys.time()
print(end_time_999 - start_time_999)

#save the trained network (weights, bias and predictions)
wba=get_wba(iteration0="No",mdata)
saveRDS(wba,"/data/dtFernando/NN/mnist_NN/wba_trained.rds")




##################################
## test prediction performance ###
##################################
wba_test=get_wba(iteration0="No",mdata)
As_last_layer<<-wba_test$As[[n_layers]]
percentage_success(As_last_layer)
