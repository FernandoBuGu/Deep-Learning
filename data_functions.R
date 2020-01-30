#Project: Predicting breast cancer survival from expression data using deep learning
#Laboratorio 19 CIC (Centro de Investigación del Cáncer, Salamanca)
#January 2020 
#Fernando Bueno Gutierrez
#fernando.bueno.gutierrez@usal.es
#fernando.bueno.gutie@gmail.com


#This script contains functions to prepare data for My_Neural_Network.R


data_preprocessing<-function(mdata){
  #removes columns where values for all samples are 0, and autoscales the predictors columns
  
  #mdata (input):df, samples are in rows. First "nneuronsL" columns are the dummy variables for the label
  #mdata (output):df, that may have fewer columns as input (columns with 0 in all rows are removed). Also the predictors have been autoscaled
  
  #Get from the data the number of neurons in the last layer
  nneuronsL=sum(str_count(colnames(mdata),"Y"))#All output neurons in mdata start with Y (inputs start with X)
  
  #autoscale the input data
  X<-mdata[,(nneuronsL+1):ncol(mdata)]
  scaledX=autoscale(X,exclude = F)
  scaledX[is.nan(scaledX)] <- 0#autoscale gives NAN instead of 0 for variables for which all samples were 0
  
  #cbind the output data with the autoscaled input data
  Y<-mdata[,1:nneuronsL]
  mdata<-cbind(Y,scaledX)
  
  return(mdata)
  
}



source_functions<-function(mypath){
  #loads the functions required to train and test the network
  
  #mypath: char, directory where data_functions.R, predictions_functions.R and backprop_functions.R are
  
  #computes zs, states, cost and percentage of samples properly allocated
  source(paste0(mypath,"/predictions_functions.R"))
  
  #BACKPROPAGATION to get the gradients
  source(paste0(mypath,"/backprop_functions.R"))
}



create_output_for_each_neuron<-function(Y,max_output_value=9){
  #transforms categorical expected output to dummy, so one value for each possible output neuron. "1"for the expected neuron, "0"for all others
  
  #Y: df with 1 single column called X1. Each row is a sample and the value corresponds to the only neuron that should get 
      #value "1" for that sample. The other neurons should get "0" for that sample.
  #max_output_value: int, maximum  value that an output can have. HARD CODED TO SEQUENTIAL FROM 0 TO max_output_value.
  
  #each_neuron_output: df with nrow=nrow(Y)=nsamples and ncol=max(Y). That is, for every value in Y$X1, a column will be created
  
  each_neuron_output=data.frame()
  
  for(sample in 1:nrow(Y)){
    mvalue=Y$X1[sample]
    prev=rep(0,mvalue)
    post=rep(0,max_output_value-mvalue)
    sample_output=c(prev,1,post)
    each_neuron_output<-rbind(each_neuron_output,sample_output)
  }
  colnames(each_neuron_output)<-paste0("Y_N",0:max_output_value)
  return(each_neuron_output)
}



download_mnist_data<-function(mypath){
  #download mnist data and store it in 
  #data with 60000 observations and 784 independent variables and X1 as dependent variable.. Independent variables take values from 0 (most likelly to ~300 i.e. refering to how dark it is)
  
  #mypath: char, path where downloaded data will be stored

  library(readr);library(dplyr);library(pracma)#pracma for sigmoid
  mnist_raw <- read_csv("https://pjreddie.com/media/files/mnist_train.csv", col_names = FALSE)#X_784 (60000 x 784) with values between 0(i.e. white) and 1 (i.e. black)
  mnist_raw_test <- read_csv("https://pjreddie.com/media/files/mnist_test.csv", col_names = FALSE)#X_784 (10000 x 784) with values between 0(i.e. white) and 1 (i.e. black)

  X_784_raw=as.data.frame(mnist_raw[,2:ncol(mnist_raw)])
  Y=as.data.frame(mnist_raw[,1])#Y (60000 x 1): expected results
  
  X_784_raw_test=as.data.frame(mnist_raw_test[,2:ncol(mnist_raw_test)])
  Y_test=as.data.frame(mnist_raw_test[,1])#Y (60000 x 1): expected results
  
  saveRDS(X_784_raw,paste0(mypath,"/X_784.rds"))
  saveRDS(Y,paste0(mypath,"/Y.rds"))
  saveRDS(X_784_raw_test,paste0(mypath,"/Xtest_784.rds"))
  saveRDS(Y_test,paste0(mypath,"/Ytest.rds"))
}




load_mnist_data<-function(mypath){
  #return a list with two df of names: mdatatrain and mdatatest. These are ready to use by the programme. However, this is slow, so by doing save_MNIST_data() after, the list will be saved
  
  #mypath: char, directory where MNIST downloaded data is. MNIST data can be downloaded with the function download_mnist_data(mypath)
  
  #load the dependent variable (network expected output)
  Y=readRDS(paste0(mypath,"/Y.rds"))
  Ytest=readRDS(paste0(mypath,"/Ytest.rds"))

  #convert categorical to dummy
  each_neuron_output<-create_output_for_each_neuron(Y,max_output_value=9)
  each_neuron_output_TEST<-create_output_for_each_neuron(Ytest,max_output_value=9)
  
  #load predictors data
  X=readRDS(paste0(mypath,"/X_784.rds"))
  Xtest=readRDS(paste0(mypath,"/Xtest_784.rds"))

  #cbind output and input data
  mdatatrain=cbind(each_neuron_output,X)#this should have 60000 rows and 10+784 columns, where the first 10 columns are called Y_N0, YN1... YN9 and take 0 or 1 depending on the expected output for N for the sample. The other 784 columns contain int corresponding to the input data
  mdatatest=cbind(each_neuron_output_TEST,Xtest)#this should have 60000 rows and 10+784 columns, where the first 10 columns are called Y_N0, YN1... YN9 and take 0 or 1 depending on the expected output for N for the sample.
  
  return(list("mdatatrain"=mdatatrain,"mdatatest"=mdatatest))
}



save_MNIST_data<-function(mypath){
  #saves two files: "/fullTrainData.rds" and "/fullTestData.rds" in mypath. These are ready to use by the programme
  
  #mypath: char, directory where MNIST downloaded data is. MNIST data can be downloaded with the function download_mnist_data(mypath)
  
  #load data
  X=readRDS(paste0(mypath,"/X_784.rds"))#this already was sigmoid (to get values bt 0 and 1 for th einput neurons)
  Y=readRDS(paste0(mypath,"/Y.rds"))
  X_test=readRDS(paste0(mypath,"/Xtest_784.rds"))#this already was sigmoid (to get values bt 0 and 1 for th einput neurons)
  Y_test=readRDS(paste0(mypath,"/Ytest.rds"))
  
  #Get dummyes for Y
  each_neuron_output<-create_output_for_each_neuron(Y,max_output_value=9)
  each_neuron_output_TEST<-create_output_for_each_neuron(Y_test,max_output_value=9)
  
  #combine Y and X into a single df
  mdata=cbind(each_neuron_output,X)
  mdataTEST=cbind(each_neuron_output_TEST,X_test)
  
  #name the data
  mfilename=paste0(mypath,"/fullTrainData.rds")
  mfilename_test=paste0(mypath,"/fullTestData.rds")
  
  #save data
  saveRDS(mdata,mfilename)
  saveRDS(mdataTEST,mfilename_test)
  
}



extract_samples_bin<-function(binsize){
  #sample with replacement from m_complete_data (so, the training set)
  #binsizse:int, number of samples per bin
  
  #OUT: mdata: same as mdata but with only binsizse samples (rows)
  idx_selected_samples=sample(1:nrow(m_complete_data),binsize)
  mdata=m_complete_data[idx_selected_samples,]
  
  return(mdata)
  
}
