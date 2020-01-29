#Project: Predicting breast cancer survival from expression data using deep learning
#Laboratorio 19 CIC (Centro de Investigación del Cáncer, Salamanca)
#January 2020 
#Fernando Bueno Gutierrez
#fernando.bueno.gutierrez@usal.es
#fernando.bueno.gutie@gmail.com


#This script contains functions to prepare data for My_Neural_Network.R


data_autoscaling<-function(mdata){
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

get_msamples<-function(minput){
  #minput: int, refers toi the set of samples
  #msamples: numeric where each element is a sample index from mdata
  
  mfirst_sample_in_the_minput_minibin=minibins_first_samples[minput]
  
  if(minput==minibins_first_samples[length(minibins_first_samples)]){
    msamples=seq(minibins_first_samples[length(minibins_first_samples)],binsize,1)
  } else {
    msamples=seq(mfirst_sample_in_the_minput_minibin,mfirst_sample_in_the_minput_minibin+binsize/nsets-1)
  }
  
  return(msamples)
}


source_functions<-function(mypath){
  #DATA preparation
  #downloads mnist data
  #transforms expected results to dummy
  #makes data simplifications
  #creates random weights and bias
  
  #GET OUTPUT and cost (compute neuron's states from the left to the right)
  #computes zs and states and assigns them to the global environment 
  #prints average cost across samples so you can know when the data is trainned
  source(paste0(mypath,"/predictions_functions.R"))
  
  #BACKPROPAGATION to get the gradients
  source("/data/dtFernando/NN/mnist_NN/backprop_functions.R")
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



download_mnist_data<-function(){
  #datatype: char that is "train" or "test"
  #download mnist data
  #https://www.r-bloggers.com/exploring-handwritten-digit-classification-a-tidy-analysis-of-the-mnist-dataset/
  #data with 60000 observations and 784 independent variables and X1 as dependent variable.. Independent variables take values from 0 (most likelly to ~300 i.e. refering to how dark it is)
  
  library(readr);library(dplyr);library(pracma)#pracma for sigmoid
  mnist_raw <- read_csv("https://pjreddie.com/media/files/mnist_train.csv", col_names = FALSE)#X_784 (60000 x 784) with values between 0(i.e. white) and 1 (i.e. black)
  mnist_raw_test <- read_csv("https://pjreddie.com/media/files/mnist_test.csv", col_names = FALSE)#X_784 (10000 x 784) with values between 0(i.e. white) and 1 (i.e. black)

  X_784_raw=as.data.frame(mnist_raw[,2:ncol(mnist_raw)])
  Y=as.data.frame(mnist_raw[,1])#Y (60000 x 1): expected results
  
  X_784_raw_test=as.data.frame(mnist_raw_test[,2:ncol(mnist_raw_test)])
  Y_test=as.data.frame(mnist_raw_test[,1])#Y (60000 x 1): expected results
  
  saveRDS(X_784_raw,"/data/dtFernando/NN/X_784.rds")
  saveRDS(Y,"/data/dtFernando/NN/Y.rds")
  saveRDS(X_784_raw_test,"/data/dtFernando/NN/Xtest_784.rds")
  saveRDS(Y_test,"/data/dtFernando/NN/Ytest.rds")
}





load_mnist_data<-function(datatype="train"){
  #
  if(datatype=="train"){
    Y=readRDS("/data/dtFernando/NN/Y.rds")
  }
  if(datatype=="test"){
    Y=readRDS("/data/dtFernando/NN/Ytest.rds")
  }
  each_neuron_output<-create_output_for_each_neuron(Y,max_output_value=9)#Run this line and next for the complete data
  nneuronsL<<-ncol(each_neuron_output)
  if(datatype=="train"){
    X=readRDS("/data/dtFernando/NN/X_784.rds")
  }
  if(datatype=="test"){
    X=readRDS("/data/dtFernando/NN/Xtest_784.rds")
  }  
  
  nneuronsLm3<<-ncol(X)#Run this line and next for the complete data
  mdata=cbind(each_neuron_output,X)#this should have 60000 rows and 10+784 columns, where the first 10 columns are called Y_N0, YN1... YN9 and take 0 or 1 depending on the expected output for N for the sample. The other 784 columns contain int corresponding to the input data
  return(mdata)
}



load_small_mnist_data<-function(datatype="train"){
  #
  if(datatype=="train"){
    Y=readRDS("/data/dtFernando/NN/Y.rds")
  }
  if(datatype=="test"){
    Y=readRDS("/data/dtFernando/NN/Ytest.rds")
  }
  each_neuron_output<-create_output_for_each_neuron(Y,max_output_value=9)#Run this line and next for the complete data
  nneuronsL<<-ncol(each_neuron_output)
  if(datatype=="train"){
    X=readRDS("/data/dtFernando/NN/X_100samples_784neurons.rds")
  }
  if(datatype=="test"){
    X=readRDS("/data/dtFernando/NN/Xtest_784.rds")
  }  
  
  nneuronsLm3<<-ncol(X)#Run this line and next for the complete data
  mdata=cbind(each_neuron_output,X)#this should have 60000 rows and 10+784 columns, where the first 10 columns are called Y_N0, YN1... YN9 and take 0 or 1 depending on the expected output for N for the sample. The other 784 columns contain int corresponding to the input data
  return(mdata)
}


prepare_whole_data<-function(){
  #subsets nsamples from the X and Y datas and prepares it for the network. The ready data is stored in /data/dtFernando/NN/ and indicates the number of samples
  #create a dataset with nsamples from which the bins of 100 will be taken
  #nsamples:int: number of samples
  
  #load data
  X=readRDS("/data/dtFernando/NN/X_784.rds")#this already was sigmoid (to get values bt 0 and 1 for th einput neurons)
  Y=readRDS("/data/dtFernando/NN/Y.rds")
  X_test=readRDS("/data/dtFernando/NN/Xtest_784.rds")#this already was sigmoid (to get values bt 0 and 1 for th einput neurons)
  Y_test=readRDS("/data/dtFernando/NN/Ytest.rds")
  
  #Get dummyes for Y
  each_neuron_output<-create_output_for_each_neuron(Y,max_output_value=9)
  each_neuron_output_TEST<-create_output_for_each_neuron(Y_test,max_output_value=9)
  
  
  #combine Y and X into a single df
  mdata=cbind(each_neuron_output,X)
  mdataTEST=cbind(each_neuron_output_TEST,X_test)
  
  
  #name the data
  mfilename=paste0("/data/dtFernando/NN/fullTrainData.rds")
  mfilename_test=paste0("/data/dtFernando/NN/fullTestData.rds")
  
  
  #save data
  saveRDS(mdata,mfilename)
  saveRDS(mdataTEST,mfilename_test)
  
}


create_data_of_nsamples<-function(nsamples){
  #subsets nsamples from the X and Y datas and prepares it for the network. The ready data is stored in /data/dtFernando/NN/ and indicates the number of samples
  #create a dataset with nsamples from which the bins of 100 will be taken
  #nsamples:int: number of samples
  
  #load data
  X=readRDS("/data/dtFernando/NN/X_784.rds")#this already was sigmoid (to get values bt 0 and 1 for th einput neurons)
  Y=readRDS("/data/dtFernando/NN/Y.rds")
  
  #subset nsamples
  Y=as.data.frame(Y[sample(1:nrow(X),nsamples),1])
  colnames(Y)<-"X1"
  X=X[sample(1:nrow(X),nsamples),]
  
  #Get dummyes for Y
  each_neuron_output<-create_output_for_each_neuron(Y,max_output_value=9)
  
  #combine Y and X into a single df
  mdata=cbind(each_neuron_output,X)
  
  #name the data
  mfilename=paste0("/data/dtFernando/NN/mcomplete_data_",nsamples,"samples.rds")
  
  #save data
  saveRDS(mdata,mfilename)
}



extract_samples_bin<-function(binsize){
  #sample with replacement
  #binsizse:int, number of samples per bin
  
  #OUT: mdata: same as mdata but with only binsizse samples (rows)
  idx_selected_samples=sample(1:nrow(m_complete_data),binsize)
  mdata=m_complete_data[idx_selected_samples,]
  
  return(mdata)
  
}
