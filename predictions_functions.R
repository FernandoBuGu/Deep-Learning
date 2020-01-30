#Project: Predicting breast cancer survival from expression data using deep learning
#Laboratorio 19 CIC (Centro de Investigación del Cáncer, Salamanca)
#January 2020 
#Fernando Bueno Gutierrez
#fernando.bueno.gutierrez@usal.es
#fernando.bueno.gutie@gmail.com


#This script contains functions used by My_Neural_Network.R to compute neuron prediction values (states).



create_random_parameters<-function(n_neurons){
  #Creates random weights and bias for the network. All these are samples from a uniform distribution bt -5 and 5, and returns a list with two lists: Ws and Bs for weighst and bias
  
  #n_neurons: vect with number_of_layers+1 elements. Each element is a number of neurons. The first element is the number of neurons in the input layer, the second elment is n_neurons in first layer and the last element is the n_neurons inthe last layer
  #n_layers: int, number of layers
  
  Ws=list()#List that will contain one matrix of weights per layer
  Bs=list()#List that will contain one numeric of bias per layer
  
  for(layer in 1:n_layers){
    #if(layer==1){set.seed(123)} else {set.seed(456)}
    random_weights=rnorm(n_neurons[layer]*n_neurons[layer+1],0,2)#n_neurons[layer]: number of neurons in the previous layer, since n_neurons starst at layer 0 (input)
    # random_weights=sqrt(random_weights*random_weights)#positive weights
    W=matrix(random_weights,nrow=n_neurons[layer])#W: mat (n_neurons_in_previous_layer x n_neurons_in_layer)
    Ws[[layer]]<-W
    
    #if(layer==1){set.seed(111)} else {set.seed(222)}
    random_bias=rnorm(n_neurons[layer+1],0,2)
    # random_bias=sqrt(random_bias*random_bias)#positive bias
    B=matrix(random_bias,ncol=n_neurons[layer+1])
    B=do.call("rbind", replicate(nrow(mdata), B, simplify = FALSE))#?
    Bs[[layer]]<-B[1,]#? if you select the ONLY the firts row, the previous line is useless
  }
  
  return(list("Ws"=Ws,"Bs"=Bs))
}




get_wba<-function(iteration0="No",mdata=mdata){
  #get a list with 3 names: Ws (weights), Bs (bias) and As (States). Each of these will has many elements as layers there are 
  #(i.e. after runnning this function, you can access the weights connecting layers 2-3 with wba$Ws[[2]] 
  
  #n_neurons: vect with number_of_layers+1 elements. Each element is a number of neurons. The first element is the number of neurons in the input layer, the second elment is n_neurons in first layer and the last element is the n_neurons inthe last layer
  #n_layers: int, number of layers
  
  Zs=list()#An empty list where the Z matrix (one per layer) will be allocated
  As=list()#An empty list where the states matrix (one per layer) will be allocated
  
  if(iteration0=="Yes"){
    Param<-create_random_parameters(n_neurons)#Param (parameters): a list with two elements: Ws (weights) and Bs (bias), each of these will has many elements as layers there are
    Ws<-Param$Ws  
    Bs<-Param$Bs
  } else {
    Ws<-wba$Ws#get weights from the wba oject created in the previous iteration
    Bs<-wba$Bs#defined in the global environment cause it will be used in computeZ
  }
  
 
  
  states_previous<<-as.matrix(mdata[,seq(nneuronsL+1,ncol(mdata))])#initial states directly from mdata (for layer 1), for the following layers it will be redefined (5 lines below)
  #defined in the global environment because it is used in computeZ/compute_product and since this is run with mcmapply, only integers can be given as inputs
  
  for(layer in 1:n_layers){
    
    BIASMATRIX=do.call("rbind", replicate(nrow(mdata), Bs[[layer]], simplify = FALSE))
    
    Z=states_previous%*%Ws[[layer]]+BIASMATRIX#Z: mat (nsamples x n_neurons_in_layer), is the sum of the products weights by states_previous plus de bias. For each sample, there will be as many Zs as neurons in layer. Firts row is firts sample
    Zs[[layer]]<-Z#Append the Z matrix to Zs list
    A=sigmoid(Z)#A: mat (nsamples x n_neurons_in_layer), is the states of the neurons. For each sample, there will be as many states as neurons in layer. Firts row is firts sample
    As[[layer]]<-A#Append the states matrix to As list
    states_previous<<-A#Redefine states previous for the next computeZ call
  }
  
  return(list("Ws"=Ws,"Bs"=Bs,"As"=As, "Zs"=Zs))
  
}




compute_average_cost<-function(As_last_layer,mdata=mdata){
  #returns the average netwrok cost across samples in one of the learning iteration
  
  #As_last_layer: mat (n_samples x n_neurons_in_last_layer)
  #mean(sum_cost_neurons): int, average cost across samples in bin. This was computed through least squares differences of states in the last layer and given tags
  
  expected=mdata[,seq(1,nneuronsL)]#mat (n_samples x n_neurons_in_LASTlayer): Tags directly extracted from data
  cost=(As_last_layer-expected)^2#mat (n_samples x n_neurons_in_LASTlayer). Least squares differences
  sum_cost_neurons=rowSums(cost)#num of length n_samples, each element is the cost for a sample, and to compute the cost of a sample, the cost of all nerons in the last layer has been summed up
  
  return(mean(sum_cost_neurons))#the final cost for a given iteration is the average across samples
  
}



percentage_success<-function(As_last_layer){
  #returns an integer that is the percentage of samples whose label was correctly predicted by the NN 
  
  #As_last_layer: df with rows as samples and states of output neurons as columns
  
  expected=mdata[,seq(1,nneuronsL)]
  
  samples_tested<<-0
  successes<<-0
  
  res=mcmapply(annotate_sample_result,1:nrow(mdata),mc.cores=nsets)
  
  return(sum(res)*100/length(res))
  
}



annotate_sample_result<-function(sample_idx){
  #retrun a 1 if the sample label was properly precdicted or a 0 otherwise
  
  #sample_idx: int, index of the sample with respect to mdata rows
  
  mrow_expected=expected[sample_idx,]
  m_expected=which(mrow_expected==1)
  
  mrow_predicted=As_last_layer[sample_idx,]
  m_predicted=which(mrow_predicted==max(mrow_predicted))
  
  
  if(m_expected==m_predicted){
    return(1)
  } else {
    return(0)
  }
  
}
