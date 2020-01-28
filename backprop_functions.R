#Project: Predicting breast cancer survival from expression data using deep learning
#Laboratorio 19 CIC (Centro de Investigación del Cáncer, Salamanca)
#January 2020 
#Fernando Bueno Gutierrez
#fernando.bueno.gutierrez@usal.es
#fernando.bueno.gutie@gmail.com


#This script contains functions used by My_Neural_Network.R to update the network weights and bias through the backpropagation algorithm



backpropagation<-function(){
  #updates weights and bias to minimize the sum of squares between predicted states and tags
  
  #n_layers: int: number of layers
  ##wba: a list containing weights, bias and states in each of the layers (i.e. wba$Ws is a list of length equal to the number of layers. Each element in this inner list is a matrix) 
  #nneuronsL: int, number of neurons in the last layer
  #mdata: df with samples as rows. First nneuronsL columns correspond to the tags (summy variables), remaining columns are the input data in each input neuron
  
  
  #layer by layer. Start in the last layer and end in the first one
  for(layer in n_layers:1){
    
    layer<<-layer#define in the global env
    ###################################
    #### define parameters for layer ##
    ###################################
    
    #state in layer previous to "layer"
    if(layer>1){
      A_previous_layer<<-wba$As[[layer-1]]
    } else {
      A_previous_layer<<-as.matrix(mdata[,seq(nneuronsL+1,ncol(mdata))])#for layer 1, the states of the previous layer are the initial states (input data)
    }
    
    #Z: mat (nsamples x n_neurons_in_layer), is the sum of the products weights by states_previous, plus de bias. Each column corresponds to a neuon in layer. 
    Z<<-wba$Zs[[layer]]
    #For each sample (one row), there will be as many Zs as neurons in layer
    #In global environment, since it will be used in update_weights/get_weight_gradiant/get_derivatives
    
    
    #predicted: mat (nsamples x n_neurons_in_layer), states in each neuron in "layer" as computed from left to right in the network
    predicted<<-wba$As[[layer]]
    #In global environment, since it will be used in update_weights/get_weight_gradiant/get_derivatives and update_bias/get_bias_gradiant/get_bias_derivatives
    
    
    #expected mat (nsamples x n_neurons_in_layer), states in each neuron in "layer" as expected from (tags, data, for last layer) or from backpropagation
    if(layer==n_layers){
      expected<<-mdata[,seq(1,nneuronsL)]#in the last layer, the expected states are the tags
    } else {
      expected<<-predicted-A_gradiant# or predicted+A_gradiant. in the remaining layers, the expected states are those computed after adding the state gradiants computed in the layer to its right (previous step in this for loop)
    }
    #In global environment, since it will be used in update_weights/get_weight_gradiant/get_derivatives and update_bias/get_bias_gradiant/get_bias_derivatives
    
    
    
    ###################################
    #### backpropagation calculus  ####
    ###################################
    
    #update the weights and return the weighst prior to update. This weight will be used in get_states_gradiant()
    W_prior2update<<-update_weights(layer)
    #In global environment, since it will be used in get_states_gradiant
    
    #update the bias
    update_bias(layer)
    
    #get the state gradianst that will be used to compute the cost in the layer to the left of "layer"
    if(layer>1){#The gradiants of the states to the left of layer 1 are not of interest
      A_gradiant<<-get_states_gradiant(layer)
    }
  }
  print("weights and bias have been updated")
  
}



update_weights<-function(layer){
  #Get a matrix with the gradiants of each weigh connecting "layer" with its previous layer. Each column corresponds to a neuron in "layer" and each row to a neuron in previous layer
  
  #layer: int between 1 and the number of layers
  #n_neurons: vect with number_of_layers+1 elements. Each element is a number of neurons. The first element is the number of neurons in the input layer, the second elment is n_neurons in first layer and the last element is the n_neurons inthe last layer
  #NOTE THAT n_neurons[layer+1] REFERS TO "LAYER" GIVEN THAT n_neurons STARTS AT 0 (INPUT LAYER)
  
  #weight_prior2update: mat, weights connecting layer and the layer to its left, prior to update in this iteration. Each column corresponds to a neruon in "layer", wheras each row correspond to a neuron in the previous layer
  

  gs_layer_1=mcmapply(get_weight_gradiant,1:n_neurons[layer+1],mc.cores=nsets)
  #Make avarege eithe every 100 rows or evey 5 rows
  gs_layer_2=split(gs_layer_1, ceiling(seq_along(gs_layer_1)/binsize))#The firts 100 elements correspond to neuron1_L. By splitting in bits of 100 elements, each bit corresponds to a neuron in L
  gs_layer_3=unlist(lapply(gs_layer_2,mean))#Take the 
  gs_layer_4=matrix(unlist(gs_layer_3),ncol =n_neurons[layer+1])
  
  weight_prior2update=wba$Ws[[layer]]#The weights of layer (prior to update) are required for the backpropagation of the states; since they will be updated in the wbs object in the nex line, the weights prior to update are returned in this function
  
  #replace the weights in the wba oject
  wba$Ws[[layer]]<<-wba$Ws[[layer]]-gs_layer_4

  
  return(weight_prior2update)
  
}




get_weight_gradiant<-function(neuron_number){
  #computes weight-gradiant for one individual weight. Note that there is one weight for each combination bt: (1)sample (2)neuron in "layer", (3) neuron in the previous layer 
  
  #sample_number: int, index of the sample in the mdata rows
  #neuron_number: int, index of the neuron in "layer" for whcih we want to compute weight_gradiants
  #neuron_number_previous_layer: int, index of the neuron in the previous layer for whcih we want to compute weight_gradiants
  
  #wg: int, weight gradiant: derivative of the weight in "layer" with respect to the cost between "layer" and the posterior layer. For the last layer, the posterior layer is the tag
  
  #get derivatives of weight with respect to cost, which is the product of 3 partial derivatives 
  
  derivatives<-get_derivatives(neuron_number)

  #d1=A_previous_layer[,1:n_neurons[layer] 
  wg=A_previous_layer[,1:n_neurons[layer]]*derivatives$d2*derivatives$d3
  
  return(wg)
}



get_derivatives<-function(neuron_number){
  #get the three partial derivatives that are required to compute the weight gradiant, that is the derivative of weight with respect to cost 
  #you get a pack of 3 partial derivatives for each combination bt: (1)sample (2)neuron in "layer", (3) neuron in the previous layer 
  
  #sample_number: int, index of the sample in the mdata rows
  #neuron_number: int, index of the neuron in "layer" for whcih we want to compute weight_gradiants
  #neuron_number_previous_layer: int, index of the neuron in the previous layer for whcih we want to compute weight_gradiants
  
  # retuns a list with each of the following elements (each beeing a partial derivative require for computing the weight gradiant)
  #d1: derivative of z with respect to weight
  #d2: derivative of state in "layer" with respect to z
  #d3: derivative cost with respect to state in "layer"
  
  #4'09 in https://www.youtube.com/watch?v=tIeHLnjs5U8&t=324s
  
  #1) deriv(zL,wL)
  #d1=A_previous_layer[,1:n_neurons[layer-1]]
  
  #2) deriv(aL,zL)
  d2=dsigmoid(Z[,neuron_number])
  
  #3) deriv(C,aL)
  d3<-2*(predicted[,neuron_number]-expected[,neuron_number])
  
  return(list("d2"=d2,"d3"=d3))
}


update_bias<-function(layer){
  #updates the bias of "layer" in the wba object
  
  #layer: int between 1 and the number of layers
  
  #create a grid with the combinations for which get_bias_gradiant will be computed. There should be as many bias gradiants as bias there are
 
  gs_layer_1=mcmapply(get_bias_gradiant,1:n_neurons[layer+1],mc.cores=nsets)
  gs_layer_2=colMeans(gs_layer_1)
  
  #replace the bias in the wba object
  wba$Bs[[layer]]<<- wba$Bs[[layer]]-gs_layer_2
  
  
}



get_bias_gradiant<-function(neuron_number){
  #computes weight-gradiant for one individual bias Note that there is one bias for each combination bt: (1)sample (2)neuron in "layer"
  
  #sample_number: int, index of the sample in the mdata rows
  #neuron_number: int, index of the neuron in "layer" for whcih we want to compute weight_gradiants
  
  #bg: int, bias gradiant: derivative of the bias in "layer" with respect to the cost between "layer" and the posterior layer. For the last layer, the posterior layer is the tag
  
  #get derivatives of bias with respect to cost, which is the product of 3 partial derivatives 
  
  derivatives<-get_bias_derivatives(neuron_number)
  #d1 for the bias is 1
  bg=derivatives$d2*derivatives$d3
  
  return(bg)
}



get_bias_derivatives<-function(neuron_number){
  # retuns a list with each of the following elements (each beeing a partial derivative for computing the bias gradiant)
  
  #d1: derivative of z with respect to bias
  #d2: derivative of state in "layer" with respect to z
  #d3: derivative cost with respect to state in "layer"
  
  #6'03 in https://www.youtube.com/watch?v=tIeHLnjs5U8&t=324s
  
  #1) deriv(zL,bL)
  #This derivative is just 1
  
  #2) deriv(aL,zL)
  d2=dsigmoid(Z[,neuron_number])
  
  #3) deriv(C,aL)
  d3<-2*(predicted[,neuron_number]-expected[,neuron_number])
  
  return(list("d2"=d2,"d3"=d3))
}



get_states_gradiant<-function(layer){
  #computes state-gradiant for one individual state. Note that there is one state-gradiant for each combination bt: (1)sample (2)neuron in "layer", (3) neuron in the previous layer 
  
  #layer: int between 1 and the number of layers
  #W_prior2update: mat, weights prior to update in this iteration. Each column corresponds to a neruon in "layer", wheras each row correspond to a neuron in the previous layer
  
  #gs_layer_4: mat, binsize x n_neurons[layer]. In other words, binsize x number of neurons in the previous layer. Firts column are the state gradiants for each sample for the first neuon in the layer previous to "layer"
  

  gs_layer_1=mcmapply(get_state_gradiant,1:n_neurons[layer],mc.cores=nsets)
  gs_layer_2=matrix(unlist(gs_layer_1),ncol =n_neurons[layer])

  #return states gradiant
  return(gs_layer_2)
  
  
}



get_state_gradiant<-function(neuron_number_previous_layer){
  #compute state gradiant for a combination bt: (1)sample (2)neuron in "layer", (3) neuron in the previous layer 
  
  #W_prior2update: mat, weights prior to update in this iteration. Each column corresponds to a neruon in "layer", wheras each row correspond to a neuron in the previous layer
  #sample_number: int, index of the sample in the mdata rows
  #neuron_number: int, index of the neuron in "layer" for whcih we want to compute weight_gradiants
  #neuron_number_previous_layer: int, index of the neuron in the previous layer for whcih we want to compute weight_gradiants
  
  #sg: int, state gradiant for acombination bt: (1)sample (2)neuron in "layer", (3) neuron in the previous layer 
  
  #get the weight for each combination bt: (3) neuron in the previous layer and (1)neuron in "layer" 
  WEIGHT=W_prior2update[neuron_number_previous_layer,1:n_neurons[layer+1]]

  #get the derivatives for each combination bt: (2)sample number (1)neuron in "layer" 
  derivatives<-get_bias_derivatives(1:n_neurons[layer+1])
  
  #compute state gradiant for each combination bt: (1)sample (2)neuron in "layer", (3) neuron in the previous layer 
  derivatives_product<-function(N){
    
    return(WEIGHT[N]*derivatives$d2[,N]*derivatives$d3[,N])
    
  }
  
  derivatives_products=mapply(derivatives_product,1:n_neurons[layer+1])
  
  return(rowSums(derivatives_products))
}

  



