import numpy as np 

'''
We are going to use the California housing dataset provided by sklearn
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
to train a 2-layer fully connected neural net. We are going to build the neural network from scratch.
'''


class dlnet:

    def __init__(self, y, use_dropout, use_momentum, lr = 0.01, batch_size=64, momentum=0.5, dropout_prob=0.3):
        '''
        This method initializes the class, it is implemented for you. 
        Args:
            x: data
            y: labels
            Yh: predicted labels
            dims: dimensions of different layers
            alpha: slope coefficient for leaky relu
            param: dictionary of different layers parameters
            ch: Cache dictionary to store forward parameters that are used in backpropagation
            loss: list to store loss values
            lr: learning rate
            sam: number of training samples we have

            momentum: coefficient for momentum-based update step
            change: dict of previous changes for each layer
        '''
        self.Y=y # ground truth labels

        # OTHER HYPERPARAMTERS
        self.Yh=np.zeros((1,self.Y.shape[1])) # estimated labels
        self.dims = [8, 15, 1] # dimensions of different layers
        self.alpha = 0.05

        # DROPOUT
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob

        # PARAMETERS
        self.param = {} # dictionary for different layer variables
        self.ch = {} # cache for holding variables during forward propagation to use them in back prop
        self.loss = [] # list to store loss values
        self.batch_y = [] # list of y batched numpy arrays

        # TRAINING HYPERPARAMETERS
        self.iter = 0 # iterator to index into data for making a batch 
        self.batch_size = batch_size # batch size 
        
        # NEURAL NETWORK INFORMATION
        self.lr=lr # learning rate
        self.sam = self.Y.shape[1] # number of training samples we have
        self._estimator_type = 'regression'
        self.neural_net_type = "Leaky Relu -> Tanh"

        # MOMENTUM
        self.use_momentum = use_momentum
        self.momentum = momentum # momentum factor
        self.change = {} # dictionary for previous changes for momentum


    def nInit(self, param=None): 
        '''
        This method initializes the neural network variables, it is already implemented for you. 
        Check it and relate to the mathematical description above.
        You are going to use these variables in forward and backward propagation.
        '''
        if param is None:
            np.random.seed(1)
            self.param['theta1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
            self.param['b1'] = np.zeros((self.dims[1], 1))        
            self.param['theta2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
            self.param['b2'] = np.zeros((self.dims[2], 1))
        else:
            self.param = param

        for layer in self.param:
            self.change[layer] = np.zeros_like(self.param[layer])


    def Leaky_Relu(self,alpha, u):
        '''
        In this method you are going to implement element wise Leaky_Relu. 
        Make sure that all operations here are element wise and can be applied to an input of any dimension. 
        Input: 
            u of any dimension
            alpha: the slope coefficent of the negative part.
        return: Leaky_Relu(u) 

        HINT 1: When calculating the tanh and leaky relu function, make sure you are not modifying 
        the values in the original passed in matrix. You may find np.copy() helpful (`u` should not 
        be modified in the method.)

        '''
        # TODO: IMPLEMENT THIS METHOD
        
        raise NotImplementedError
        

    def Tanh(self, u):
        '''
        In this method you are going to implement element wise Tanh. 
        Make sure that all operations here are element wise and can be applied to an input of any dimension.
        Do NOT use np.tanh. 
        Input: u of any dimension
        return: Tanh(u) 

        HINT 1: When calculating the tanh and leaky relu function, make sure you are not modifying 
        the values in the original passed in matrix. You may find np.copy() helpful (`u` should not 
        be modified in the method.)

        '''
        # TODO: IMPLEMENT THIS METHOD
        
        raise NotImplementedError
    
    
    def dL_Relu(self,alpha, u):
        '''
        This method implements element wise differentiation of Leaky Relu, it is already implemented for you.  
        Input: 
             u of any dimension
             alpha: the slope coefficent of the negative part.
        return: dL_Relu(u) 
        '''
        return np.where(u > 0, 1.0, alpha)


    def dTanh(self, u):
        '''
        This method implements element wise differentiation of Tanh, it is already implemented for you.
        Input: u of any dimension
        return: dTanh(u) 
        '''
        return 1 - np.square(np.tanh(u))
    

    def nloss(self,y, yh):
        '''
        In this method you are going to implement mean squared loss. 
        Refer to the description above and implement the appropriate mathematical equation.
        Input: y 1xN: ground truth labels
               yh 1xN: neural network output

        return: MSE 1x1: loss value 
        '''
        
        # TODO: IMPLEMENT THIS METHOD

        raise NotImplementedError

    @staticmethod
    def _dropout(u, prob):
        '''
        This method implements the dropout layer. Refer to the description for implementation details.
        Input: u D x N: input to dropout layer
               prob: the probability of dropping an unit
        return: u_after_dropout D x N
                dropout_mask DxN
                
        Hint: scale the units after dropout
              use np.random.choice to sample from Bernoulli(prob) the inactivated nodes for each iteration  
        '''
        # TODO: IMPLEMENT THIS METHOD
        
        raise NotImplementedError



    def forward(self, x, use_dropout):
        '''
        Fill in the missing code lines, please refer to the description for more details.
        Check nInit method and use variables from there as well as other implemented methods.
        Refer to the description above and implement the appropriate mathematical equations.
        Do not change the lines followed by #keep. 

        Input: x DxN: input to neural network
               use_dropout: True if using dropout in forward
        return: o2 1xN
        '''  
        self.ch['X'] = x #keep
        
        # TODO: IMPLEMENT THIS METHOD
        u1 = None # IMPLEMENT THIS LINE
        o1 = None # IMPLEMENT THIS LINE

        if use_dropout: #keep
            o1, dropout_mask = None # IMPLEMENT THIS LINE
            self.ch['u1'], self.ch['mask'], self.ch['o1'] = u1, dropout_mask, o1 #keep
        else: #keep
            self.ch['u1'], self.ch['o1'] = u1, o1 #keep

        u2 = None # IMPLEMENT THIS LINE
        o2 = None # IMPLEMENT THIS LINE
        self.ch['u2'], self.ch['o2'] = u2, o2 #keep

        return o2 #keep

    def compute_gradients(self, y, yh, use_dropout):
        '''
        Compute the gradients for each layer given the predicted outputs and ground truths.
        The dropout mask you stored at forward may be helpful.

        Input:
            y: 1 x N numpy array, ground truth values
            yh: 1 x N numpy array, predicting outputs

        Output:
            dLoss: dictionary that maps layer names (strings) to gradients (numpy arrays)

        Note: You will have to use the cache (self.ch) to retrieve the values 
        from the forward pass!

        HINT 2: Division by N only needs to occur ONCE for any derivative that requires a division 
        by N. Make sure you avoid cascading divisions by N where you might accidentally divide your 
        derivative by N^2 or greater.
        '''
       # TODO: IMPLEMENT THIS METHOD

        dLoss_o2 = None # IMPLEMENT THIS LINE
        dLoss_u2 = None # IMPLEMENT THIS LINE
        dLoss_theta2 = None # IMPLEMENT THIS LINE
        dLoss_b2 = None # IMPLEMENT THIS LINE
        dLoss_o1 = None # IMPLEMENT THIS LINE
        
        if use_dropout:
            dLoss_u1 = None # IMPLEMENT THIS LINE
        else:
            dLoss_u1 = None # IMPLEMENT THIS LINE


        dLoss_theta1 = None # IMPLEMENT THIS LINE
        dLoss_b1 = None # IMPLEMENT THIS LINE
        
        dLoss = {'theta1': dLoss_theta1, 'b1': dLoss_b1, 'theta2': dLoss_theta2, 'b2': dLoss_b2}
        return dLoss


    def update_weights(self, dLoss, use_momentum):
        '''
        Update weights of neural network based on learning rate given gradients for each layer. 
        Can also use momentum to smoothen descent.
        
        Input:
            dLoss: dictionary that maps layer names (strings) to gradients (numpy arrays)

        Return:
            None

        HINT: both self.change and self.param need to be updated for use_momentum=True and only self.param needs to be updated when use_momentum=False
              momentum records are kept in self.change
        '''
        # TODO: IMPLEMENT THIS METHOD

        for layer in dLoss:
            if use_momentum:
                continue # IMPLEMENT THIS LINE
            else:
                continue # IMPLEMENT THIS LINE

    def backward(self, y, yh, use_dropout, use_momentum):
        '''
        Fill in the missing code lines, please refer to the description for more details
        You will need to use cache variables, some of the implemented methods, and other variables as well
        Refer to the description above and implement the appropriate mathematical equations.
        do not change the lines followed by #keep.  

        Input: y 1xN: ground truth labels
               yh 1xN: neural network output

        Return: dLoss_theta2 (1x15), dLoss_b2 (1x1), dLoss_theta1 (15xD), dLoss_b1 (15x1)

        Hint: make calls to compute_gradients and update_weights
        '''    
        # TODO: IMPLEMENT THIS METHOD
        
        raise NotImplementedError


    def gradient_descent(self, x, y, iter = 60000, use_momentum=False, local_test=False):
        '''
        This function is an implementation of the gradient descent algorithm.
        Notes:
        1. GD considers all examples in the dataset in one go and learns a gradient from them. 
        2. One iteration here is one round of forward and backward propagation on the complete dataset. 
        3. Append loss at multiples of 1000 i.e. at 0th, 1000th, 2000th .... iterations to self.loss
        **For LOCAL TEST append and print out loss at every iteration instead of every 1000th multiple.

        Input: x DxN: input
               y 1xN: labels
               iter: scalar, number of epochs to iterate through
        ''' 
        
        self.nInit() #keep
        
        # TODO: IMPLEMENT THIS METHOD
        
        raise NotImplementedError

        # for i in ....:


            # Print every one iteration for local test, and every 1000th iteration for AG and 1.2
            print_multiple = 1 if local_test else 1000 #keep
            if i % print_multiple == 0: #keep
                print ("Loss after iteration %i: %f" %(i, loss)) #keep 
                self.loss.append(loss) #keep
       
    
    #bonus for undergraduate students 
    def batch_gradient_descent(self, x, y, use_momentum, iter = 60000, local_test=False):
        '''
        This function is an implementation of the batch gradient descent algorithm

        Notes: 
        1. Batch GD loops over all mini batches in the dataset one by one and learns a gradient 
        2. One iteration here is one round of forward and backward propagation on one minibatch. 
           You will use self.iter and self.batch_size to index into x and y to get a batch. This batch will be
           fed into the forward and backward functions.

        3. Append and printout loss at multiples of 1000 iterations i.e. at 0th, 1000th, 2000th .... iterations. 
           **For LOCAL TEST append and print out loss at every iteration instead of every 1000th multiple.

        4. Append the y batched numpy array to self.batch_y at every 1000 iterations i.e. at 0th, 1000th, 
           2000th .... iterations. We will use this to determine if batching is done correctly.
           **For LOCAL TEST append the y batched array at every iteration instead of every 1000th multiple

        5. We expect a noisy plot since learning on a batch adds variance to the 
           gradients learnt
        6. Be sure that your batch size remains constant (see notebook for more detail). Please 
           batch your data in a wraparound manner. For example, given a dataset of 9 numbers, 
           [1, 2, 3, 4, 5, 6, 7, 8, 9], and a batch size of 6, the first iteration batch will 
           be [1, 2, 3, 4, 5, 6], the second iteration batch will be [7, 8, 9, 1, 2, 3], 
           the third iteration batch will be [4, 5, 6, 7, 8, 9], etc... 

        Input: x DxN: input
               y 1xN: labels
               iter: scalar, number of BATCHES to iterate through
               local_test: boolean, True if calling local test, default False for autograder and Q1.3 
                    this variable can be used to switch between autograder and local test requirement for
                    appending/printing out loss and y batch arrays

        '''
        
        self.nInit() #keep
        
        # TODO: IMPLEMENT THIS METHOD
        
        raise NotImplementedError

        # for i in ....:


            # Print every one iteration for local test, and every 1000th iteration for AG and 1.3
            print_multiple = 1 if local_test else 1000 #keep
            if i % print_multiple == 0: #keep
                print ("Loss after iteration %i: %f" %(i, loss)) #keep
                self.loss.append(loss) #keep
                self.batch_y.append(y_batch) #keep


    def predict(self, x): 
        '''
        This function predicts new data points
        It is implemented for you

        Input: x DxN: inputs
        Return: y 1xN: predictions

        '''
        Yh = self.forward(x, False)
        return Yh
