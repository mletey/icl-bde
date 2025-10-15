import numpy as np
import random

class fulltasksampler:
    def __init__(self, d, l, n, rho, Ctask, functionality=1) -> None:
        self.d = d # DIMENSION
        self.l = l # CONTEXT LENGTH
        self.n = n # NUMBER OF CONTEXTS
        self.rho = rho # LABEL NOISE
        self.Ctask = Ctask # TASK COVARIANCE
        self.rng = np.random.default_rng(None) # SAMPLER OBJECT 
        self.functionality = functionality

    def __next__(self):
        xs = self.rng.normal(loc=0, scale = 1/np.sqrt(self.d), size=(self.n, self.l + 1, self.d))
        ws = self.rng.multivariate_normal(mean=np.zeros(self.d), cov=self.Ctask, size=self.n)
        ws = ws[..., np.newaxis] 
        if self.functionality==0:
            ys = np.tanh(xs @ ws) + self.rng.normal(loc=0, scale = np.sqrt(self.rho), size=(self.n, self.l+1, 1))
        else:
            #TODO FIGURE OUT WHY TF THIS DOESNT WORK FOR POWER!=1
            ys = (xs @ ws)**self.functionality + self.rng.normal(loc=0, scale = np.sqrt(self.rho), size=(self.n, self.l+1, 1))
        Z = np.zeros((self.n, self.l + 1, self.d + 1))
        Z[:,:,0:self.d] = xs
        Z[:,:,-1] = ys.squeeze()
        Z[:,-1, self.d] = 0 # padding for final context
	    
	    # returns the Z [x,y,x,y]... configuration and the true N+1 value for testing 
        return Z, ys[:,-1].squeeze()

    def __iter__(self):
        return self

# We introduce a new parameter here as well
# diversity: K = number of distinct beta_k to be sampled UNIFORMLY for each context 
class finitetasksampler:
    def __init__(self, d, l, n, k, rho, Ctask, functionality=1) -> None:
        self.d = d # DIMENSION
        self.l = l # CONTEXT LENGTH
        self.n = n # NUMBER OF CONTEXTS
        self.k = k # TASK DIVERSITY
        self.rho = rho # LABEL NOISE
        self.Ctask = Ctask # TASK COVARIANCE
        self.rng = np.random.default_rng(None) # SAMPLER OBJECT 
        # Now we fix a set of TASKS which will be sampled from during all other calls to iter or next once this object is instantiated. 
        # Once we get to the actual sampling, we will use
        self.E = np.random.multivariate_normal(mean=np.zeros(self.d), cov=self.Ctask, size=self.k).T  # Shape: (D, K)
        self.functionality = functionality
        
    def __next__(self):
        context_length = self.n + 1
        uniform_ps = np.array([random.randrange(self.k) for _ in range(self.n)])
        ws = np.array([self.E[:,uniform_ps[i]] for i in range(len(uniform_ps))]) 
        ws = ws[:,:,np.newaxis] # batch_size x n_dims x 1 as before
        xs = self.rng.normal(loc=0, scale = 1/np.sqrt(self.d), size=(self.n, context_length, self.d))
        ys = (xs @ ws)**self.functionality + self.rng.normal(loc=0, scale = np.sqrt(self.rho), size=(self.n, context_length, 1))
        Z = np.zeros((self.n, context_length, self.d + 1))
        Z[:,:,0:self.d] = xs
        Z[:,:,-1] = ys.squeeze()
        Z[:,-1, self.d] = 0 # padding for final context
        # returns the Z [x,y,x,y]... configuration and the true N+1 value for testing 
        return Z, ys[:,-1].squeeze()

    def __iter__(self):
        return self