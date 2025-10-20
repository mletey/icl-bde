import numpy as np
import random
from typing import Callable, Optional, List, Tuple

class FullTaskSampler_variablecontexts:
    def __init__(
        self,
        d: int,
        l: int,
        n: int,
        rho: float,
        Ctask: np.ndarray,
        *,
        functionality: float = 1.0,
        length_sampler: Optional[Callable[[int, int, np.random.Generator], np.ndarray]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.d = d
        self.l = l
        self.n = n
        self.rho = rho
        self.Ctask = Ctask
        self.functionality = functionality
        self.rng = np.random.default_rng(seed)
        self.length_sampler = length_sampler

    def __iter__(self):
        return self

    def _sample_lengths(self) -> np.ndarray:
        if self.length_sampler is None:
            # Default: constant lengths (keeps compatibility until you plug in your sampler)
            return np.full(self.n, int(max(1, self.l)), dtype=int)
        l_vec = self.length_sampler(self.n, self.l, self.rng).astype(int)
        if np.any(l_vec < 1):
            raise ValueError("All sampled context lengths must be >= 1.")
        return l_vec

    def __next__(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Returns:
          Z_list: list of length n; each Z_i has shape (l_i+1, d+1) with:
                  - Z_i[:, :d] = x tokens (context + query)
                  - Z_i[:,  d] = y tokens, BUT Z_i[l_i, d] = 0 (query label hidden)
          y_query: (n,) the true query labels (y at index l_i for each sample)
        """
        d, n = self.d, self.n
        l_vec = self._sample_lengths()

        # Draw one task vector per sample
        ws = self.rng.multivariate_normal(mean=np.zeros(d), cov=self.Ctask, size=n)  # (n, d)
        ws = ws[:, :, None]  # (n, d, 1)

        Z_list: List[np.ndarray] = []
        y_query = np.empty(n, dtype=float)

        for i in range(n):
            Li = int(l_vec[i])  # number of context tokens (query at index Li)
            # x_i shape: (Li+1, d)
            x_i = self.rng.normal(loc=0.0, scale=1.0/np.sqrt(d), size=(Li + 1, d))
            # noise_i shape: (Li+1, 1)
            noise_i = self.rng.normal(loc=0.0, scale=np.sqrt(self.rho), size=(Li + 1, 1))
            # y_i shape: (Li+1, 1)
            y_i = (x_i @ ws[i]) ** self.functionality + noise_i

            # Build Z_i: (Li+1, d+1) with last column = y, but hide query label
            Z_i = np.zeros((Li + 1, d + 1), dtype=float)
            Z_i[:, :d] = x_i
            Z_i[:, d] = y_i.squeeze(-1)

            # Hide the query label inside Z (match your original convention)
            y_query[i] = Z_i[Li, d]    # save true query y
            Z_i[Li, d] = 0.0           # set hidden/padding for query label

            Z_list.append(Z_i)

        # Return ragged batch (list) + vector of true query labels
        return Z_list, y_query

class fulltasksampler_fixedcontext:
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

class FullTaskSampler_variablecontexts:
    def __init__(
        self,
        d: int,
        l: int,
        n: int,
        rho: float,
        Ctask: np.ndarray,
        *,
        functionality: float = 1.0,
        length_sampler: Optional[Callable[[int, int, np.random.Generator], np.ndarray]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.d = d
        self.l = l
        self.n = n
        self.rho = rho
        self.Ctask = Ctask
        self.functionality = functionality
        self.rng = np.random.default_rng(seed)
        self.length_sampler = length_sampler
        self.E = self.rng.multivariate_normal(mean=np.zeros(self.d), cov=self.Ctask, size=self.k).T  # Shape: (D, K)

    def __iter__(self):
        return self

    def _sample_lengths(self) -> np.ndarray:
        if self.length_sampler is None:
            # Default: constant lengths (keeps compatibility until you plug in your sampler)
            return np.full(self.n, int(max(1, self.l)), dtype=int)
        l_vec = self.length_sampler(self.n, self.l, self.rng).astype(int)
        if np.any(l_vec < 1):
            raise ValueError("All sampled context lengths must be >= 1.")
        return l_vec

    def __next__(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Returns:
          Z_list: list of length n; each Z_i has shape (l_i+1, d+1) with:
                  - Z_i[:, :d] = x tokens (context + query)
                  - Z_i[:,  d] = y tokens, BUT Z_i[l_i, d] = 0 (query label hidden)
          y_query: (n,) the true query labels (y at index l_i for each sample)
        """
        d, n = self.d, self.n
        l_vec = self._sample_lengths()

        # Draw one task vector per sample
        uniform_ps = np.array([random.randrange(self.k) for _ in range(self.n)])
        ws = np.array([self.E[:,uniform_ps[i]] for i in range(len(uniform_ps))]) 
        ws = ws[:,:,np.newaxis] # batch_size x n_dims x 1 as before

        Z_list: List[np.ndarray] = []
        y_query = np.empty(n, dtype=float)

        for i in range(n):
            Li = int(l_vec[i])  # number of context tokens (query at index Li)
            # x_i shape: (Li+1, d)
            x_i = self.rng.normal(loc=0.0, scale=1.0/np.sqrt(d), size=(Li + 1, d))
            # noise_i shape: (Li+1, 1)
            noise_i = self.rng.normal(loc=0.0, scale=np.sqrt(self.rho), size=(Li + 1, 1))
            # y_i shape: (Li+1, 1)
            y_i = (x_i @ ws[i]) ** self.functionality + noise_i

            # Build Z_i: (Li+1, d+1) with last column = y, but hide query label
            Z_i = np.zeros((Li + 1, d + 1), dtype=float)
            Z_i[:, :d] = x_i
            Z_i[:, d] = y_i.squeeze(-1)

            # Hide the query label inside Z (match your original convention)
            y_query[i] = Z_i[Li, d]    # save true query y
            Z_i[Li, d] = 0.0           # set hidden/padding for query label

            Z_list.append(Z_i)

        # Return ragged batch (list) + vector of true query labels
        return Z_list, y_query

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
