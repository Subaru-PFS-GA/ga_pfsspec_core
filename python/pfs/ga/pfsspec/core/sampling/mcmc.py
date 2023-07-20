import numpy as np

class MCMC():
    """
    Implements a very simple adaptive MC sampled with optional Gibbs sampling.
    """

    def __init__(self, lp_fun, step_size=None, gibbs_blocks=None, walkers=3, gamma=0.99):
        self.lp_fun = lp_fun                    # Function to sample
        self.step_size = step_size              # Initial step size for each variable
        self.gibbs_blocks = gibbs_blocks        # Gibbs step parameter blocks
        self.walkers = walkers                  # Number of independent walkers
        self.gamma = gamma                      # Adaptive proposal memory

    def sample(self, x_0, burnin, samples, gamma=None):

        # Verify initial step size
        step_size = self.step_size

        if step_size is None or np.any(np.isnan(step_size)):
            raise Exception("Initial step size is invalid.")

        walkers = self.walkers
        gamma = gamma if gamma is not None else self.gamma

        # Broadcast initial state to the number of walkers. This changes the shape only
        # when x_0 is not already two dimensional. Make a copy so x can be updated!
        # original shape if x_0: (dim) or (dim, walkers)
        # broadcast to: (dim, walkers)
        x = np.broadcast_to(x_0, (np.shape(x_0)[0], walkers)).transpose().copy()

        if self.gibbs_blocks is not None:
            gibbs_blocks = self.gibbs_blocks
        else:
            gibbs_blocks = [[ i for i in range(x.shape[0])]]
        
        # Evaluate the probability at the initial state
        lp = np.zeros(walkers)
        for w in range(walkers):
            lp[w] = self.lp_fun(x[w])

        if np.any(np.isnan(lp) | np.isinf(lp)):
            raise Exception("Initial log L is invalid.")

        # Initialize proposal distributions
        org = []    # Zero vectors to pass into multivariate_normal
        loc = []    # Sample mean updated iteratively to calculate proposal
        cov = []    # Proposal covariance, updated iteratively
        for bi, b in enumerate(gibbs_blocks):
            cov.append(np.broadcast_to(np.diag(step_size[b]), (walkers,) + step_size[b].shape + step_size[b].shape).copy())
            loc.append(x[:, b])
            org.append(np.zeros_like(x[:, b]))

        for phase in ['burnin', 'sampling']:            
            if phase == 'burnin':
                n = burnin
            elif phase == 'sampling':
                n = samples
            else:
                raise NotImplementedError()

            res_x = np.empty((x.shape[-1], n, walkers))
            res_lp = np.empty((n, walkers))
            accepted = np.zeros((len(gibbs_blocks), walkers), dtype=int)
            
            for i in range(n):
                # Propose step for the next Gibbs parameter block
                bi = i % len(gibbs_blocks)
                b = gibbs_blocks[bi]

                prop = np.empty_like(loc[bi])
                for w in range(walkers):
                    prop[w] = np.random.multivariate_normal(np.zeros_like(loc[bi][w]), cov[bi][w])

                # Calculate new state and probability
                nx = x.copy()
                nx[:, b] = nx[:, b] + prop
                
                nlp = np.empty_like(lp)
                for w in range(walkers):
                    nlp[w] = self.lp_fun(nx[w])

                # Update state if accepted
                accept = (np.log(np.random.rand(*np.shape(lp))) < nlp - lp)
                x[accept] = nx[accept]
                lp[accept] = nlp[accept]
                accepted[bi, accept] = accepted[bi, accept] + 1

                # Store the state
                # Unpack_params expects that the leading dimension is number of params
                # shape: (dim, samples, walkers)
                res_x[:, i, :] = x.T
                res_lp[i] = lp

                # Update proposals but only in the sampling phase
                if True: # phase == 'sampling':
                    # Outer product of the new sample with itself
                    nn = x[:, b] - loc[bi]
                    nd = np.einsum('ij,ik->ijk', nn, nn)

                    loc[bi] = gamma * loc[bi] + (1 - gamma) * x[:, b]
                    cov[bi] = gamma * cov[bi] + (1 - gamma) * (2.38**2 / len(b)) * nd

                    pass
                           
        # Acceptance rate for each walker, summed up over Gibbs steps
        accept_rate = np.sum(np.stack([ accepted[bi] for bi in range(len(gibbs_blocks)) ]), axis=0) / samples

        return res_x, res_lp, accept_rate