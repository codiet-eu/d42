import math
import numpy as np

from scipy.special import gammaln

from codietpgm.dag_gflownet.scores.base import LocalScore
from codietpgm.dag_gflownet.scores.bge_score import BGeScore
from codietpgm.dag_gflownet.scores.bge_score import logdet


class DBNBGeScore(BGeScore):
        
    """BGe score for DBN 
    """

    def __init__(
            self,
            data,
            prior,
            num_vars, 
            num_time_slices,
            mean_obs=None,
            alpha_mu=1.,
            alpha_w=None
        ):

        super().__init__(data, prior, mean_obs, alpha_mu, alpha_w)
        print(self.num_variables)
        # Initialize DBNBGeScore specific parameters
        self.num_vars = num_vars
        self.num_time_slices = num_time_slices

    def local_score(self, target, indices):

        num_parents = len(indices)
        score = 0 

        #if target/variable is in prior network B0
        if target < self.num_vars :  
                                
            self.mean_obs = np.zeros((self.num_vars,))
            self.alpha_w = self.num_vars + 2.
            self.num_samples = self.data.shape[0] 
            self.t = (self.alpha_mu * (self.alpha_w - self.num_vars - 1)) / (self.alpha_mu + 1) 

            T = self.t * np.eye(self.num_vars) 
            
            data = np.asarray(self.data)
            data_prior = data[:,:self.num_vars]
            data_prior_mean = np.mean(data_prior, axis=0, keepdims=True)
            data_prior_centered = data_prior - data_prior_mean 
                
            self.R = (T + np.dot(data_prior_centered.T, data_prior_centered)
                + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu)) * np.dot((data_prior_mean - self.mean_obs).T, data_prior_mean - self.mean_obs)
                )

            all_parents = np.arange(self.num_vars) 
            self.log_gamma_term = (
                0.5 * (math.log(self.alpha_mu) - math.log(self.num_samples + self.alpha_mu)) + gammaln(0.5 * (self.num_samples + self.alpha_w - self.num_vars + all_parents + 1)) - gammaln(0.5 * (self.alpha_w - self.num_vars + all_parents + 1)) - 0.5 * self.num_samples * math.log(math.pi) + 0.5 * (self.alpha_w - self.num_vars + 2 * all_parents + 1) * math.log(self.t)
                )

            if indices:
                variables = [target] + list(indices)
                log_term_r = (
                    0.5 * (self.num_samples + self.alpha_w - self.num_vars + num_parents) * logdet(self.R[np.ix_(indices, indices)]) - 0.5 * (self.num_samples + self.alpha_w - self.num_vars + num_parents + 1) * logdet(self.R[np.ix_(variables, variables)])
                    )
            else:
                log_term_r = (-0.5 * (self.num_samples + self.alpha_w - self.num_vars + 1) * np.log(np.abs(self.R[target, target])))

            score = self.log_gamma_term[num_parents] + log_term_r

        # transition network  
        else: 
            num_vars_per_transition = self.num_vars*2

            self.mean_obs = np.zeros((num_vars_per_transition,))
            self.alpha_w = num_vars_per_transition + 2.
            self.num_samples = self.data.shape[0]*(self.num_time_slices-1) #number of transitions 

            self.t_term = (self.alpha_mu * (self.alpha_w - num_vars_per_transition - 1)) / (self.alpha_mu + 1)
            T = self.t_term * np.eye(num_vars_per_transition) 
            for t in range(self.num_time_slices-1):
                data_trans = np.asarray(self.data)
                data_transition = data_trans[:,self.num_vars*t:num_vars_per_transition+(self.num_vars*t)]
                data_transition_mean = np.mean(data_transition, axis=0, keepdims=True)
                data_transition_centered = data_transition - data_transition_mean 
                self.R = (T + np.dot(data_transition_centered.T, data_transition_centered) + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu)) * np.dot((data_transition_mean - self.mean_obs).T, data_transition_mean - self.mean_obs)
                        )
                 
                all_parents = np.arange(self.num_vars*t,(self.num_vars*t)+num_vars_per_transition) 
                self.log_gamma_term = (
                    0.5 * (math.log(self.alpha_mu) - math.log(self.num_samples + self.alpha_mu)) + gammaln(0.5 * (self.num_samples + self.alpha_w - num_vars_per_transition + all_parents + 1)) - gammaln(0.5 * (self.alpha_w - num_vars_per_transition + all_parents + 1)) - 0.5 * self.num_samples * math.log(math.pi) + 0.5 * (self.alpha_w - num_vars_per_transition + 2 * all_parents + 1) * math.log(self.t_term) 
                    )

                if indices:
                    variables = [target] + list(indices)
                    ltr = (0.5 * (self.num_samples + self.alpha_w - num_vars_per_transition + num_parents) * logdet(self.R[np.ix_(indices, indices)])
- 0.5 * (self.num_samples + self.alpha_w - num_vars_per_transition + num_parents + 1) * logdet(self.R[np.ix_(variables, variables)])
                       )

                else:
                    ltr = (-0.5 * (self.num_samples + self.alpha_w - num_vars_per_transition + 1) * np.log(np.abs(self.R[target, target])))
       
                score_t = self.log_gamma_term[num_parents] + ltr
                score += score_t

        return LocalScore(
            key=(target, tuple(indices)),
            score=score, 
            prior=self.prior(num_parents)
        )


