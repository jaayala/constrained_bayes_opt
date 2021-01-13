import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings

from .action_space import ActionSpace



class ConstrainedBayesOpt():
    def __init__(self, all_actions_dict, contexts, kernels, constraint_thres, constr_greater, noise=1e-6, points=[], 
                 rewards=[], constraint_vals=[], beta_val=2.5, safeset=[], optimizedKernels=0, init_random=3):

        if len(safeset) == 0:
            warnings.warn('No initial safe set provided.')
            exit()

        self._space = ActionSpace(all_actions_dict, contexts, beta_val, safeset, constraint_thres, constr_greater)
        self.init_random = init_random
        self.S_sizes = []
        
        assert(len(points) == len(rewards) and len(points) == len(constraint_vals) and len(constraint_thres) == len(constr_greater))

        
        kernel_list = []
        if optimizedKernels == 1:
            print('Loading kernel hyperparameters from disk...')
            print('GP Obj funct: {}'.format(kernels[0].get_params()))
            kernel_obj = kernels[0].clone_with_theta(theta=kernels[0].theta)
            for i in range(len(constraint_thres)): # for each constraint
                kernel_list.append(kernels[i+1].clone_with_theta(theta=kernels[i+1].theta))
                print('GP constraint {}: {}'.format(i, kernels[i+1].get_params()))
                
            optimizer = None
                
        elif len(points) > 0:
            
            gp_hyp_objective = GaussianProcessRegressor(
                kernel=kernels[0],
                alpha=noise,
                normalize_y=True,
                n_restarts_optimizer=5)
            
            print('Optimizing kernel hyperparameters for objective GP...')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp_hyp_objective.fit(points, rewards)
            print('Done!')
            print('GP Obj funct: {}'.format(gp_hyp_objective.kernel_.get_params()))
            
            opt_theta_objective = gp_hyp_objective.kernel_.theta
            kernel_obj = kernels[0].clone_with_theta(opt_theta_objective)
            
            
            for i in range(len(constraint_thres)):
            
                gp_hyp_constraint = GaussianProcessRegressor(
                    kernel=kernels[i+1],
                    alpha=noise,
                    normalize_y=True,
                    n_restarts_optimizer=5)
                
                print('Optimizing kernel hyperparameters for constraint GP...')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp_hyp_constraint.fit(points, constraint_vals[:,i])
                print('Done!')
                print('GP constraint {}: {}'.format(i, gp_hyp_constraint.kernel_.get_params()))
   
                opt_theta_constraint = gp_hyp_constraint.kernel_.theta
                kernel_list.append(kernels[i+1].clone_with_theta(opt_theta_constraint))
            optimizer = None
                    
        else:
            warnings.warn('Kernel hyperparameters will be computed during the optimization.')
            optimizer = 'fmin_l_bfgs_b'
            kernel_obj = kernels[0].clone_with_theta(theta=kernels[0].theta)
            for i in range(len(constraint_thres)): # for each constraint
                kernel_list.append(kernels[i+1].clone_with_theta(theta=kernels[i+1].theta))
                
            
        self.gp_list = []
        
        gp1 = GaussianProcessRegressor(
            kernel=kernel_obj,
            alpha=noise,
            normalize_y=True,
            optimizer=optimizer)
        self.gp_list.append(gp1)

        for ker in kernel_list:
            gp2 = GaussianProcessRegressor(
                kernel=ker,
                alpha=noise,
                normalize_y=True,
                optimizer=optimizer)
            self.gp_list.append(gp2)
        

    @property
    def space(self):
        return self._space

    @property
    def res(self):
        return self._space.res()

    def register(self, context, action, reward, constraint_val):
        """Expect observation with known reward"""
        self._space.register(context, action, reward, constraint_val)

    def array_to_context(self, context):
        return self._space.array_to_context(context)
    
    def action_to_array(self, action):
        return self._space.action_to_array(action)

    def context_to_array(self, context):
        return self._space.context_to_array(context)

    def get_kernel_hyper(self):
        hyper = []
        for gp in self.gp_list:
            hyper.append(gp.kernel.get_params()['length_scale'])
        return hyper

    def suggest(self, context):
        """Most promissing point to probe next"""
        assert len(context) == self._space.context_dim
        context = self._space.context_to_array(context)
        
        if len(self._space) < self.init_random:
            suggestion = self._space.random_sample_from_S()
        else:
            self._space.generate_safeset(context, self.gp_list)
            suggestion = self._space.suggest_ucb(context, self.gp_list)
            
        self.S_sizes.append(np.sum(self._space._S))
        sizeS = np.sum(self._space._S)

        return self._space.array_to_action(suggestion), sizeS

