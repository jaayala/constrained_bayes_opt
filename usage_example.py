from constrained_bayes_opt import ConstrainedBayesOpt
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt


def dummy_environment(context, action):
    return np.sum(action)

def dummy_environment2(context, action):
    return np.sum(action) / 2


discvars = {'a1': np.linspace(0, 1, 101),
            'a2': np.linspace(0, 1, 101)}
action_dim = len(discvars)
contexts = {'c1': '', 'c2': ''}
context_dim = len(contexts)


length_scale = np.ones(context_dim+action_dim)
kernel = WhiteKernel(noise_level=1) + Matern(nu=1.5, length_scale=length_scale)


nConstraints = 1

length_scale_list = [np.ones(context_dim+action_dim) for _ in range(nConstraints+1)]
kernels = [Matern(nu=1.5, length_scale=length_scale_list[i]) for i in range(len(length_scale_list))]

constraints_thres = np.array([0.5]) # Maximum value that the constraint can take
constr_greater = np.array([0]) # the contrained function should take values *lower than* the threshold specified above
optimizedKernels = 0


safeset = np.array([[0, 0],
                   [0, 0.01],
                   [0.01, 0],
                   [0.01, 0.01]])
    
    
# In this example, we do not provide an initial dataset to perform an initial optimization of the kernel hyperparameters. This may lead the algorithm to stuck in local optima.
optimizer = ConstrainedBayesOpt(all_actions_dict=discvars, contexts=contexts, kernels=kernels, 
                     constraint_thres=constraints_thres, constr_greater=constr_greater, safeset=safeset, 
                     optimizedKernels=optimizedKernels)


vSize_S = []

nIters = 30
for i in range(nIters):

    print(i)
    rand_context = np.random.rand(context_dim)
    context = optimizer.array_to_context(rand_context)
    
    action, size_S = optimizer.suggest(context)
    
    vContext = optimizer.context_to_array(context)
    vAction = optimizer.action_to_array(action)
    reward = dummy_environment(vContext, vAction)
    constr_function = dummy_environment2(vContext, vAction)
    vSize_S.append(size_S)

    cost  = -reward # the algorithm performs minimization
    optimizer.register(context, action, cost, np.array([constr_function]))




res = optimizer.res
vReward = []
vConstraint = []
for i in range(nIters):
    vReward.append(-res[i]['reward'])
    vConstraint.append(res[i]['constraint_val'][0])
    
plt.figure()
plt.plot(vReward)
plt.xlabel('Iterations')
plt.ylabel('Reward')

plt.figure()
plt.plot(vConstraint)
plt.xlabel('Iterations')
plt.ylabel('Constrained function values')

plt.figure()
plt.plot(vSize_S)
plt.xlabel('Iterations')
plt.ylabel('Size of safe set')
