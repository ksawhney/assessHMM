from matplotlib import pyplot as plt
import pyhsmm.models as models
import pyhsmm.basic.distributions as distributions
from pyhsmm.util.text import progprint_xrange
from pyhsmm.util.general import stateseq_hamming_error

from HMM_Simulator import *

# Define parameters for HMM_Simulator
map_size = 5
obs_size = 8
obstacles = [9,10,11,12,14,16]
seed = 123
sigma = 0.5
p_stay = 0.2

hmm_map = HMM_Simulator(map_size=map_size,
                        obs_size=obs_size,
                        p_stay=p_stay,
                        obstacles=obstacles,
                        sigma=sigma,
                        seed=seed)
hmm_map.perturb_transition(a = -0.1, b = 0.1)

# Define parameters before data generation
N = 200
num = 10
path_type = "markov_generate_conti"
seq = hmm_map.multi_generate(N=N, num=num, path_type=path_type)
X = seq[0]
Z = seq[1]

# Define hyperparameters for WeakLimitHDPHSMM
obs_dim = 1
Nmax = 25
trunc = 50
obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.25,
                'nu_0':obs_dim+2}
dur_hypparams = {'alpha_0':2*30,
                 'beta_0':2}

# Initialize observation and duration distributions
obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

# Initialize HDP-HSMM object
posteriormodel = models.WeakLimitHDPHSMM(
        alpha_a_0=1., alpha_b_0=1. / 4,
        gamma_a_0=1., gamma_b_0=1. / 4,
        init_state_concentration=1.,
        obs_distns=obs_distns,
        dur_distns=dur_distns)

# Add data to HDP-HSMM object
for idx in range(Z.shape[0]):
    posteriormodel.add_data(Z[idx,np.newaxis].T, trunc=trunc)

# Perform resampling
for idx in progprint_xrange(100):
    posteriormodel.resample_model()

# Plot one sequence
fig1 = plt.figure(1)
posteriormodel.plot_stateseq(1)

# Compute Hamming distance
hamming_error = np.zeros(num)
for idx in range(num):
    hamming_error[idx] = stateseq_hamming_error(posteriormodel.stateseqs[idx], X[idx]) / N

print(hamming_error)
print(np.median(hamming_error))

# Plot sorted means
means = []
for o in posteriormodel.obs_distns:
    means.append(o.mu[0])

sorted_means = sorted(means)
used_means = []
for idx in range(len(means)):
    if idx in posteriormodel.used_states:
        used_means.append(means[idx])

fig2 = plt.figure(2)
plt.plot(sorted_means, "o", color = "blue")
for m in used_means:
    plt.plot(sorted_means.index(m), m, "o", color = "red")

plt.plot(sorted(hmm_map.S_type), "o", color = "green")