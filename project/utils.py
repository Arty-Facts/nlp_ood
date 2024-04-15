import torch
import numpy as np
from scipy import integrate
from sklearn import metrics
import pickle
import time
import multiprocessing as mp
import random
from torchdiffeq import odeint as odeint_torch


class dummy:
    @staticmethod
    def is_alive():
        return False
    @staticmethod
    def close():
        pass
    @staticmethod
    def terminate():
        pass
    @staticmethod
    def join():
        pass
    @staticmethod
    def kill():
        pass

class Worker:
    def __init__(self, id, node):
        self.id = id
        self.node = node
        self.process = dummy
        self.start_time = time.perf_counter()


def parallelize(func, jobs, gpu_nodes, verbose=True, timeout=3600):
    if verbose:
        print(f'Launching {len(jobs)} jobs on {len(set(gpu_nodes))} GPUs. {len(gpu_nodes)//len(set(gpu_nodes))} jobs per GPU in parallel..')
    workers = [Worker(id, node) for id, node in enumerate(gpu_nodes)]
    while len(jobs) > 0:
        random.shuffle(workers)
        for worker in workers:
            if time.perf_counter() - worker.start_time > timeout:
                if verbose:
                    print(f'Job on cuda:{worker.node} in slot {worker.id} timed out. Killing it...')
                worker.process.terminate()
            if not worker.process.is_alive():
                worker.process.kill()
                if verbose:
                    print(f'Launching job on cuda:{worker.node} in slot {worker.id}. {len(jobs)} jobs to left...')
                if len(jobs) == 0:
                    break
                args = list(jobs.pop())
                args.append(f'cuda:{worker.node}')
                p = mp.Process(target=func, args=args)
                p.start()
                worker.process = p
                worker.start_time = time.perf_counter()
                time.sleep(1)
        time.sleep(1)
    for worker in workers:
        worker.process.join()
    if verbose:
        print('Done!')

def load_data_blob(feature_dir, encoder, filename):
    """Load the data blob from the given path.

    Args:
        feature_dir: The directory containing the data.
        encoder: The encoder used to extract the features.
        filename: The name of the file containing the data.

    Returns:
      The data.
    """
    with open(f'{feature_dir}/{encoder}/{filename}', 'rb') as f:
        blob = pickle.load(f)
    return blob


def marginal_prob_std(time_step, sigma, device="cuda"):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:
    time_step: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The standard deviation.
  """
  if not isinstance(time_step, (torch.Tensor, torch.cuda.FloatTensor)):
    time_step = torch.tensor(time_step, device=device)
  return torch.sqrt((sigma**(2 * time_step) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(time_step, sigma, device="cuda"):
  """Compute the diffusion coefficient of our SDE.

  Args:
    time_step: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The vector of diffusion coefficients.
  """
  if not isinstance(time_step, (torch.Tensor, torch.cuda.FloatTensor)):
    time_step = torch.tensor(time_step, device=device)
  return sigma**time_step

def prior_likelihood(z, sigma):
  """The likelihood of a Gaussian distribution with mean zero and
      standard deviation sigma."""
  shape = z.shape
  N = np.prod(shape[1:])
  return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=1) / (2 * sigma**2)

def ode_likelihood(x,
                   score_model,
                   marginal_prob_std,
                   diffusion_coeff,
                   batch_size=64, #TODO: we are not using this
                   device='cuda',
                   eps=1e-5):
    """Compute the likelihood with probability flow ODE.

    Args:
        x: Input data.
        score_model: A PyTorch model representing the score-based model.
        marginal_prob_std: A function that gives the standard deviation of the
        perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the
        forward SDE.
        batch_size: The batch size. Equals to the leading dimension of `x`.
        device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
        eps: A `float` number. The smallest time step for numerical stability.

    Returns:
        z: The latent code for `x`.
        bpd: The log-likelihoods in bits/dim.
    """

    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    epsilon = torch.randn_like(x)

    def divergence_eval(sample, time_steps, epsilon):
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        with torch.enable_grad():
            sample.requires_grad_(True)
            score_e = torch.sum(score_model(sample, time_steps) * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample)[0]
        return torch.sum(grad_score_e * epsilon, dim=1)

    shape = x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = sample.reshape(shape)
        time_steps = time_steps.reshape((sample.shape[0], ))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score
    
    def divergence_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad():
            # Obtain x(t) by solving the probability flow ODE.
            sample = sample.reshape(shape)
            time_steps = time_steps.reshape((sample.shape[0], ))
            # Compute likelihood.
            div = divergence_eval(sample, time_steps, epsilon)
        return div

    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        time_steps = torch.ones((shape[0],), device=device) * t
        sample = x[:-shape[0]]
        logp = x[-shape[0]:]
        g = diffusion_coeff(t)
        sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, time_steps)
        return torch.cat([sample_grad.reshape(-1), logp_grad.reshape(-1)], dim=0)

    init_state = torch.cat([x.reshape(-1), torch.zeros(x.size(0), device=device)], dim=0)  # Concatenate x (flattened) and logp
    timesteps = torch.tensor([eps, 1.0], device=device)
    # Solve the ODE
    # 'dopri8' 7s
    # 'dopri5' 1.9s - good same as scipy.solve_ivp rk45
    # 'bosh3' 2.5s
    # 'fehlberg2' 1.4s - is scipy.solve_ivp rkf45
    # 'adaptive_heun' 4s
    # 'euler' nan
    # 'midpoint' nan
    # 'rk4' 1s inaccurate 
    # 'explicit_adams' 1s inaccurate 
    # 'implicit_adams' 1s inaccurate
    # 'fixed_adams' 1s inaccurate
    # 'scipy_solver'
    res = odeint_torch(ode_func, init_state, timesteps, rtol=1e-5, atol=1e-5, method='fehlberg2')
    zp = res[-1]

    z = zp[:-shape[0]].reshape(shape)
    delta_logp = zp[-shape[0]:].reshape(shape[0])
    sigma_max = marginal_prob_std(1.)
    prior_logp = prior_likelihood(z, sigma_max) #TODO: do we need this?
    bpd = -(prior_logp + delta_logp) / np.log(2)
    N = np.prod(shape[1:])
    bpd = bpd / N + 8.
    return bpd


def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out

def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh

def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh