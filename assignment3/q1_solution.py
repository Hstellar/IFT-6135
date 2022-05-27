import math
import numpy as np
import torch


def log_likelihood_bernoulli(mu, target):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)

    # log_likelihood_bernoulli
    bll1 = (target * torch.log(mu) + (1-target) * torch.log(1-mu)).sum(axis=1)
    return bll1


def log_likelihood_normal(mu, logvar, z):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    # batch_size = mu.size(0)
    # mu = mu.view(batch_size, -1)
    # logvar = logvar.view(batch_size, -1)
    # z = z.view(batch_size, -1)
    # input_size = mu.size(1)

    # res = []
    # for i in range(batch_size):
    #     to_learn = torch.distributions.MultivariateNormal(loc=mu[i,:], covariance_matrix=torch.diag(np.exp(logvar[i,:])))
    #     res.append((to_learn.log_prob(z[i,:])).mean())

    # res = torch.stack(res)
    # res = res.view(batch_size,-1)
    # # print(res)
    # return res
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)

    # log normal
    variance = torch.exp(logvar)
    log_normal = torch.sum(-0.5 * torch.log(2 * math.pi * variance) - torch.pow(z - mu, 2) / (2 * variance), dim=1)
    return log_normal



def log_mean_exp(y):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)
    res = []
    print(y.size())
    # log_mean_exp
    for i in range(batch_size):
        a= torch.max(y[i,:])
        res.append(torch.log(torch.mean(torch.exp(y[i,:] - a))) + a)

    res = torch.stack(res)
    res = res.view(batch_size,-1)
    # print(res)
    return res
    # batch_size = y.size(0)
    # sample_size = y.size(1)

    # # log_mean_exp
    # max_sample = torch.max(y, dim=1)[0]
    # reshaped_max_sample = max_sample.view(batch_size, -1)
    # log_mean_exp = torch.log(torch.mean(torch.exp(y - reshaped_max_sample), dim=1)) + max_sample
    # return log_mean_exp


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    # kld

    tr_like = (torch.exp(logvar_q) + ((mu_q - mu_p) ** 2))/(2*torch.exp(logvar_p))
    kl = (1/2)*logvar_p - (1/2)*logvar_q + tr_like - (1/2)
    res = kl.sum(1)
    # for i in range(batch_size):
        # p = torch.distributions.MultivariateNormal(loc=mu_p[i,:], covariance_matrix=torch.diag(torch.exp(logvar_p[i,:])))
        # q = torch.distributions.MultivariateNormal(loc=mu_q[i,:], covariance_matrix=torch.diag(torch.exp(logvar_q[i,:])))
        # kl = torch.distributions.kl_divergence(p, q).mean()
    return res


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    # batch_size = mu_q.size(0)
    # input_size = np.prod(mu_q.size()[1:])
    # mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    # logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    # mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    # logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    # kld
    var_p = torch.exp(logvar_p)
    var_q = torch.exp(logvar_q)
    q = torch.distributions.normal.Normal(mu_q, var_q ** 0.5)

    z = q.rsample((num_samples,))
    prob_p = (-0.5*(torch.log((2*np.pi)*var_p))) - (0.5 * ((z-mu_p)**2/var_p))
    prob_q = (-0.5*(torch.log((2*np.pi)*var_q))) - (0.5 * ((z-mu_q)**2/var_q))
    res = torch.mean(torch.sum(prob_q-prob_p, dim=1))
    return res

    