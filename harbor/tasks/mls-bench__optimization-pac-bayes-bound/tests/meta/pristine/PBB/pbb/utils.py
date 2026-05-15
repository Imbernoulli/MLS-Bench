import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from pbb.models import NNet4l, CNNet4l, ProbNNet4l, ProbCNNet4l, ProbCNNet9l, CNNet9l, CNNet13l, ProbCNNet13l, ProbCNNet15l, CNNet15l, trainNNet, testNNet, Lambda_var, trainPNNet, computeRiskCertificates, testPosteriorMean, testStochastic, testEnsemble
from pbb.bounds import PBBobj
from pbb import data

# TODOS: 1. make a train prior function (bbb, erm)
#        2. make train posterior function 
#        3. rename partitions of data (prior_data, posterior_data, eval_data)
#        4. implement early stopping with validation set & speed
#        5. add data augmentation (maria)
#        6. better way of logging

def runexp(name_data, objective, prior_type, model, sigma_prior, pmin, learning_rate, momentum, 
learning_rate_prior=0.01, momentum_prior=0.95, delta=0.025, layers=9, delta_test=0.01, mc_samples=1000, 
samples_ensemble=100, kl_penalty=1, initial_lamb=6.0, train_epochs=100, prior_dist='gaussian', 
verbose=False, device='cuda', prior_epochs=20, dropout_prob=0.2, perc_train=1.0, verbose_test=False, 
perc_prior=0.2, batch_size=250):
    """Run an experiment with PAC-Bayes inspired training objectives

    Parameters
    ----------
    name_data : string
        name of the dataset to use (check data file for more info)

    objective : string
        training objective to use

    prior_type : string
        could be rand or learnt depending on whether the prior 
        is data-free or data-dependent
    
    model : string
        could be cnn or fcn
    
    sigma_prior : float
        scale hyperparameter for the prior
    
    pmin : float
        minimum probability to clamp the output of the cross entropy loss
    
    learning_rate : float
        learning rate hyperparameter used for the optimiser

    momentum : float
        momentum hyperparameter used for the optimiser

    learning_rate_prior : float
        learning rate used in the optimiser for learning the prior (only
        applicable if prior is learnt)

    momentum_prior : float
        momentum used in the optimiser for learning the prior (only
        applicable if prior is learnt)
    
    delta : float
        confidence parameter for the risk certificate
    
    layers : int
        integer indicating the number of layers (applicable for CIFAR-10, 
        to choose between 9, 13 and 15)
    
    delta_test : float
        confidence parameter for chernoff bound

    mc_samples : int
        number of monte carlo samples for estimating the risk certificate
        (set to 1000 by default as it is more computationally efficient, 
        although larger values lead to tighter risk certificates)

    samples_ensemble : int
        number of members for the ensemble predictor

    kl_penalty : float
        penalty for the kl coefficient in the training objective

    initial_lamb : float
        initial value for the lambda variable used in flamb objective
        (scaled later)
    
    train_epochs : int
        numer of training epochs for training

    prior_dist : string
        type of prior and posterior distribution (can be gaussian or laplace)

    verbose : bool
        whether to print metrics during training

    device : string
        device the code will run in (e.g. 'cuda')

    prior_epochs : int
        number of epochs used for learning the prior (not applicable if prior is rand)

    dropout_prob : float
        probability of an element to be zeroed.

    perc_train : float
        percentage of train data to use for the entire experiment (can be used to run
        experiments with reduced datasets to test small data scenarios)
    
    verbose_test : bool
        whether to print test and risk certificate stats during training epochs

    perc_prior : float
        percentage of data to be used to learn the prior

    batch_size : int
        batch size for experiments
    """

    # this makes the initialised prior the same for all bounds
    torch.manual_seed(7)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_kargs = {'num_workers': 1,
                    'pin_memory': True} if torch.cuda.is_available() else {}

    train, test = data.loaddataset(name_data)
    rho_prior = math.log(math.exp(sigma_prior)-1.0)

    if prior_type == 'rand':
        dropout_prob = 0.0

    # initialise model
    if model == 'cnn':
        if name_data == 'cifar10':
            # only cnn models are tested for cifar10, fcns are only used 
            # with mnist
            if layers == 9:
                net0 = CNNet9l(dropout_prob=dropout_prob).to(device)
            elif layers == 13:
                net0 = CNNet13l(dropout_prob=dropout_prob).to(device)
            elif layers == 15:
                net0 = CNNet15l(dropout_prob=dropout_prob).to(device)
            else: 
                raise RuntimeError(f'Wrong number of layers {layers}')
        else:
            net0 = CNNet4l(dropout_prob=dropout_prob).to(device)
    else:
        net0 = NNet4l(dropout_prob=dropout_prob, device=device).to(device)

    if prior_type == 'rand':
        train_loader, test_loader, _, val_bound_one_batch, _, val_bound = data.loadbatches(
            train, test, loader_kargs, batch_size, prior=False, perc_train=perc_train, perc_prior=perc_prior)
        errornet0 = testNNet(net0, test_loader, device=device)
    elif prior_type == 'learnt':
        train_loader, test_loader, valid_loader, val_bound_one_batch, _, val_bound = data.loadbatches(
            train, test, loader_kargs, batch_size, prior=True, perc_train=perc_train, perc_prior=perc_prior)
        optimizer = optim.SGD(
            net0.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
        for epoch in trange(prior_epochs):
            trainNNet(net0, optimizer, epoch, valid_loader,
                      device=device, verbose=verbose)
        errornet0 = testNNet(net0, test_loader, device=device)

    posterior_n_size = len(train_loader.dataset)
    bound_n_size = len(val_bound.dataset)

    toolarge = False
    train_size = len(train_loader.dataset)
    classes = len(train_loader.dataset.classes)

    if model == 'cnn':
        toolarge = True
        if name_data == 'cifar10':
            if layers == 9:
                net = ProbCNNet9l(rho_prior, prior_dist=prior_dist,
                                    device=device, init_net=net0).to(device)
            elif layers == 13:
                net = ProbCNNet13l(rho_prior, prior_dist=prior_dist,
                                   device=device, init_net=net0).to(device)
            elif layers == 15: 
                net = ProbCNNet15l(rho_prior, prior_dist=prior_dist,
                                   device=device, init_net=net0).to(device)
            else: 
                raise RuntimeError(f'Wrong number of layers {layers}')
        else:
            net = ProbCNNet4l(rho_prior, prior_dist=prior_dist,
                          device=device, init_net=net0).to(device)
    elif model == 'fcn':
        if name_data == 'cifar10':
            raise RuntimeError(f'Cifar10 not supported with given architecture {model}')
        elif name_data == 'mnist':
            net = ProbNNet4l(rho_prior, prior_dist=prior_dist,
                        device=device, init_net=net0).to(device)
    else:
        raise RuntimeError(f'Architecture {model} not supported')
    # import ipdb
    # ipdb.set_trace()
    bound = PBBobj(objective, pmin, classes, delta,
                    delta_test, mc_samples, kl_penalty, device, n_posterior = posterior_n_size, n_bound=bound_n_size)

    if objective == 'flamb':
        lambda_var = Lambda_var(initial_lamb, train_size).to(device)
        optimizer_lambda = optim.SGD(lambda_var.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer_lambda = None
        lambda_var = None

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in trange(train_epochs):
        trainPNNet(net, optimizer, bound, epoch, train_loader, lambda_var, optimizer_lambda, verbose)
        if verbose_test and ((epoch+1) % 5 == 0):
            train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(net, toolarge,
            bound, device=device, lambda_var=lambda_var, train_loader=val_bound, whole_train=val_bound_one_batch)

            stch_loss, stch_err = testStochastic(net, test_loader, bound, device=device)
            post_loss, post_err = testPosteriorMean(net, test_loader, bound, device=device)
            ens_loss, ens_err = testEnsemble(net, test_loader, bound, device=device, samples=samples_ensemble)

            print(f"***Checkpoint results***")         
            print(f"Objective, Dataset, Sigma, pmin, LR, momentum, LR_prior, momentum_prior, kl_penalty, dropout, Obj_train, Risk_CE, Risk_01, KL, Train NLL loss, Train 01 error, Stch loss, Stch 01 error, Post mean loss, Post mean 01 error, Ens loss, Ens 01 error, 01 error prior net, perc_train, perc_prior")
            print(f"{objective}, {name_data}, {sigma_prior :.5f}, {pmin :.5f}, {learning_rate :.5f}, {momentum :.5f}, {learning_rate_prior :.5f}, {momentum_prior :.5f}, {kl_penalty : .5f}, {dropout_prob :.5f}, {train_obj :.5f}, {risk_ce :.5f}, {risk_01 :.5f}, {kl :.5f}, {loss_ce_train :.5f}, {loss_01_train :.5f}, {stch_loss :.5f}, {stch_err :.5f}, {post_loss :.5f}, {post_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}, {errornet0 :.5f}, {perc_train :.5f}, {perc_prior :.5f}")

    train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(net, toolarge, bound, device=device,
    lambda_var=lambda_var, train_loader=val_bound, whole_train=val_bound_one_batch)

    stch_loss, stch_err = testStochastic(net, test_loader, bound, device=device)
    post_loss, post_err = testPosteriorMean(net, test_loader, bound, device=device)
    ens_loss, ens_err = testEnsemble(net, test_loader, bound, device=device, samples=samples_ensemble)

    print(f"***Final results***") 
    print(f"Objective, Dataset, Sigma, pmin, LR, momentum, LR_prior, momentum_prior, kl_penalty, dropout, Obj_train, Risk_CE, Risk_01, KL, Train NLL loss, Train 01 error, Stch loss, Stch 01 error, Post mean loss, Post mean 01 error, Ens loss, Ens 01 error, 01 error prior net, perc_train, perc_prior")
    print(f"{objective}, {name_data}, {sigma_prior :.5f}, {pmin :.5f}, {learning_rate :.5f}, {momentum :.5f}, {learning_rate_prior :.5f}, {momentum_prior :.5f}, {kl_penalty : .5f}, {dropout_prob :.5f}, {train_obj :.5f}, {risk_ce :.5f}, {risk_01 :.5f}, {kl :.5f}, {loss_ce_train :.5f}, {loss_01_train :.5f}, {stch_loss :.5f}, {stch_err :.5f}, {post_loss :.5f}, {post_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}, {errornet0 :.5f}, {perc_train :.5f}, {perc_prior :.5f}")


def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
