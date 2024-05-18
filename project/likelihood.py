import argparse
import functools
import pathlib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
import ood_detectors.likelihood as likelihood
import ood_detectors.eval_utils as eval_utils
import ood_detectors.plot_utils as plot_utils
import ood_detectors.sde as sde_lib 
import ood_detectors.models as models
import ood_detectors.losses as losses
from ood_detectors.residual import Residual
from data import load_datasets

def train(config):
	#data parameters
	ind = config["id"]
	ood = config["ood"]
	encoder = config["encoder"]
	device = config["device"]
	token_len = config["token_len"]

	#load dataset
	train_dataset, val_dataset, *ood_datasets = load_datasets(id = ind, ood=ood, embed_model=encoder, device=device, token_len=token_len)

	#network parameters
	bottleneck_channels=config["bottleneck_channels"]
	num_res_blocks=config["num_res_blocks"]
	time_embed_dim=config["time_embed_dim"]
	dropout = config["dropout"]

	# optim
	lr= config["lr"]
	beta1 = config["beta1"]
	beta2 = config["beta2"]
	eps = config["eps"]
	weight_decay = config["weight_decay"]
	warmup = config["warmup"]
	grad_clip = config["grad_clip"]
	ema_rate=config["ema_rate"]

	# train
	continuous = config["continuous"]
	reduce_mean = config["reduce_mean"]
	likelihood_weighting = config["likelihood_weighting"]
	beta_min = config["beta_min"]
	beta_max = config["beta_max"]
	sigma_min = config["sigma_min"]
	sigma_max = config["sigma_max"]
	save_path = config["save_path"]
	method = config["method"]

	if method =='Likelihood' or method =='All':

		sde = sde_lib.subVPSDE(beta_min=beta_min, beta_max=beta_max)

		likelihood_model = models.SimpleMLP(
		    channels=feat_dim,
		    bottleneck_channels=bottleneck_channels,
		    num_res_blocks=num_res_blocks,
		    time_embed_dim=time_embed_dim,
		    dropout=dropout,
		)

		optimizer = functools.partial(
		                torch.optim.Adam,
		                lr=lr,
		                betas=(beta1, beta2),
		                eps=eps,
		                weight_decay=weight_decay,
		                )

		likelihood_ood = likelihood.Likelihood(
		    sde = sde,
		    model = likelihood_model,
		    optimizer = optimizer,
		    ).to(device)

		update_fn = functools.partial(
		    losses.SDE_LRS_BF16, 
		    total_steps=len(train_dataset)//batch_size * n_epochs,
		    continuous=continuous,
		    reduce_mean=reduce_mean,
		    likelihood_weighting=likelihood_weighting,
		    )

		print("============================================================")
		print("Begin Likelihood Model Training!")
		print("============================================================")
		likelihood_train_loss =  likelihood_ood.fit(
			train_dataset,  
			n_epochs=n_epochs,
			batch_size=batch_size,
			update_fn=update_fn,
			)
		print("Training Completed!\n")
		likelihood_results = eval_utils.eval_ood(likelihood_model, train_dataset, val_dataset, ood_datasets, batch_size, verbose=False)
		print("============================================================")
		print("Saving Likelihood model results")
		print("============================================================")
	    plot_utils.plot(likelihood_results, train_dataset.name, [od.name for od in ood_datasets], encoder=encoder, model=model.name,
	                    train_loss=train_loss, save = True, path = save_path,)

	elif method == 'Residual' or method == 'All':
		#Residual Implementation    

	else:
		print("Error: Undefined OOD Method")
		exit-

    return print('Finished!')

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("config", type=str, help="Path to Config File")
	args = parser.parse_args()
	with open(args.config,"r") as file:
		config = yaml.safe_load(file)
	train(config)

if __name__ == '__main__':
	main()