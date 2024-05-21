import argparse
import yaml
import functools
import pathlib
import torch
import torch.nn as nn
import numpy as np
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

def train(config, encoder):
	#data parameters
	ind = config["idd"]
	ood = config["ood"]
	device = config["device"]
	token_len = config["token_len"]
	datasets = ind+ood

	#load dataset
	train_dataset, val_dataset, *ood_datasets = load_datasets(id = ind, ood=ood, embed_model=encoder, device=device, token_len=token_len)

	#network parameters
	bottleneck_channels=config["bottleneck_channels"]
	num_res_blocks=config["num_res_blocks"]
	time_embed_dim=config["time_embed_dim"]
	dropout = config["dropout"]

	# optim
	n_epochs = config['epoch']
	batch_size = config['batch_size']
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

	feat_dim=train_dataset.feat_dim

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
		likelihood_results = eval_utils.eval_ood(likelihood_ood, train_dataset, val_dataset, ood_datasets, batch_size, verbose=False)
		print("============================================================")
		print("Saving Likelihood model results")
		print("============================================================")
		plot_utils.plot(likelihood_results, train_dataset.name, [od.name for od in ood_datasets], encoder=encoder, model=likelihood_ood.name,
				  train_loss=likelihood_train_loss, out_dir = save_path,verbose=False)
		auc = np.append(likelihood_results['ref_auc'],likelihood_results['auc'])
		fpr = np.append(likelihood_results['ref_fpr'],likelihood_results['fpr'])
		df_likelihood = pd.DataFrame({'AUC_Likelihood': auc, 'AUC_Likelihood':fpr}, index=datasets)
		df_likelihood = df_likelihood.transpose()
		df_likelihood['Encoder']= encoder
		del likelihood_ood

	elif method == 'Residual' or method == 'All':
		#Residual Implementation
		u=0
		dim = 512
		model_residual = Residual(dim, u)
		print("============================================================")
		print("Begin Residual Model Training!")
		print("============================================================") 
		model_residual.fit(train_dataset)
		residual_results = eval_utils.eval_ood(model_residual, train_dataset, val_dataset, ood_datasets, batch_size, verbose=False)
		print("============================================================")
		print("Saving Residual model results")
		print("============================================================")
		plot_utils.plot(residual_results, train_dataset.name, [od.name for od in ood_datasets], encoder=encoder, model=model_residual.name,
                train_loss=None, out_dir =save_path, verbose=False)
		auc = np.append(residual_results['ref_auc'],residual_results['auc'])
		fpr = np.append(residual_results['ref_fpr'],residual_results['fpr'])
		df_residual = pd.DataFrame({'AUC_Residual': auc, 'AUC_Residual':fpr}, index=datasets)
		df_residual = df_residual.transpose()
		df_residual['Encoder']= encoder 
	
	else:
		print("Error: Undefined OOD Method")
		exit
	
	if df_residual.empty() and df_likelihood.empty():
		print('Error: No evaluation results')
	
	elif not df_residual.empty():
		df = df_residual.copy()
	
	elif not df_likelihood.empty():
		df = df_likelihood.copy()
	
	else:
		df = df_likelihood.append(df_residual)

	del train_dataset, val_dataset, ood_datasets
	torch.cuda.empty_cache()
	return df

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("config", type=str, help="Path to Config File")
	args = parser.parse_args()
	with open(args.config,"r") as file:
		config = yaml.safe_load(file)
	encoders = config["encoders"]
	dfs = []
	for encoder in encoders:
		df = train(config, encoder)
		dfs.append(df)
	print(result_df)
	result_df = pd.concat(dfs, ignore_index=True)
	df.to_csv(config['save_path']+config['config'], index=False)
	print('Finished!')

if __name__ == '__main__':
	main()