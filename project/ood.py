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
	verbose = config.get("verbose", False)

	#load dataset

	print("============================================================")
	print(f"Loading datasets for {encoder} Representation!")
	print("============================================================")
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
	beta1 = config.get("beta1", 0.9)
	beta2 = config.get("beta2", 0.999)
	eps = config.get("eps", 1e-8)
	weight_decay = config.get("weight_decay", 0.0)


	# train
	continuous = config.get("continuous", True)
	reduce_mean = config.get("reduce_mean", True)
	likelihood_weighting = config.get("likelihood_weighting", False)
	beta_min = config["beta_min"]
	beta_max = config["beta_max"]
	save_path = config["save_path"]
	method = config["method"]

	feat_dim=train_dataset.feat_dim

	#Initialise DataFrame variables
	df_residual = pd.DataFrame()
	df_likelihood = pd.DataFrame()

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
		    losses.SDE_BF16, 
		    continuous=continuous,
		    reduce_mean=reduce_mean,
		    likelihood_weighting=likelihood_weighting,
		    )

		print("============================================================")
		print(f"Begin Likelihood Model Training for {encoder} Representation!")
		print("============================================================")
		likelihood_train_loss =  likelihood_ood.fit(
			train_dataset,  
			n_epochs=n_epochs,
			batch_size=batch_size,
			update_fn=update_fn,
			verbose=verbose,
			)
		print("Training Completed!\n")
		likelihood_results = eval_utils.eval_ood(likelihood_ood, train_dataset, val_dataset, ood_datasets, batch_size, verbose=False)
		print("============================================================")
		print("Saving Likelihood model & Results")
		print("============================================================")
		plot_utils.plot(likelihood_results, train_dataset.name, [od.name for od in ood_datasets], encoder=encoder, model=likelihood_ood.name,
				  train_loss=likelihood_train_loss, out_dir = save_path,verbose=False)
		out_dir = pathlib.Path(save_path) / encoder / ind[0]
		out_dir.mkdir(exist_ok=True, parents=True)
		filename = f"{encoder}_{ind[0]}_likelihood.pth"
		torch.save(likelihood_ood.state_dict(), out_dir/filename)
		auc = [likelihood_results['ref_auc']] + likelihood_results['auc'] + ['AUC']
		fpr = [likelihood_results['ref_fpr']] + likelihood_results['fpr'] + ['FPR']
		df_likelihood = pd.DataFrame([auc,fpr], columns=datasets+['Score'])
		df_likelihood['Method'] = 'Likelihood'
		df_likelihood['Encoder']= encoder
		print(df_likelihood)
		del likelihood_ood

	if method == 'Residual' or method == 'All':
		#Residual Implementation
		u=0
		dim = 512
		model_residual = Residual(dim, u)
		print("============================================================")
		print(f"Begin Residual Model Training for {encoder} Representation!")
		print("============================================================") 
		model_residual.fit(train_dataset)
		residual_results = eval_utils.eval_ood(model_residual, train_dataset, val_dataset, ood_datasets, batch_size, verbose=False)
		print("============================================================")
		print("Saving Residual model & Results")
		print("============================================================")
		plot_utils.plot(residual_results, train_dataset.name, [od.name for od in ood_datasets], encoder=encoder, model=model_residual.name,
                train_loss=None, out_dir =save_path, verbose=False)
		out_dir = pathlib.Path(save_path) / encoder / ind[0]
		out_dir.mkdir(exist_ok=True, parents=True)
		filename = f"{encoder}_{ind[0]}_residual.pth"
		torch.save(model_residual, out_dir/filename)
		auc = [residual_results['ref_auc']] + residual_results['auc'] + ['AUC']
		fpr = [residual_results['ref_fpr']] + residual_results['fpr'] + ['FPR']
		df_residual = pd.DataFrame([auc,fpr], columns=datasets+['Score'])
		df_residual['Method']='Residual'
		df_residual['Encoder']= encoder 
		print(df_residual)
	if df_residual.empty and df_likelihood.empty:
		print('Error: No evaluation results')
	else:
		df = pd.concat([df_likelihood, df_residual], ignore_index=True)

	del train_dataset, val_dataset, ood_datasets
	torch.cuda.empty_cache()
	print("============================================================")
	print(f"Completed training on {encoder} Representation!!")
	print("============================================================")
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
	result_df = pd.concat(dfs, ignore_index=True)
	print(result_df)
	result_df.to_csv(config['save_path']+config['config'], index=False)
	print('Finished!')

if __name__ == '__main__':
	main()