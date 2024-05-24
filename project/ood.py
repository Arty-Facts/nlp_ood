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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pathlib


def plot(eval_data, id_name, ood_names, encoder, model, out_dir='figs', config=None, verbose=True, train_loss=None):
    if verbose:
        print('Generating plots...')
    # Unpack eval_data
    score, score_ref = eval_data['score'], eval_data['score_ref']
    ref_auc, ref_fpr = eval_data['ref_auc'], eval_data['ref_fpr']
    score_oods, auc_oods, fpr_oods = eval_data['score_oods'], eval_data['auc'], eval_data['fpr']

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Adjust the size as needed
    fig.suptitle(f'{model} Evaluation on {encoder}')

    def add_shadow(ax, data): 
        if data.var() > 1e-6:
            l = ax.lines[-1]
            x = l.get_xydata()[:,0]
            y = l.get_xydata()[:,1]
            ax.fill_between(x,y, alpha=0.1)
            # Calculate and plot the mean
            mean_value = np.mean(data)
            line_color = l.get_color()
            ax.axvline(mean_value, color=line_color, linestyle=':', linewidth=1.5)
    # Subplot 1: KDE plots
    sns.kdeplot(data=score, bw_adjust=.2, ax=axs[0, 0], label=f'{id_name} training: {np.mean(score):.2f}')
    add_shadow(axs[0, 0], score)

    sns.kdeplot(data=score_ref, bw_adjust=.2, ax=axs[0, 0], label=f'{id_name} validation: {np.mean(score_ref):.2f}')
    add_shadow(axs[0, 0], score_ref)

    for ood_name, score_ood in zip(ood_names, score_oods):
        sns.kdeplot(data=score_ood, bw_adjust=.2, ax=axs[0, 0], label=f'{ood_name}: {np.mean(score_ood):.2f}')
        add_shadow(axs[0, 0], score_ood)
    # axs[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
    # axs[0, 0].xaxis.set_minor_locator(ticker.MultipleLocator(base=0.1))
    axs[0, 0].set_title('Density Plots')
    axs[0, 0].set_xlabel('bits/dim')
    axs[0, 0].set_ylabel('Density')
    # axs[0, 0].set_xlim(6.5, 8)
    # axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Subplot 2: Bar chart for AUC and FPR
    x = np.arange(len(ood_names)+1)  # the label locations
    width = 0.35  # the width of the bars
    disp_auc = [ref_auc] + auc_oods
    disp_fpr = [ref_fpr] + fpr_oods
    rects1 = axs[0, 1].bar(x - width/2, disp_auc, width, label='AUC', alpha=0.6)
    rects2 = axs[0, 1].bar(x + width/2, disp_fpr, width, label='FPR', alpha=0.6)
    axs[0, 1].set_ylabel('Metric Value')
    axs[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(base=0.1))
    axs[0, 1].yaxis.set_minor_locator(ticker.MultipleLocator(base=0.05))
    axs[0, 1].set_title(f'AUC and FPR Metrics\nMean AUC: {np.mean(disp_auc[1:]):.2f}, Mean FPR: {np.mean(disp_fpr[1:]):.2f}')
    axs[0, 1].set_xticks(x)
    names = [f'{name}\nAUC: {auc:.2f}\nFPR: {fpr:.2f}' for name, auc, fpr in zip([id_name]+list(ood_names), disp_auc, disp_fpr)]
    axs[0, 1].set_xticklabels(names)
    axs[0, 1].grid(True)
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].legend()
    # add line at 0.5
    axs[0, 1].axhline(0.5, color='red', linestyle='--', linewidth=1.5) 

    if train_loss is not None:
        # Subplot 3: Training loss over time
        if isinstance(train_loss, list):
            train_loss = np.array(train_loss)
        if train_loss.ndim == 2:
            train_mean, train_std = train_loss.mean(axis=0), train_loss.std(axis=0)
            x = np.arange(len(train_mean))
            axs[1, 0].plot(x, train_mean, label='Training Loss')
            axs[1, 0].fill_between(x, train_mean-train_std, train_mean+train_std, alpha=0.1)
        else:
            axs[1, 0].plot(train_loss, label='Training Loss')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].set_title('Training Loss Over Time')
        axs[1, 0].grid(True)
        axs[1, 0].legend()
    else:
        axs[1, 0].axis('off')

    # Subplot 4: scatter plot of scores
    if score.ndim == 2:
        items, features = score.shape
        score1, score2 = np.mean(score[:items//2], axis=0), np.mean(score[items//2:], axis=0)
        axs[1, 1].scatter(score1, score2, alpha=0.5, label='ID Training', s=1)
        
        score_ref1, score_ref2 = np.mean(score_ref[:items//2], axis=0), np.mean(score_ref[items//2:], axis=0)
        axs[1, 1].scatter(score_ref1, score_ref2, alpha=0.5, label='ID Validation', s=1)

        for ood_name, score_ood in zip(ood_names, score_oods):
            score_ood1, score_ood2 = np.mean(score_ood[:items//2], axis=0), np.mean(score_ood[items//2:], axis=0)
            axs[1, 1].scatter(score_ood1, score_ood2, alpha=0.5, label=ood_name, s=1)
        axs[1, 1].set_xlabel(f'mean bits/dim for first {items//2} models')
        axs[1, 1].set_ylabel(f'mean bits/dim for last {items//2} models')
    else:
        axs[1, 1].off()
    


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout

    # Save the figure
    out_dir = pathlib.Path(out_dir) / encoder / id_name
    out_dir.mkdir(exist_ok=True, parents=True)
    filename = f"{encoder}_{model}_{id_name}_{int(np.mean(disp_auc[1:])*100)}.svg"
    plt.savefig(out_dir / filename, bbox_inches='tight')
    if verbose:
        plt.show()

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
	print("training data:", len(train_dataset))
	print("validation data:", len(val_dataset))
	print("ood data:", sum([len(d) for d in ood_datasets]))
	#network parameters
	bottleneck_channels=config["bottleneck_channels"]
	num_res_blocks=config["num_res_blocks"]
	time_embed_dim=config["time_embed_dim"]
	dropout = config["dropout"]
	encoder = encoder.replace('/','_') #for saving model

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
	pathlib.Path(save_path).mkdir(exist_ok=True, parents=True) 
	method = config["method"].lower()

	feat_dim=train_dataset.feat_dim

	#Initialise DataFrame variables
	df_residual = pd.DataFrame()
	df_likelihood = pd.DataFrame()

	if method =='likelihood' or method =='all':
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
		
		likelihood_results = eval_utils.eval_ood(likelihood_ood, train_dataset, val_dataset, ood_datasets, batch_size, verbose=False)
		
		plot(likelihood_results, train_dataset.name, [od.name for od in ood_datasets], encoder=encoder, model=likelihood_ood.name,
			train_loss=likelihood_train_loss, out_dir =save_path, verbose=False)
		
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

	if method == 'residual' or method == 'all':
		#Residual Implementation
		u=0
		dim = 512
		model_residual = Residual(dim, u)
		print("============================================================")
		print(f"Begin Residual Model Training for {encoder} Representation!")
		print("============================================================") 
		model_residual.fit(train_dataset)
		residual_results = eval_utils.eval_ood(model_residual, train_dataset, val_dataset, ood_datasets, batch_size, verbose=False)
		plot(residual_results, train_dataset.name, [od.name for od in ood_datasets], encoder=encoder, model=model_residual.name,
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