import sys
sys.path.append("src/")
from src import *


ct = 'LA'
data = pd.read_parquet(f"data/preprocessed_data/{ct}/data.parquet")
prior = pd.read_parquet(f"data/preprocessed_data/{ct}/prior.parquet")

dp1 = 0.05
dp2 = 0
alpha_array = [0.6,0.7,0.8,0.9]
print(f"dp1: {dp1}, dp2: {dp2}\n")
for alpha in alpha_array:
    fit_model(data, prior, output_path=f"./result_{ct}_threshold_1_alpha_{alpha}/dp1_{dp1}_dp2_{dp2}/", 
            data_val_size=0.3, batch_size=32, fraction_gs=0.2, 
            num_epochs=200, cvs=5, num_epochs_refit=50, refit_iters=10, refit_resample=True, 
            weight_decays=(-10, -1, 4), lr=1e-4, 
            scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR, scheduler_kwargs={'T_max': 10}, 
            optimizer_class=torch.optim.Adam, optimizerkw={}, optimizer_paramskw={},
            dropout_rate1=dp1, dropout_rate2=dp2, dropout_rate3=dp2, activation=ReLU0(),
            eps=torch.finfo(torch.float).eps, eps_factor=10, fill_zeroed=True, device='cuda:1',alpha = alpha)