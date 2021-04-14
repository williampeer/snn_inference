import torch

load_path = '/Users/william/repos/archives_snn_inference/archive 9/saved/plot_data/01-20_15-05-33-991/plot_all_param_pairs_with_variance01-21_22-08-07-671.pt'

# import saved model, import target model
data = torch.load(load_path)
print('Loaded saved plot data.')

plot_data = data['plot_data']

# full dimensionality
m_ps = plot_data['param_means']  # so not really means, but full after training
t_ps = plot_data['target_params']
p_names = plot_data['param_names']

for p_i, pname in enumerate(p_names):
    print(p_i, pname)

# calc euclid dist all weights || per pop./ensemble

# ----------

# iff model saved at train iter. #0:
#   calc. dist at #0 and #-1 to target. plot/report relative improvement (%?)
