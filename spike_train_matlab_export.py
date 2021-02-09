import sys

import numpy as np
import torch

from IO import save_model_params
from data_util import save_spiketrain_in_sparse_matlab_format, convert_to_sparse_vectors
from experiments import poisson_input
from model_util import generate_model_data
from plot import plot_spike_train


def main():
    # model_path = '/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-05-33-991/'
    # model_path = ''

    model_paths = []
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-04-49-336/LIF_exp_num_0_data_set_None_mean_loss_65.547_uuid_01-20_15-04-49-336.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-04-49-336/LIF_exp_num_1_data_set_None_mean_loss_61.167_uuid_01-20_15-04-49-336.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-04-49-336/LIF_exp_num_2_data_set_None_mean_loss_38.016_uuid_01-20_15-04-49-336.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-04-49-336/LIF_exp_num_3_data_set_None_mean_loss_39.588_uuid_01-20_15-04-49-336.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-04-49-336/LIF_exp_num_4_data_set_None_mean_loss_27.772_uuid_01-20_15-04-49-336.pt']
    #
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_22-30-30-455/LIF_exp_num_0_data_set_None_mean_loss_30.836_uuid_01-21_22-30-30-455.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_22-30-30-455/LIF_exp_num_1_data_set_None_mean_loss_13.194_uuid_01-21_22-30-30-455.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_22-30-30-455/LIF_exp_num_2_data_set_None_mean_loss_31.775_uuid_01-21_22-30-30-455.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_22-30-30-455/LIF_exp_num_3_data_set_None_mean_loss_18.227_uuid_01-21_22-30-30-455.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_22-30-30-455/LIF_exp_num_4_data_set_None_mean_loss_14.051_uuid_01-21_22-30-30-455.pt']
    #
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_06-07-29-245/LIF_exp_num_0_data_set_None_mean_loss_317.612_uuid_01-23_06-07-29-245.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_06-07-29-245/LIF_exp_num_1_data_set_None_mean_loss_54.696_uuid_01-23_06-07-29-245.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_06-07-29-245/LIF_exp_num_2_data_set_None_mean_loss_50.540_uuid_01-23_06-07-29-245.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_06-07-29-245/LIF_exp_num_3_data_set_None_mean_loss_71.706_uuid_01-23_06-07-29-245.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_06-07-29-245/LIF_exp_num_4_data_set_None_mean_loss_59.141_uuid_01-23_06-07-29-245.pt']

    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-06-33-420/LIF_exp_num_0_data_set_None_mean_loss_6.120_uuid_01-20_15-06-33-420.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-06-33-420/LIF_exp_num_1_data_set_None_mean_loss_4.918_uuid_01-20_15-06-33-420.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-06-33-420/LIF_exp_num_2_data_set_None_mean_loss_5.139_uuid_01-20_15-06-33-420.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-06-33-420/LIF_exp_num_3_data_set_None_mean_loss_5.081_uuid_01-20_15-06-33-420.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-06-33-420/LIF_exp_num_4_data_set_None_mean_loss_5.075_uuid_01-20_15-06-33-420.pt']
    #
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_16-48-38-729/LIF_exp_num_0_data_set_None_mean_loss_5.367_uuid_01-21_16-48-38-729.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_16-48-38-729/LIF_exp_num_1_data_set_None_mean_loss_3.487_uuid_01-21_16-48-38-729.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_16-48-38-729/LIF_exp_num_2_data_set_None_mean_loss_3.918_uuid_01-21_16-48-38-729.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_16-48-38-729/LIF_exp_num_3_data_set_None_mean_loss_4.155_uuid_01-21_16-48-38-729.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_16-48-38-729/LIF_exp_num_4_data_set_None_mean_loss_4.461_uuid_01-21_16-48-38-729.pt']
    #
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_18-58-47-549/LIF_exp_num_0_data_set_None_mean_loss_5.757_uuid_01-22_18-58-47-549.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_18-58-47-549/LIF_exp_num_1_data_set_None_mean_loss_4.164_uuid_01-22_18-58-47-549.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_18-58-47-549/LIF_exp_num_2_data_set_None_mean_loss_4.561_uuid_01-22_18-58-47-549.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_18-58-47-549/LIF_exp_num_3_data_set_None_mean_loss_4.804_uuid_01-22_18-58-47-549.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_18-58-47-549/LIF_exp_num_4_data_set_None_mean_loss_4.790_uuid_01-22_18-58-47-549.pt']

    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-05-16-907/LIF_exp_num_0_data_set_None_mean_loss_52.921_uuid_01-20_15-05-16-907.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-05-16-907/LIF_exp_num_1_data_set_None_mean_loss_38.483_uuid_01-20_15-05-16-907.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-05-16-907/LIF_exp_num_2_data_set_None_mean_loss_49.349_uuid_01-20_15-05-16-907.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-05-16-907/LIF_exp_num_3_data_set_None_mean_loss_39.583_uuid_01-20_15-05-16-907.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-05-16-907/LIF_exp_num_4_data_set_None_mean_loss_37.063_uuid_01-20_15-05-16-907.pt']
    #
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_15-38-22-857/LIF_exp_num_0_data_set_None_mean_loss_43.225_uuid_01-21_15-38-22-857.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_15-38-22-857/LIF_exp_num_1_data_set_None_mean_loss_15.598_uuid_01-21_15-38-22-857.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_15-38-22-857/LIF_exp_num_2_data_set_None_mean_loss_34.582_uuid_01-21_15-38-22-857.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_15-38-22-857/LIF_exp_num_3_data_set_None_mean_loss_32.154_uuid_01-21_15-38-22-857.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_15-38-22-857/LIF_exp_num_4_data_set_None_mean_loss_22.386_uuid_01-21_15-38-22-857.pt']
    #
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_16-37-23-061/LIF_exp_num_0_data_set_None_mean_loss_82.193_uuid_01-22_16-37-23-061.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_16-37-23-061/LIF_exp_num_1_data_set_None_mean_loss_39.025_uuid_01-22_16-37-23-061.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_16-37-23-061/LIF_exp_num_2_data_set_None_mean_loss_62.308_uuid_01-22_16-37-23-061.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_16-37-23-061/LIF_exp_num_3_data_set_None_mean_loss_40.546_uuid_01-22_16-37-23-061.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_16-37-23-061/LIF_exp_num_4_data_set_None_mean_loss_54.480_uuid_01-22_16-37-23-061.pt']
    #
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_18-54-27-129/LIF_exp_num_0_data_set_None_mean_loss_51.368_uuid_01-23_18-54-27-129.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_18-54-27-129/LIF_exp_num_1_data_set_None_mean_loss_29.710_uuid_01-23_18-54-27-129.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_18-54-27-129/LIF_exp_num_2_data_set_None_mean_loss_45.786_uuid_01-23_18-54-27-129.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_18-54-27-129/LIF_exp_num_3_data_set_None_mean_loss_36.853_uuid_01-23_18-54-27-129.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_18-54-27-129/LIF_exp_num_4_data_set_None_mean_loss_38.116_uuid_01-23_18-54-27-129.pt']

    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_22-08-07-832/LIF_exp_num_0_data_set_None_mean_loss_16.059_uuid_01-21_22-08-07-832.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_22-08-07-832/LIF_exp_num_1_data_set_None_mean_loss_16.429_uuid_01-21_22-08-07-832.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_22-08-07-832/LIF_exp_num_2_data_set_None_mean_loss_19.590_uuid_01-21_22-08-07-832.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_22-08-07-832/LIF_exp_num_3_data_set_None_mean_loss_13.801_uuid_01-21_22-08-07-832.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_22-08-07-832/LIF_exp_num_4_data_set_None_mean_loss_12.059_uuid_01-21_22-08-07-832.pt']
    #
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_05-11-39-091/LIF_exp_num_0_data_set_None_mean_loss_60.484_uuid_01-23_05-11-39-091.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_05-11-39-091/LIF_exp_num_1_data_set_None_mean_loss_58.241_uuid_01-23_05-11-39-091.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_05-11-39-091/LIF_exp_num_2_data_set_None_mean_loss_46.015_uuid_01-23_05-11-39-091.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_05-11-39-091/LIF_exp_num_3_data_set_None_mean_loss_44.075_uuid_01-23_05-11-39-091.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-23_05-11-39-091/LIF_exp_num_4_data_set_None_mean_loss_165.832_uuid_01-23_05-11-39-091.pt']
    #
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-07-44-795/LIF_exp_num_0_data_set_None_mean_loss_4.951_uuid_01-20_15-07-44-795.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-07-44-795/LIF_exp_num_1_data_set_None_mean_loss_4.954_uuid_01-20_15-07-44-795.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-07-44-795/LIF_exp_num_2_data_set_None_mean_loss_4.967_uuid_01-20_15-07-44-795.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-07-44-795/LIF_exp_num_3_data_set_None_mean_loss_4.631_uuid_01-20_15-07-44-795.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-20_15-07-44-795/LIF_exp_num_4_data_set_None_mean_loss_4.150_uuid_01-20_15-07-44-795.pt']
    #
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_18-12-56-967/LIF_exp_num_0_data_set_None_mean_loss_3.515_uuid_01-21_18-12-56-967.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_18-12-56-967/LIF_exp_num_1_data_set_None_mean_loss_3.565_uuid_01-21_18-12-56-967.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_18-12-56-967/LIF_exp_num_2_data_set_None_mean_loss_3.562_uuid_01-21_18-12-56-967.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_18-12-56-967/LIF_exp_num_3_data_set_None_mean_loss_3.629_uuid_01-21_18-12-56-967.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-21_18-12-56-967/LIF_exp_num_4_data_set_None_mean_loss_3.239_uuid_01-21_18-12-56-967.pt']
    #
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_21-37-40-455/LIF_exp_num_0_data_set_None_mean_loss_4.354_uuid_01-22_21-37-40-455.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_21-37-40-455/LIF_exp_num_1_data_set_None_mean_loss_4.304_uuid_01-22_21-37-40-455.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_21-37-40-455/LIF_exp_num_2_data_set_None_mean_loss_4.200_uuid_01-22_21-37-40-455.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_21-37-40-455/LIF_exp_num_3_data_set_None_mean_loss_4.315_uuid_01-22_21-37-40-455.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 7/saved/01-22_21-37-40-455/LIF_exp_num_4_data_set_None_mean_loss_4.048_uuid_01-22_21-37-40-455.pt']

    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_13-44-20-353/LIF_exp_num_0_data_set_None_mean_loss_25.227_uuid_01-24_13-44-20-353.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_13-44-20-353/LIF_exp_num_1_data_set_None_mean_loss_25.457_uuid_01-24_13-44-20-353.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_13-44-20-353/LIF_exp_num_2_data_set_None_mean_loss_25.986_uuid_01-24_13-44-20-353.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_13-44-20-353/LIF_exp_num_3_data_set_None_mean_loss_19.753_uuid_01-24_13-44-20-353.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_13-44-20-353/LIF_exp_num_4_data_set_None_mean_loss_17.947_uuid_01-24_13-44-20-353.pt']
    #
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_02-32-43-141/LIF_exp_num_0_data_set_None_mean_loss_4.592_uuid_01-24_02-32-43-141.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_02-32-43-141/LIF_exp_num_1_data_set_None_mean_loss_4.534_uuid_01-24_02-32-43-141.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_02-32-43-141/LIF_exp_num_2_data_set_None_mean_loss_4.640_uuid_01-24_02-32-43-141.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_02-32-43-141/LIF_exp_num_3_data_set_None_mean_loss_4.596_uuid_01-24_02-32-43-141.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_02-32-43-141/LIF_exp_num_4_data_set_None_mean_loss_3.612_uuid_01-24_02-32-43-141.pt']

    # # lr 0.01
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-25_03-47-10-327/LIF_exp_num_0_data_set_None_mean_loss_54.980_uuid_01-25_03-47-10-327.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-25_03-47-10-327/LIF_exp_num_1_data_set_None_mean_loss_39.062_uuid_01-25_03-47-10-327.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-25_03-47-10-327/LIF_exp_num_2_data_set_None_mean_loss_45.534_uuid_01-25_03-47-10-327.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-25_03-47-10-327/LIF_exp_num_3_data_set_None_mean_loss_34.986_uuid_01-25_03-47-10-327.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-25_03-47-10-327/LIF_exp_num_4_data_set_None_mean_loss_34.768_uuid_01-25_03-47-10-327.pt']

    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_15-14-04-415/LIF_exp_num_0_data_set_None_mean_loss_40.321_uuid_01-24_15-14-04-415.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_15-14-04-415/LIF_exp_num_1_data_set_None_mean_loss_33.254_uuid_01-24_15-14-04-415.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_15-14-04-415/LIF_exp_num_2_data_set_None_mean_loss_37.121_uuid_01-24_15-14-04-415.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_15-14-04-415/LIF_exp_num_3_data_set_None_mean_loss_36.953_uuid_01-24_15-14-04-415.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-24_15-14-04-415/LIF_exp_num_4_data_set_None_mean_loss_23.412_uuid_01-24_15-14-04-415.pt']

    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-25_11-53-35-101/LIF_exp_num_0_data_set_None_mean_loss_42.144_uuid_01-25_11-53-35-101.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-25_11-53-35-101/LIF_exp_num_1_data_set_None_mean_loss_18.351_uuid_01-25_11-53-35-101.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-25_11-53-35-101/LIF_exp_num_2_data_set_None_mean_loss_38.188_uuid_01-25_11-53-35-101.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-25_11-53-35-101/LIF_exp_num_3_data_set_None_mean_loss_30.634_uuid_01-25_11-53-35-101.pt']
    # model_paths += ['/Users/william/repos/archives_snn_inference/archive 9/saved/01-25_11-53-35-101/LIF_exp_num_4_data_set_None_mean_loss_25.222_uuid_01-25_11-53-35-101.pt']

    # model_path = '/Users/william/repos/archives_snn_inference/archive 9/saved/'

    for mp in model_paths:
        load_and_export_sim_data(mp)


def load_and_export_sim_data(model_path, fname=False, t = 60 * 1000):
    # print('Argument List:', str(argv))

    # opts = [opt for opt in argv if opt.startswith("-")]
    # args = [arg for arg in argv if not arg.startswith("-")]

    # loss_fn = model_path.split('loss_fn_')[1].split('_budget')[0]
    # cur_model_descr = model_path.split('fitted_params_')[1].split('_optim')[0]
    cur_model_name = model_path.split('_exp_num')[0].split('/')[-1]
    exp_num = model_path.split('exp_num_')[1].split('_data_set')[0]
    # optim = model_path.split('optim_')[1].split('_loss_fn')[0]
    # id = optim + '_' + model_path.split('.pt')[0].split('-')[-1]
    # lr = ''


    # for i, opt in enumerate(opts):
    #     if opt == '-h':
    #         print('load_and_export_model_data.py -p <path> -t <time> -r <poisson-rate>')
    #         sys.exit()
    #     elif opt in ("-p", "--path"):
    #         model_path = args[i]
    #     elif opt in ("-t", "--time"):
    #         t = int(args[i])
    #
    # if model_path is None:
    #     print('No path to load model from specified.')
    #     sys.exit(1)
    # poisson_rate = 10.

    exp_res = torch.load(model_path)
    model = exp_res['model']
    poisson_rate = exp_res['rate']
    # loss = data['loss']

    print('Loaded model.')

    interval_size = 4000
    interval_range = int(t/interval_size)
    assert interval_range > 0, "t must be greater than the interval size, {}. t={}".format(interval_size, t)

    spike_indices = np.array([], dtype='int8')
    spike_times = np.array([], dtype='float32')
    input_indices = np.array([], dtype='int8')
    input_times = np.array([], dtype='float32')
    print('Simulating data..')
    for t_i in range(interval_range):
        model.reset_hidden_state()
        # spiketrain = generate_synthetic_data(model, poisson_rate, t=interval_size)
        gen_input = poisson_input(rate=poisson_rate, t=interval_size, N=model.N)
        gen_spiketrain = generate_model_data(model=model, inputs=gen_input)
        # for gen spiketrain this may be thresholded to binary values:
        gen_spiketrain = torch.round(gen_spiketrain)

        inputs = gen_input.clone().detach()
        spiketrain = gen_spiketrain.clone().detach()

        cur_input_indices, cur_input_times = convert_to_sparse_vectors(inputs, t_offset=t_i*interval_size)
        input_indices = np.append(input_indices, cur_input_indices)
        input_times = np.append(input_times, cur_input_times)
        cur_spike_indices, cur_spike_times = convert_to_sparse_vectors(spiketrain, t_offset=t_i*interval_size)
        spike_indices = np.append(spike_indices, cur_spike_indices)
        spike_times = np.append(spike_times, cur_spike_times)
        print('{} seconds ({:.2f} min) simulated.'.format(interval_size * (t_i+1)/1000., interval_size * (t_i+1)/(60.*1000)))

    # fname = model_path.split('/')[-1]
    # model_name = fname.split('.pt')[0]
    # save_fname_input = 'poisson_inputs_{}_t_{:.0f}s_rate_{}'.format(model_name, t/1000., poisson_rate).replace('.', '_') + '.mat'
    # save_spiketrain_in_sparse_matlab_format(fname=save_fname_input, spike_indices=input_indices, spike_times=input_times)
    # save_model_params(model, fname=save_fname_input.replace('.mat', '_params'))
    save_fname_output = 'fitted_spikes_'.format(cur_model_name, id, exp_num).replace('.', '_')
    if not fname:
        fname = save_fname_output
    plot_spike_train(spiketrain, 'Plot imported SNN', 'plot_imported_model', fname=fname)

    # save_fname_output = 'fitted_spike_train_{}_{}_{}{}_exp_num_{}'.format(cur_model_descr, optim, loss_fn, lr, exp_num).replace('.', '_') + '.mat'
    # save_fname_output = 'fitted_spike_train_{}_id_{}_exp_no_{}'.format(cur_model_name, id, exp_num).replace('.', '_') + '.mat'
    save_spiketrain_in_sparse_matlab_format(fname=fname + '.mat', spike_indices=spike_indices, spike_times=spike_times)
    save_model_params(model, fname=save_fname_output.replace('.mat', '_params'))


if __name__ == "__main__":
    main()
    sys.exit(0)
