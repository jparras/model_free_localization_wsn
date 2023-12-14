import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import pickle
import gc
import platform
import os


if __name__ == '__main__':

    number_of_seeds = 1
    channels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    snr_v = [20, 10, 0]
    feat_types = ['power', 'cov']
    nn_vals = [8, 16, 32, 64, 128]
    seq_length = 10
    rep = 100
    n_epochs = 1000


    # Training loop
    for index_of_channel in range(len(channels)):
        for snr in snr_v:
            data = {'channel_option': channels[index_of_channel], 'snr': snr, 'feat_types': feat_types,
                    'nn_vals': nn_vals, 'number_of_seeds': number_of_seeds, 'rep': rep, 'seq_length': seq_length,
                    'n_epochs': n_epochs}
            with open('aux.pickle', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if platform.system() == 'Windows':
                print("Running on Windows")
                _ = os.system('python  channel_generator.py')  # Windows order
            else:
                print("Running on Linux")
                _ = os.system('python3 channel_generator.py')  # Linux order
            os.remove('aux.pickle')
            gc.collect()

    # Obtain final plots
    tr_curves = np.zeros([len(channels), len(snr_v), len(nn_vals), len(feat_types), number_of_seeds, 1000])
    tr_curves_val = np.zeros_like(tr_curves)
    tr_curves_mse = np.zeros_like(tr_curves)
    tr_curves_val_mse = np.zeros_like(tr_curves)
    for index_of_channel in range(len(channels)):
        for snr in snr_v:
            for nn in nn_vals:
                for feat in feat_types:
                    for seed in range(number_of_seeds):
                        name = str(channels[index_of_channel]) + '_' + str(snr) + '_' + feat + '_' + str(nn) + '_' + str(seed)
                        with open(name + '_data.pickle', 'rb') as handle:
                            data = pickle.load(handle)
                        tr_curves[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat), seed,
                        :] = data['train_mae']
                        tr_curves_val[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat),
                        seed, :] = data['val_mae']
                        tr_curves_mse[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat),
                        seed, :] = data['train']
                        tr_curves_val_mse[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat),
                        seed, :] = data['val']

    # Plot values
    for index_of_channel in range(len(channels)):
        for snr in snr_v:
            for nn in nn_vals:
                for feat in feat_types:
                    mean = np.mean(
                        tr_curves[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat), :, :],
                        axis=0)
                    std = np.std(
                        tr_curves[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat), :, :],
                        axis=0)
                    ma = np.amax(
                        tr_curves[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat), :, :],
                        axis=0)
                    mi = np.amin(
                        tr_curves[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat), :, :],
                        axis=0)
                    name = str(channels[index_of_channel]) + '_' + str(snr) + '_' + feat + '_' + str(nn)
                    plt.semilogy(mean, label=name)
                    plt.fill_between(np.arange(1000), ma, mi, alpha=0.3)
    plt.legend(loc='best')
    tikz_save('all_loss.tikz', figureheight='\\figureheight', figurewidth='\\figurewidth')
    plt.savefig('all_loss.png', bbox_inches='tight')
    plt.close()

    # Final values
    for index_of_channel in range(len(channels)):
        for snr in snr_v:
            for nn in nn_vals:
                for feat in feat_types:
                    print('Channel = ', channels[index_of_channel], "; SNR = ", snr, "; neurons = ", nn, "; feature = ",
                          feat,
                          "; Train MAE = ", np.mean(
                            tr_curves[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat), :,
                            -1]),
                          ' +- ', np.std(
                            tr_curves[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat), :,
                            -1]),
                          "; Val MAE = ", np.mean(
                            tr_curves_val[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat),
                            :, -1]),
                          ' +- ', np.std(
                            tr_curves_val[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat),
                            :, -1]))
    for index_of_channel in range(len(channels)):
        for snr in snr_v:
            for nn in nn_vals:
                for feat in feat_types:
                    print('Channel = ', channels[index_of_channel], "; SNR = ", snr, "; neurons = ", nn, "; feature = ",
                          feat,
                          "; Train MSE = ", np.mean(
                            tr_curves_mse[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat),
                            :, -1]),
                          ' +- ', np.std(
                            tr_curves_mse[index_of_channel, snr_v.index(snr), nn_vals.index(nn), feat_types.index(feat),
                            :, -1]),
                          "; Val MSE = ", np.mean(
                            tr_curves_val_mse[index_of_channel, snr_v.index(snr), nn_vals.index(nn),
                            feat_types.index(feat), :, -1]),
                          ' +- ', np.std(tr_curves_val_mse[index_of_channel, snr_v.index(snr), nn_vals.index(nn),
                                         feat_types.index(feat), :, -1]))
