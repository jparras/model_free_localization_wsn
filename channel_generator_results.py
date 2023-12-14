import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import pickle


def get_cmap(n, name='hsv'):
    '''
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    '''
    return plt.cm.get_cmap(name, n)


if __name__ == '__main__':

    number_of_seeds = 1
    channels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    snr_v = [20, 10, 0]
    feat_types = ['power', 'cov']
    nn_vals = [8, 16, 32, 64, 128]
    seq_length = 10
    rep = 100
    n_epochs = 1000

    save_figs = True

    # Load data for plotting
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
    # Plot 1: Fixed SNR, all features, mean reuslts
    colors = get_cmap(len(nn_vals) + 1)  # Differentiate nn by colors
    markers = ['o', 's'] # Differentiate features by markers
    lstyle = ['-', '--']
    for snr in snr_v:
        for nn in nn_vals:
            for feat in feat_types:
                values = np.zeros(len(channels))
                for index_of_channel in range(len(channels)):
                    values[index_of_channel] = np.mean(tr_curves_val[index_of_channel, snr_v.index(snr), nn_vals.index(nn),
                                                       feat_types.index(feat), :, -1])
                plt.plot(channels, values, color=colors(nn_vals.index(nn)), marker=markers[feat_types.index(feat)],
                         label='nn = ' + str(nn) + '; feat = ' + str(feat), linestyle=lstyle[feat_types.index(feat)])
        plt.legend(loc='best')
        plt.title('Validation MAE values for SNR = ' + str(snr))
        if save_figs:
            tikz_save('all_loss_' + str(snr) + '.tikz', figureheight='\\figureheight', figurewidth='\\figurewidth')
            plt.savefig('all_loss_' + str(snr) + '.png', bbox_inches='tight')
        plt.show()

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