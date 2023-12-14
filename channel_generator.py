import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
from keras.models import Sequential
from keras import layers
from keras import metrics
from keras import optimizers
import os
import datetime
from matplotlib2tikz import save as tikz_save
import pickle
from collections import Counter
import gc


def range_m(init, end, step):  # Matlab-like range, including end!
    out = np.arange(init, end, step)
    if out[-1] + step <= end:  # Include end only if it is in the interval
        out = np.append(out, out[-1] + step)
    return out


def power(complex_array):
    return np.sum(np.power(np.absolute(complex_array), 2))


class channel():
    def __init__(self, params=None, number_of_processes=-1, load=False, name='0'):
        # params: channel parameters (default if None)
        # number of processes: for parallelization (1 for single, -1 for as much as possible)
        self.h = None  # To store the impulse reponse
        self.traj = None  # To store trajectories
        self.nop = number_of_processes
        if self.nop == -1:
            self.nop = cpu_count()

        self.params = params
        if self.params is None:
            print('Loading channel params: default used')
            self.default_params()
        else:
            print('Loading channel params: non default used')
        if load:
            try:
                print('Loading channel matrix')
                self.h = np.load('channel_h_' + name + '.npy')
                self.traj = np.load('trajs_' + name + '.npy')
            except:
                print('Obtaining channel matrix')
                self.h = self.obtain_h().astype(np.complex64)
                np.save('channel_h_' + name, self.h)
                np.save('trajs_' + name, self.traj)
        else:
            print('Obtaining channel matrix')
            self.h = self.obtain_h().astype(np.complex64)
            np.save('channel_h_' + name, self.h)
            np.save('trajs_' + name, self.traj)

    def default_params(self):
        channel_info = {'h0': 50,  # Surface height(depth)[m]
                        'ht0': 50,  # TX height [m]
                        'hr0': 1,  # RX height [m]
                        'd0': 500,  # channel distance [m]
                        'k': 1.700000,  # spreading_factor
                        'c': 1500.000000,  # speed_of_sound_in_water_[m / s]
                        'c2': 1200.000000,  # speed_of_sound_in_bottom_[m / s]
                        'cut': 50.000000,  # minimum_relative_path_strength
                        'fmin': 10000.000000,  # minimum_frequency_[Hz]
                        'B': 10000.000000,  # bandwidth_[Hz]
                        'df': 25.000000,  # frequency_resolution_[Hz]
                        'dt': 6.045000,  # time_resolution_[seconds]
                        'T_SS': 6.000000,  # coherence_time_of_the_small - scale_variations_[seconds]
                        'sig2s': 0.0,  # 1.125000, # variance_of_S_S_surface_variations_[m ^ 2]
                        'sig2b': 0.0,  # 0.562500, # variance_of_S_S_bottom_variations_[m ^ 2]
                        'B_delp': 0.000500,  # 3 - dB_width_of_the_p.s.d._of_intra - path_delays_[Hz]
                        'Sp': 0,  # 20, # number_of_intra - paths
                        'mu_p': 0.0,  # 0.025000, # mean_of_intra - path_amplitudes
                        'nu_p': 0.0,  # 0.000001, # variance_of_intra - path_amplitudes
                        'T_tot': 6.000000,  # total_duration_of_the_simulated_signal_[seconds]
                        'h_bnd': [0, 0],  # [-1.0, 1.0], # Range of surface height
                        'ht_bnd': [0, 0],  # [-1.0, 1.0], # Range of tx height
                        'hr_bnd': [0, 0],  # [-1.0, 1.0], # Range of rx height
                        'd_bnd': [0, 0],  # [-10.0, 10.0], # Range of channel distance
                        'sig_h': 0.0,  # 1.000000, #L_S_standard_deviation_of_surface_height_[m]
                        'sig_ht': 0.0,  # 1.000000, # L_S_standard_deviation_of_transmitter_height_[m]
                        'sig_hr': 0.0,  # 1.000000, # L_S_standard_deviation_of_receiver_height_[m]
                        'sig_d': 0.0,  # 1, # L_S_standard_deviation_of_receiver_height_[m]
                        'a_AR': 0.900000,  # AR_parameter_for_generating_L_S_variations
                        }

        channel_info['fmax'] = channel_info['fmin'] + channel_info['B']
        channel_info['N_LS'] = int(np.round(channel_info['T_tot'] / channel_info['T_SS']))

        dopp_params = [0.197777, 0.797882, 0.100000, 5.738910, 0.318765, 0.000000, 0.000000, 0.000000, 0.050000,
                       0.010000]

        self.params = {'ci': channel_info,
                       'Dopp_params': dopp_params,
                       'n_sensors': 10,  # Number of sensors
                       'radius_r': 1000,  # To obtain sensor position
                       'n_traj': 150,  # Number of trajectories
                       'ppt': 40,  # Number of points per trajectory
                       'm': 0.7,  # Modulation index
                       'T': 20,  # Sampling period
                       }
        self.params['w0'] = np.pi / self.params['T']  # Rotation pulsation

    def obtain_h(self):
        # Obtain sensor positions
        x_r = self.params['radius_r'] * np.cos(2 * np.pi / self.params['n_sensors'] *
                                               np.linspace(0, self.params['n_sensors'], self.params['n_sensors'] + 1))
        y_r = self.params['radius_r'] / 2 * np.sin(2 * np.pi / self.params['n_sensors'] *
                                                   np.linspace(0, self.params['n_sensors'],
                                                               self.params['n_sensors'] + 1))
        r_posicion = np.zeros([3, self.params['n_sensors']])  # Coordinates, sensor
        r_posicion[0, :] = 0 + x_r[0: self.params['n_sensors']]
        r_posicion[1, :] = 0 + y_r[0: self.params['n_sensors']]
        r_posicion[2, :] = self.params['ci']['hr0']

        # Obtain trajectories
        radio_t = 100 + 1000 * np.random.rand(self.params['n_traj'])
        fase0 = 2 * np.pi * np.random.rand(self.params['n_traj'])
        omega0 = 2 * np.pi * np.random.rand(self.params['n_traj'])

        traj = np.zeros([3, self.params['n_traj'], self.params['ppt'] + 1])  # To store coord, trajectories
        aux_1 = np.linspace(0, self.params['ppt'], self.params['ppt'] + 1)
        aux_2 = np.linspace(self.params['ppt'], 2 * self.params['ppt'], self.params['ppt'] + 1)
        for it in range(self.params['n_traj']):  # For each trajectory
            traj[0, it, :] = radio_t[it] / self.params['ppt'] * aux_2 * np.cos(
                omega0[it] / self.params['ppt'] * aux_1 + fase0[it])
            traj[1, it, :] = radio_t[it] / self.params['ppt'] * aux_2 * np.sin(
                omega0[it] / self.params['ppt'] * aux_1 + fase0[it])
            traj[2, it, :] = 0  # self.params['ci']['ht0']

        self.traj = traj  # Save trajectories!

        # Process values for each pair (sensor, trajectory)
        def process_sensor(ise):
            import scipy.signal as signal
            Lf = len(range_m(self.params['ci']['fmin'], self.params['ci']['fmax'], self.params['ci']['df']))
            h_val = np.zeros([self.params['n_traj'], self.params['ppt'], Lf],
                             dtype=np.complex_)  # Sensor, point of traj
            for itx in range(self.params['n_traj']):
                print('Obtaining traj ', itx, ' of ', self.params['n_traj'],
                      ' for sensor ', ise, ' of ', self.params['n_sensors'])
                for ptx in range(self.params['ppt']):
                    # itx: index of trajectory, ptx: index of point, ise: index of sensor
                    d0 = np.linalg.norm(r_posicion[:, ise] - traj[:, itx, ptx])
                    # Obtain frequency parameters
                    f_vec = range_m(self.params['ci']['fmin'], self.params['ci']['fmax'], self.params['ci']['df'])
                    Lf = len(f_vec)
                    fc = (self.params['ci']['fmin'] + self.params['ci']['fmax']) / 2
                    f0 = self.params['ci']['fmin']
                    # Obtain rest of channel parameters
                    t_vec = range_m(0, self.params['ci']['T_SS'], self.params['ci']['dt'])
                    Lt = len(t_vec)
                    t_tot_vec = range_m(0, self.params['ci']['T_tot'], self.params['ci']['dt'])
                    Lt_tot = len(t_tot_vec)
                    # Obtain doppler parameters
                    Dopp_params = np.reshape(self.params['Dopp_params'], [Lt_tot, 10])
                    # Doppler drift
                    vtd_tot = Dopp_params[:, 0]
                    theta_td_tot = Dopp_params[:, 1]
                    vrd_tot = Dopp_params[:, 2]
                    theta_rd_tot = Dopp_params[:, 3]
                    # vehicular
                    v_t0 = (traj[:, itx, ptx + 1] - traj[:, itx, ptx]) / self.params['ci']['T_tot']
                    v_t0_proy = np.sum(v_t0 * (r_posicion[:, ise] - traj[:, itx, ptx])) / d0
                    vtv_tot = v_t0_proy + Dopp_params[:, 4]
                    theta_tv_tot = Dopp_params[:, 5]
                    vrv_tot = Dopp_params[:, 6]
                    theta_rv_tot = Dopp_params[:, 7]
                    # surface
                    Aw_tot = Dopp_params[:, 8]
                    fw_tot = Dopp_params[:, 9]

                    # Large-scale loop
                    H_LS = np.zeros([Lf, int(Lt * self.params['ci']['N_LS'])], dtype=np.complex_)
                    del_h = 0
                    del_ht = 0
                    del_hr = 0
                    del_d = 0
                    h = self.params['ci']['h0']
                    ht = self.params['ci']['ht0']
                    hr = self.params['ci']['hr0']
                    d = d0
                    adopp0 = np.zeros(50)

                    for LScount in range(self.params['ci']['N_LS']):
                        rndvec = np.random.randn(4)
                        del_h = self.params['ci']['a_AR'] * del_h \
                                + np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_h'] * rndvec[0]
                        if del_h > self.params['ci']['h_bnd'][1] or del_h < self.params['ci']['h_bnd'][0]:
                            del_h = del_h \
                                    - 2 * np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_h'] * \
                                    rndvec[
                                        0]
                        htemp = h
                        h = self.params['ci']['h0'] + del_h

                        del_ht = self.params['ci']['a_AR'] * del_ht \
                                 + np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_ht'] * rndvec[1]
                        if del_ht > self.params['ci']['ht_bnd'][1] or del_ht < self.params['ci']['ht_bnd'][0]:
                            del_ht = del_ht \
                                     - 2 * np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_ht'] * \
                                     rndvec[1]

                        httemp = ht
                        ht = self.params['ci']['ht0'] + del_ht

                        del_hr = self.params['ci']['a_AR'] * del_hr \
                                 + np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_hr'] * rndvec[2]
                        if del_hr > self.params['ci']['hr_bnd'][1] or del_hr < self.params['ci']['hr_bnd'][0]:
                            del_hr = del_hr \
                                     - 2 * np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_hr'] * \
                                     rndvec[2]

                        hrtemp = hr
                        hr = self.params['ci']['hr0'] + del_hr

                        del_d = self.params['ci']['a_AR'] * del_d \
                                + np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_d'] * rndvec[3]
                        if del_d > self.params['ci']['d_bnd'][1] or del_d < self.params['ci']['d_bnd'][0]:
                            del_d = del_d \
                                    - 2 * np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_d'] * \
                                    rndvec[
                                        3]

                        dtemp = d
                        # d = self.params['ci']['d0'] + del_d
                        d += del_d

                        def absorption(f):
                            alpha = 0.11 * np.power(f, 2) / (1 + np.power(f, 2)) \
                                    + 44 * np.power(f, 2) / (4100 + np.power(f, 2)) \
                                    + 2.75 * 10 ** (-4) * np.power(f, 2) \
                                    + 0.003
                            indvlf = f < 0.3
                            alphas = 2 * 10 ** (-3)
                            alpha[indvlf] = alphas + 0.11 * np.power(f[indvlf], 2) / (1 + np.power(f[indvlf], 2)) \
                                            + 0.011 * np.power(f[indvlf], 2)
                            return alpha

                        def reflcoeff(theta, c1, c2):
                            rho1 = 1000  # in kg / m3
                            rho2 = 1800  # in kg / m3
                            x1 = rho2 / c1 * np.sin(theta)
                            x2 = rho1 / c2 * np.sqrt(1 - (c2 / c1) ** 2 * np.cos(theta) ** 2)
                            thetac = np.real(np.arccos(c1 / c2 + 0 * 1j))  # for theta below critical, total reflection
                            if theta < thetac:
                                if thetac == 0:
                                    refl = -1
                                else:
                                    refl = np.exp(1j * np.pi * (1 - theta / thetac))
                            if theta >= thetac:
                                refl = (x1 - x2) / (x1 + x2)

                            return refl

                        def mpgeometry(h, ht, hr, d, f, k, cut, c, c2):
                            f = np.array([f]).astype(
                                float)  # This is to avoid mistakes due to no.power(int, float) = int
                            a = np.power(10.0, absorption(f / 1000) / 10)
                            a = np.power(a, 1 / 1000)
                            nr = 0  # Direct path, no reflections

                            theta = np.array([np.arctan((ht - hr) / d)])
                            l = np.array([np.sqrt((ht - hr) ** 2 + d ** 2)])
                            dell = np.array([l[0] / c])
                            A = np.array([l[0] ** k * np.power(a, l[0])])
                            ns = np.array([0])
                            nb = np.array([0])
                            G = np.array([1 / np.sqrt(A[0])])
                            Gamma = np.array([1])
                            hp = np.array([1])
                            path = np.array([0], dtype=int)
                            tau = np.array([0])

                            while min(abs(G)) >= G[0] / cut:
                                nr = nr + 1
                                for case in range(2):
                                    if case == 0:
                                        p = 2 * nr - 1
                                    else:
                                        p = 2 * nr
                                        path = np.logical_not(path).astype(int)

                                    first = path[0]
                                    last = path[-1]
                                    nb = np.append(nb, np.sum(path))
                                    ns = np.append(ns, nr - nb[p])
                                    heff = (1 - first) * ht + first * (h - ht) + (nr - 1) * h + (
                                                1 - last) * hr + last * (
                                                   h - hr)
                                    l = np.append(l, np.sqrt(heff ** 2 + d ** 2))
                                    theta = np.append(theta, np.arctan(heff / d))
                                    if first == 1:
                                        theta[p] = - theta[p]
                                    dell = np.append(dell, l[p] / c)
                                    tau = np.append(tau, dell[p] - dell[0])
                                    A = np.append(A, (l[p] ** k) * (np.power(a, l[p])))
                                    Gamma = np.append(Gamma,
                                                      reflcoeff(np.abs(theta[p]), c, c2) ** nb[p] * (-1) ** ns[p])
                                    G = np.append(G, Gamma[p] / np.sqrt(A[p]))
                                    hp = np.append(hp, Gamma[p] / np.sqrt(((l[p] / l[0]) ** k) * (a ** (l[p] - l[0]))))

                                path = np.append(path, np.logical_not(path[-1]).astype(int))

                            return l, tau, Gamma, theta, ns, nb, hp

                        # FIND LARGE SCALE MODEL PARAMETERS
                        lmean, taumean, Gamma, theta, ns, nb, hp = mpgeometry(h, h - ht, h - hr, d, fc,
                                                                              self.params['ci']['k'],
                                                                              self.params['ci']['cut'],
                                                                              self.params['ci']['c'],
                                                                              self.params['ci']['c2'])
                        # ignore paths with delays longer than allowed by frequency resolution:
                        lmean = lmean[taumean < 1 / self.params['ci']['df']]
                        theta = theta[taumean < 1 / self.params['ci']['df']]
                        ns = ns[taumean < 1 / self.params['ci']['df']]
                        nb = nb[taumean < 1 / self.params['ci']['df']]
                        hp = hp[taumean < 1 / self.params['ci']['df']]
                        taumean = taumean[taumean < 1 / self.params['ci']['df']]
                        P = len(lmean)  # Number of paths
                        # Reference path transfer function
                        H0 = 1 / np.sqrt(np.power(lmean[0], self.params['ci']['k'])
                                         * np.power(np.power(10.0, absorption(f_vec / 1000) / 10000), lmean[0]))
                        H = hp[0] * np.tile(np.exp(-1j * 2 * np.pi * f_vec * taumean[0]), [1, Lt])

                        # Find doppler rates:
                        sig_delp = np.sqrt(1 / self.params['ci']['c'] ** 2 * np.power(2 * np.sin(theta), 2)
                                           * (ns * self.params['ci']['sig2s'] + nb * self.params['ci']['sig2b']))
                        # drifting:
                        vtd = vtd_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        theta_td = theta_td_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        vrd = vrd_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        theta_rd = theta_rd_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]

                        # vehicular:
                        vtv = vtv_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        theta_tv = theta_tv_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        vrv = vrv_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        theta_rv = theta_rv_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]

                        # surface:
                        Aw = Aw_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        fw = fw_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        vw = 2 * np.pi * fw * Aw

                        # First path doppler
                        vdrift = vtd * np.cos(theta[0] - theta_td) - vrd * np.cos(theta[0] + theta_rd)
                        adrift = vdrift / self.params['ci']['c']
                        vvhcl = 0
                        avhcl = vvhcl / self.params['ci']['c']
                        vsurf = 0
                        asurf = vsurf / self.params['ci']['c']

                        adopp = adrift + avhcl + asurf * ns[0]
                        eff_adopp = adopp0[0] + np.cumsum(adopp)
                        Dopp = np.exp(1j * 2 * np.pi * f_vec * (eff_adopp * self.params['ci']['dt']))
                        adopp0[0] = eff_adopp[-1]
                        H = H * Dopp

                        # small - scale simulation: Direct method
                        for p in range(1, P):
                            gamma = np.zeros([Lf, Lt])
                            for counti in range(self.params['ci']['Sp']):
                                gamma_pi = self.params['ci']['mu_p'] + self.params['ci']['nu_p'] * np.random.randn(Lt)
                                gamma_pi = np.tile(gamma_pi, [Lf, 1]) * self.params['ci']['Sp']
                                deltau_pi = np.zeros([Lf, Lt])
                                w_delpi = sig_delp[p] \
                                          * np.sqrt(1 - np.exp(-1 * np.pi * self.params['ci']['B_delp']
                                                               * self.params['ci']['dt']) ** 2) * np.random.randn(
                                    2 * Lt)

                                temp_deltau_pi = signal.lfilter([1], [1, -np.exp(-np.pi * self.params['ci']['B_delp']
                                                                                 * self.params['ci']['dt'])], w_delpi)
                                for countf in range(Lf):
                                    deltau_pi[countf, :] = temp_deltau_pi[Lt:]
                                gamma = gamma + gamma_pi * np.exp(
                                    -1j * 2 * np.pi * np.tile(f_vec, [1, Lt]).T * deltau_pi)

                            # Doppler term:
                            vdrift = vtd * np.cos(theta[p] - theta_td) - vrd * np.cos(theta[p] + theta_rd)
                            adrift = vdrift / self.params['ci']['c']
                            vvhcl = vtv * np.cos(theta[p] - theta_tv) - vrv * np.cos(theta[p] + theta_rv) - (
                                    vtv * np.cos(theta[0] - theta_tv) - vrv * np.cos(theta[0] + theta_rv))
                            avhcl = vvhcl / self.params['ci']['c']

                            phi_pj = 2 * np.pi * np.random.rand(ns[p]) - np.pi
                            sum_j = np.zeros(Lt)
                            for jcount in range(ns[p]):
                                sum_j = sum_j + np.sin(phi_pj[jcount] + 2 * np.pi * fw * t_vec)
                            vsurf = 2 * vw * np.sin(theta[p]) * sum_j
                            asurf = vsurf / self.params['ci']['c']

                            adopp = adrift + avhcl + asurf * ns[p]
                            eff_adopp = adopp0[p] + np.cumsum(adopp)
                            Dopp = np.exp(1j * 2 * np.pi * f_vec * eff_adopp * self.params['ci']['dt'])
                            adopp0[p] = eff_adopp[-1]

                            # Multiply gamma by hp:
                            gamma = np.squeeze(gamma) * Dopp
                            H = H + hp[p] * np.tile(np.exp(-1j * 2 * np.pi * f_vec * taumean[p]), [1, Lt]) * gamma
                        H = np.tile(H0, [1, Lt]) * H
                        H_LS[:, LScount * Lt: (LScount + 1) * Lt] = H.T
                    # find channel impulse response:

                    Lt_tot = np.shape(H_LS)[1]
                    hmat = np.zeros([Lf, Lt_tot], dtype=np.complex_)
                    for countt in range(Lt_tot):
                        hmat[:, countt] = np.fft.ifft(H_LS[:, countt])

                    h_val[itx, ptx, :] = np.squeeze(hmat)
            return h_val

        out = Parallel(n_jobs=self.nop, verbose=0) \
            (delayed(process_sensor)(ise=ise) for ise in range(self.params['n_sensors']))

        return np.array(out).T  # tau x t x traj x sensor

    def filter(self, n, snr=0, nt=10, multiprocessing=True, specific=None, signal_type='sinusoid', rep=1):
        # n: number of samples to obtain and filter
        # snr: signal-to-noise ratio
        # nt: number of filtered trajectories to obtain
        print('Filtering...')
        # Obtain trajs
        if specific is None:
            if nt > self.params['n_traj']:
                trjs = np.random.choice(self.params['n_traj'], nt, replace=True).tolist()
            else:
                trjs = np.random.choice(self.params['n_traj'], nt, replace=False).tolist()
        else:
            trjs = specific
        assert nt == len(trjs)
        trjs = np.repeat(trjs, rep)

        def process_sensor(ise):

            # Causalize h[n]
            h = np.roll(self.h, 50, axis=0)

            s = np.zeros([n, self.params['ppt']])  # To store noise (cavitation)
            x = np.zeros([n, self.params['ppt']])  # To store signal

            def conv(u, v):  # Matlab 'same' convolution method differs from numpy's
                npad = len(v) - 1
                u_padded = np.pad(u, (npad // 2, npad - npad // 2), mode='constant')
                return np.convolve(u_padded, v, 'valid')

            for ipt in range(self.params['ppt']):
                if signal_type == 'sinusoid':
                    x[:, ipt] = np.real(np.exp(1j * self.params['w0'] * np.arange(n)))
                elif signal_type == 'sinusoid_cav':
                    s[:, ipt] = conv(np.exp(-10 * np.arange(n) / n), np.random.randn(2 * n - 1))
                    x[:, ipt] = (1 + self.params['m'] * np.cos(self.params['w0'] * np.arange(n))) * s[:, ipt]
                    x[:, ipt] = x[:, ipt] / np.sqrt(np.sum(np.power(x[:, ipt], 2)))  # Normalize signal power
                    # print(np.sum(np.power(x[:, ipt], 2)))
                else:
                    raise RuntimeError('Signal type not recognized')

            y_o = np.zeros([h.shape[0], h.shape[1], len(trjs)])
            i = 0
            for itx in trjs:
                if n < 2 * h.shape[0]:
                    raise RuntimeError('Too low number of samples')
                y = np.zeros([h.shape[0], self.params['ppt']])
                n_aux = np.random.randn(h.shape[0], self.params['ppt'])
                for ptx in range(self.params['ppt']):
                    signal = np.real(np.convolve(x[:, ptx], h[:, ptx, itx, ise], 'valid'))[0: h.shape[0]]
                    # Add noise
                    noise = n_aux[:, ptx]
                    signal_power = np.sum(np.power(signal, 2))
                    noise_power = np.sum(np.power(noise, 2))
                    noise = noise * np.sqrt(np.power(10, - snr / 10) * signal_power / noise_power)
                    y[:, ptx] = signal + noise
                y_o[:, :, i] = y[:, :]
                i += 1
            return y_o

        if multiprocessing:
            out = Parallel(n_jobs=self.nop, verbose=0) \
                (delayed(process_sensor)(ise=ise) for ise in range(self.params['n_sensors']))
        else:
            out = [process_sensor(ise) for ise in range(self.params['n_sensors'])]

        y_out = np.zeros([self.h.shape[0], self.h.shape[1], len(trjs), self.h.shape[3]])
        for ise in range(self.params['n_sensors']):
            y_out[:, :, :, ise] = out[ise]
        return y_out, self.traj[:, trjs, 0: self.params['ppt']]


def generate_batch_of_trajs(channel, signal_type, n=1024, snr=0, rep=10):
    # Generate batch of trajectories
    data, trjs = channel.filter(n, snr=snr, nt=channel.params['n_traj'], signal_type=signal_type, rep=rep)
    return data, trjs


def generate_batch_of_features(channel, data, trjs, feature_type, val_split=0.1, normalize=True, seq_length=1, nop=1):
    # Precompute what is needed
    if feature_type == 'power':
        n_features = 1  # Only use power as feature
        input_dim = n_features * channel.params['n_sensors']  # Input dimensionality
    elif feature_type == 'fft':  # FFT of pressure
        n_features = channel.h.shape[0] * 2  # Complex values!!
        input_dim = n_features * channel.params['n_sensors']  # Input dimensionality
    elif feature_type == 'cov':  # Use the covariance matrix
        n_features = channel.params['n_sensors'] * (channel.params['n_sensors'] + 1)
        input_dim = n_features
    else:
        raise NotImplementedError
    output_dim = 3 * seq_length

    # Initialize the arrays
    n_of_trajs = trjs.shape[1]
    '''
    if seq_length == 1:
        batch_size = int(channel.params['ppt'] * n_of_trajs)
        batch_size_train = int(batch_size * (1 - val_split))
        batch_size_val = int(batch_size - batch_size_train)
        input_train = np.zeros([batch_size_train, input_dim])
        output_train = np.zeros([batch_size_train, output_dim])
        input_val = np.zeros([batch_size_val, input_dim])
        output_val = np.zeros([batch_size_val, output_dim])

        # Obtain the features
        i=0
        index_of_tr = np.zeros(batch_size, dtype=np.int)  # To store trajectory indexes for plotting after
        for itx, ipt in itertools.product(range(n_of_trajs), range(channel.params['ppt'])):
            if i < batch_size_train:
                input_train[i, :] = adapt_data_for_batch(data[:, ipt, itx, :], feature_type, n_features,
                                                         channel.params['n_sensors'])
                output_train[i, :] = trjs[:, itx, ipt].flatten()
                index_of_tr[i] = itx
            elif i < batch_size_train + batch_size_val:
                input_val[i - batch_size_train, :] = adapt_data_for_batch(data[:, ipt, itx, :], feature_type, n_features,
                                                                          channel.params['n_sensors'])
                output_val[i - batch_size_train, :] = trjs[:, itx, ipt].flatten()
                index_of_tr[i] = itx
            else:
                pass  # Trajectories generated but not used for training!
            i += 1
    else:
    '''
    batch_size = int((channel.params['ppt'] - seq_length + 1) * n_of_trajs)
    batch_size_train = int(batch_size * (1 - val_split))
    batch_size_val = int(batch_size - batch_size_train)
    input_train = np.zeros([batch_size_train, seq_length, input_dim])
    output_train = np.zeros([batch_size_train, output_dim])
    input_val = np.zeros([batch_size_val, seq_length,input_dim])
    output_val = np.zeros([batch_size_val, output_dim])

    # Obtain the features
    i = 0
    index_of_tr = np.zeros(batch_size, dtype=np.int)  # To store trajectory indexes for plotting after

    def obt_feat(itx):
        features_traj = [adapt_data_for_batch(data[:, ipt, itx, :], feature_type, n_features,
                                              channel.params['n_sensors'])
                         for ipt in range(channel.params['ppt'])]
        return [features_traj, itx]
    # Parallelize for speed
    out = Parallel(n_jobs=nop, verbose=0) (delayed(obt_feat)(itx=itx) for itx in range(n_of_trajs))

    for val in out:
        features_traj = val[0]
        itx = val[1]
        for ipt in range(channel.params['ppt'] - seq_length + 1):
            if i < batch_size_train:
                input_train[i, :, :] = np.array([features_traj[ipt: ipt + seq_length]])
                output_train[i, :] = trjs[:, itx, ipt: ipt + seq_length].flatten()
                index_of_tr[i] = itx
            elif i < batch_size_train + batch_size_val:
                input_val[i - batch_size_train, :] = np.array([features_traj[ipt: ipt + seq_length]])
                output_val[i - batch_size_train, :] = trjs[:, itx, ipt: ipt + seq_length].flatten()
                index_of_tr[i] = itx
            else:
                pass  # Trajectories generated but not used for training!
            i += 1
    #plt.scatter(get_dist(output_val, channel), input_val.flatten())
    #plt.show()
    if normalize:
        mean = np.mean(np.concatenate((input_train, input_val)))
        std = np.std(np.concatenate((input_train, input_val)))
        input_train = (input_train - mean) / std
        input_val = (input_val - mean) / std

        #plt.scatter(get_dist(output_val, channel), input_val.flatten())
        #plt.show()
    return input_train, output_train, input_val, output_val, index_of_tr


def adapt_data_for_batch(data_input, feature_type, n_features, ns):

    def extract_features(data):
        if feature_type == 'power':
            # power = (10 * np.log10(np.sum(np.power(data, 2))) + 150) / 15 - 6
            power = (10 * np.log10(np.sum(np.power(data, 2))))
            return power
        elif feature_type == 'fft':
            fft = np.fft.fft(data)
            out = np.concatenate((np.real(fft), np.imag(fft)))
            # out = - np.sign(out) * np.log10(np.clip(np.abs(out / 5), np.spacing(1), 1))
            out = - np.sign(out) * np.log10(np.abs(out))
            return out
        else:
            raise NotImplementedError

    if feature_type == 'power' or feature_type == 'fft':
        out = np.zeros([n_features, ns])
        for ise in range(ns):
            out[:, ise] = extract_features(data_input[:, ise])
    elif feature_type == 'cov':
        fft_val = []
        arg_max = []
        for ise in range(ns):
            fft_val.append(np.fft.fft(data_input[:, ise]))
            arg_max.append(np.argmax(np.abs(fft_val[ise])))
        data = Counter(arg_max)  # Use the freq bin with highest energy for all cases
        index = data.most_common(1)[0][0]  # Returns the highest occurring item
        p = np.array([fft_val[ise][index] for ise in range(ns)])
        p = p / np.linalg.norm(p)
        p = np.expand_dims(p, 1)
        cov = p @ p.conj().T
        out_idx = np.triu_indices_from(cov)  # Only use the upper triangular values
        out = np.concatenate((np.real(cov[out_idx]), np.imag(cov[out_idx])))
    else:
        raise NotImplementedError
    return out.flatten()


def get_dist(output, channel):
    # Obtain sensor positions
    x_r = channel.params['radius_r'] * np.cos(2 * np.pi / channel.params['n_sensors'] *
                                           np.linspace(0, channel.params['n_sensors'], channel.params['n_sensors'] + 1))
    y_r = channel.params['radius_r'] / 2 * np.sin(2 * np.pi / channel.params['n_sensors'] *
                                               np.linspace(0, channel.params['n_sensors'],
                                                           channel.params['n_sensors'] + 1))
    r_posicion = np.zeros([3, channel.params['n_sensors']])  # Coordinates, sensor
    r_posicion[0, :] = 0 + x_r[0: channel.params['n_sensors']]
    r_posicion[1, :] = 0 + y_r[0: channel.params['n_sensors']]
    r_posicion[2, :] = channel.params['ci']['hr0']
    dist = np.zeros((output.shape[0], channel.params['n_sensors']))
    for s in range(channel.params['n_sensors']):
        dist[:, s] = np.linalg.norm(output - r_posicion[:, s], axis=1)
    return dist


class channel_learner():
    def __init__(self, input_dim, output_dim, nn=128, channel=None, seq_length=1):
        self.nn = nn  # Number of neurons
        self.hist = None  # To store training history
        self.channel = channel
        if self.channel is None:  # Start default channel
            self.channel = channel(number_of_processes=-1)
        # Create model
        self.freq_len = self.channel.h.shape[0]  # Number of freq bins
        self.ns = self.channel.params['n_sensors']  # Number of sensors
        self.nt = self.channel.params['n_traj']  # Number of trajs
        self.ppt = self.channel.params['ppt']  # Points per traj
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        # Dynamic memory placement if GPU is available
        from keras import backend as K
        if len(K.tensorflow_backend._get_available_gpus()) > 0:
            from keras.backend.tensorflow_backend import set_session
            import tensorflow as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
            config.log_device_placement = True  # to log device placement (on which device the operation ran)
            # (nothing gets printed in Jupyter, only if you run it standalone)
            sess = tf.Session(config=config)
            set_session(sess)  # set this TensorFlow session as the default session for Keras

        if seq_length == 1:  # Not recurrent implementation
            self.model = self.create_model_nr()
        else:
            self.model = self.create_model_r()

    def create_model_r(self):
        model = Sequential()
        from keras import backend as K
        if len(K.tensorflow_backend._get_available_gpus()) == 0: # No GPU
            model.add(layers.LSTM(self.nn, input_shape=(self.seq_length, self.input_dim)))
        else:
            model.add(layers.CuDNNLSTM(self.nn, input_shape=(self.seq_length, self.input_dim)))
        model.add(layers.Dense(self.nn, kernel_initializer='normal', activation='tanh'))
        model.add(layers.Dense(self.output_dim, kernel_initializer='normal', activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.001), metrics=[metrics.mae])
        model.summary()
        return model

    def create_model_nr(self):
        model = Sequential()
        model.add(layers.Dense(self.nn, input_dim=self.input_dim, kernel_initializer='normal', activation='tanh'))
        model.add(layers.Dense(self.nn, kernel_initializer='normal', activation='tanh'))
        model.add(layers.Dense(self.output_dim, kernel_initializer='normal', activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.001), metrics=[metrics.mae])
        model.summary()
        return model

    def train_model(self, input_train, output_train, input_val, output_val, itr, n_iter=1, n_epochs=1, plot=None, rep=1):
        self.hist = {'train': [], 'val': [], 'train_mae': [], 'val_mae': [],
                     'keys': self.model.metrics_names, 'n_iter': n_iter}
        t_init = datetime.datetime.now()
        for i in range(n_iter):
            h = self.model.fit(input_train, output_train, batch_size=1024*8, epochs=n_epochs,
                               validation_data=(input_val, output_val), verbose=0)
            loss_train = h.history['loss']
            loss_val = h.history['val_loss']
            self.hist['train'].extend(loss_train)
            self.hist['val'].extend(loss_val)
            self.hist['train_mae'].extend(h.history['mean_absolute_error'])
            self.hist['val_mae'].extend(h.history['val_mean_absolute_error'])
            print('Training iteration ', i + 1, ' of ', n_iter,
                  '; Elapsed time: ', datetime.datetime.now() - t_init,
                  '; Estimated time left: ', (datetime.datetime.now() - t_init) * (n_iter - i + 1) / (i + 1))
        if plot is not None:
            with open(plot + '_data.pickle', 'wb') as handle:
                pickle.dump(self.hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # Obtain MAE of all trajectories used for validation
            pred_traj = self.model.predict(input_val)
            orig_traj = output_val

            # Reformat to obtain trajectories
            valid_trajs_idx = np.unique(itr[input_train.shape[0]:])[
                              1: -1]  # Do not use first nor last traj, it could be fragmentary
            mae = np.zeros(len(valid_trajs_idx))
            for i in range(len(mae)):
                mask = np.where(itr == valid_trajs_idx[i]) - np.array(input_train.shape[0])  # To obtain idx of trajs
                mask = np.squeeze(mask)
                mae[i] = np.mean(np.abs(pred_traj[mask, :] - orig_traj[mask, :]))

            # Obtain sensor positions
            x_r = self.channel.params['radius_r'] * np.cos(2 * np.pi / self.channel.params['n_sensors'] *
                                                           np.linspace(0, self.channel.params['n_sensors'],
                                                                       self.channel.params['n_sensors'] + 1))
            y_r = self.channel.params['radius_r'] / 2 * np.sin(2 * np.pi / self.channel.params['n_sensors'] *
                                                               np.linspace(0, self.channel.params['n_sensors'],
                                                                           self.channel.params['n_sensors'] + 1))
            r_posicion = np.zeros([2, self.channel.params['n_sensors']])  # Coordinates, sensor
            r_posicion[0, :] = 0 + x_r[0: self.channel.params['n_sensors']]
            r_posicion[1, :] = 0 + y_r[0: self.channel.params['n_sensors']]

            n_traj_to_plot = 10  # Max number of trajs to plot from validation

            if len(valid_trajs_idx) > n_traj_to_plot * rep:
                i_range = range(0, n_traj_to_plot, rep)
            else:
                i_range = range(0, len(valid_trajs_idx), rep)

            for i in i_range:
                #i = np.argmin(mae)
                mask = np.where(itr == valid_trajs_idx[i]) - np.array(input_train.shape[0])
                mask = np.squeeze(mask)

                # Plot everything
                plt.plot(orig_traj[mask, 0].flatten(), orig_traj[mask, self.seq_length].flatten(), 'b')
                plt.plot(pred_traj[mask, 0].flatten(), pred_traj[mask, self.seq_length].flatten(), 'r')
                plt.plot(r_posicion[0, :], r_posicion[1, :], 'k*')
                plt.title('MAE = ' + str(mae[i]))
                tikz_save('mae_' + plot + '_' + str(i) + '.tikz', figureheight='\\figureheight', figurewidth='\\figurewidth')
                plt.savefig('mae_' + plot + '_' + str(i) + '.png', bbox_inches='tight')
                plt.close()

            i = np.argmin(mae)
            plt.semilogy(self.hist['train'], 'b', label='train')
            plt.semilogy(self.hist['val'], 'r', label='val')
            plt.legend(loc='best')
            plt.title('Loss: MSE' + str(mae[i]))
            tikz_save('loss_mse_' + plot + '.tikz', figureheight='\\figureheight', figurewidth='\\figurewidth')
            plt.savefig('loss_mse_' + plot + '.png', bbox_inches='tight')
            plt.close()

            plt.semilogy(self.hist['train_mae'], 'b', label='train')
            plt.semilogy(self.hist['val_mae'], 'r', label='val')
            plt.legend(loc='best')
            plt.title('MAE' + str(mae[i]))
            tikz_save('loss_mae_' + plot + '.tikz', figureheight='\\figureheight', figurewidth='\\figurewidth')
            plt.savefig('loss_mae_' + plot + '.png', bbox_inches='tight')
            plt.close()

    def save(self, name):
        # Save NN
        self.model.save(name + '.h5')


def generate_params(options=None):
    channel_info = {'h0': 50,  # Surface height(depth)[m]
                    'ht0': 50,  # TX height [m]
                    'hr0': 1,  # RX height [m]
                    'd0': 500,  # channel distance [m]
                    'k': 1.700000,  # spreading_factor
                    'c': 1500.000000,  # speed_of_sound_in_water_[m / s]
                    'c2': 1200.000000,  # speed_of_sound_in_bottom_[m / s]
                    'cut': 50.000000,  # minimum_relative_path_strength
                    'fmin': 10000.000000,  # minimum_frequency_[Hz]
                    'B': 10000.000000,  # bandwidth_[Hz]
                    'df': 25.000000,  # frequency_resolution_[Hz]
                    'dt': 6.045000,  # time_resolution_[seconds]
                    'T_SS': 6.000000,  # coherence_time_of_the_small - scale_variations_[seconds]
                    'sig2s': 1.125000,  # variance_of_S_S_surface_variations_[m ^ 2]
                    'sig2b': 0.562500,  # variance_of_S_S_bottom_variations_[m ^ 2]
                    'B_delp': 0.000500,  # 3 - dB_width_of_the_p.s.d._of_intra - path_delays_[Hz]
                    'Sp': 20,  # number_of_intra - paths
                    'mu_p': 0.025000,  # mean_of_intra - path_amplitudes
                    'nu_p': 0.000001,  # variance_of_intra - path_amplitudes
                    'T_tot': 6.000000,  # total_duration_of_the_simulated_signal_[seconds]
                    'h_bnd': [-1.0, 1.0],  # Range of surface height
                    'ht_bnd': [-1.0, 1.0],  # Range of tx height
                    'hr_bnd': [-1.0, 1.0],  # Range of rx height
                    'd_bnd': [-10.0, 10.0],  # Range of channel distance
                    'sig_h': 1.000000,  # L_S_standard_deviation_of_surface_height_[m]
                    'sig_ht': 1.000000,  # L_S_standard_deviation_of_transmitter_height_[m]
                    'sig_hr': 1.000000,  # L_S_standard_deviation_of_receiver_height_[m]
                    'sig_d': 1.000000,  # L_S_standard_deviation_of_receiver_height_[m]
                    'a_AR': 0.900000,  # AR_parameter_for_generating_L_S_variations
                    }
    if isinstance(options, str):
        if options == 'no_var':
            aux = 0.0
    else:
        aux = options
    # Modify parameters
    channel_info['sig2s'] *= aux
    channel_info['sig2b'] *= aux
    channel_info['mu_p'] *= aux
    channel_info['nu_p'] *= aux
    channel_info['Sp'] = int(channel_info['Sp'] * aux)
    channel_info['h_bnd'] = [channel_info['h_bnd'][0] * aux, channel_info['h_bnd'][1] * aux]
    channel_info['ht_bnd'] = [channel_info['ht_bnd'][0] * aux, channel_info['ht_bnd'][1] * aux]
    channel_info['hr_bnd'] = [channel_info['ht_bnd'][0] * aux, channel_info['ht_bnd'][1] * aux]
    channel_info['d_bnd'] = [channel_info['ht_bnd'][0] * aux, channel_info['ht_bnd'][1] * aux]
    channel_info['sig_h'] *= aux
    channel_info['sig_ht'] *= aux
    channel_info['sig_hr'] *= aux
    channel_info['sig_d'] *= aux

    channel_info['fmax'] = channel_info['fmin'] + channel_info['B']
    channel_info['N_LS'] = int(np.round(channel_info['T_tot'] / channel_info['T_SS']))

    dopp_params = [0.197777, 0.797882, 0.100000, 5.738910, 0.318765, 0.000000, 0.000000, 0.000000, 0.050000, 0.010000]

    params = {'ci': channel_info,
              'Dopp_params': dopp_params,
              'n_sensors': 10,  # Number of sensors
              'radius_r': 1000,  # To obtain sensor position
              'n_traj': 100,  # Number of trajectories
              'ppt': 50,  # Number of points per trajectory
              'm': 0.7,  # Modulation index
              'T': 20,  # Sampling period
              }
    params['w0'] = np.pi / params['T']  # Rotation pulsation
    return params


def process(channel_option, snr, feat_types, nn_vals, number_of_seeds, rep, seq_length, n_epochs, nop=30):

    # First, check if files exist, otherwise, process
    pr = False
    for feat in feat_types:
        for nn in nn_vals:
            for seed in range(number_of_seeds):
                name = str(channel_option) + '_' + str(snr) + '_' + feat + '_' + str(nn) + '_' + str(seed)
                if not os.path.isfile(name + '_loc_model.h5'):
                    pr = True  # Process
    if pr:
        params = generate_params(options=channel_option)
        c = channel(load=True, params=params, number_of_processes=nop, name=str(channel_option))
        # Generate batch of trajs
        data, trjs = generate_batch_of_trajs(c, 'sinusoid', n=1024, snr=snr, rep=rep)
        for feat in feat_types:
            # Check if files exist, otherwise, compute them
            pr_feat = False
            for nn in nn_vals:
                for seed in range(number_of_seeds):
                    name = str(channel_option) + '_' + str(snr) + '_' + feat + '_' + str(nn) + '_' + str(seed)
                    if not os.path.isfile(name + '_loc_model.h5'):
                        pr_feat = True  # Process
            if pr_feat:
                gc.collect()
                input_train = None
                output_train = None
                input_val = None
                output_val = None
                # Generate batch of features
                input_train, output_train, input_val, output_val, index_of_tr = \
                    generate_batch_of_features(c, data, trjs, feat, val_split=0.1, normalize=True,
                                               seq_length=seq_length, nop=nop)
                for nn in nn_vals:
                    for seed in range(number_of_seeds):
                        name = str(channel_option) + '_' + str(snr) + '_' + feat + '_' + str(
                            nn) + '_' + str(seed)
                        # Check if value exists, otherwise, proceed
                        if os.path.isfile(name + '_loc_model.h5'):
                            print('File ', name, ' already exists')
                        else:
                            print('Processing ', name)
                            trainer = channel_learner(input_dim=input_train.shape[-1],
                                                      output_dim=output_train.shape[-1],
                                                      nn=nn, channel=c, seq_length=seq_length)
                            trainer.train_model(input_train, output_train, input_val, output_val, index_of_tr,
                                                n_epochs=n_epochs, plot=name, rep=rep)
                            trainer.save(name + '_loc_model')
                            print('Processing of  ', name, ' finished')
                            gc.collect()
        # Delete data
        c = None
        data = None
        trjs = None
        input_train = None
        output_train = None
        input_val = None
        output_val = None
        gc.collect()
    else:
        print('Files already processed for channel ', channel_option, ' and snr = ', snr)


if __name__ == '__main__':

    with open('aux.pickle', 'rb') as handle:
        data = pickle.load(handle)
    channel_option = data['channel_option']
    snr = data['snr']
    feat_types = data['feat_types']
    nn_vals = data['nn_vals']
    number_of_seeds = data['number_of_seeds']
    rep = data['rep']
    seq_length=data['seq_length']
    n_epochs = data['n_epochs']

    process(channel_option, snr, feat_types, nn_vals, number_of_seeds, rep, seq_length, n_epochs)