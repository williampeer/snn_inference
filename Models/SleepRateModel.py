#!/usr/bin/python

"""
Adapted to Pytorch from https://github.com/SakataLab/Sleep_Computational_Model_2019/blob/master/sleepCompModel.py:
Computational model core Herice C. and Sakata S. (2019)
Sleep/wake regulation model and synapses alterations.
Original model from Costa et al. (2016) and Diniz Behn et al. (2012).
Charlotte HERICE - January 2019
"""

import torch
from torch import nn
from torch import tensor


################################################
# Sleep/wake regulation
################################################

class SleepRegulationModel(nn.Module):
    def __init__(self, g_RRe, g_RWe, g_WNi, g_WRi, g_NRi, g_NWi):
        """
            totalSimDuration: table with all time step values
            res: simulation resolution time
            g_RRe: weight for synapse from REM to REM
            g_RWe: weight for synapse from REM to Wake
            g_WNi: weight for synapse from Wake to NREM
            g_WRi: weight for synapse from Wake to REM
            g_NRi: weight for synapse from NREM to REM
            g_NWi: weight for synapse from NREM to Wake
        """
        super(SleepRegulationModel, self).__init__()

        print("init_Reg")
        # self.totalSimDuration = totalSimDuration
        # self.res = res
        # self.g_RRe = g_RRe
        # self.g_RWe = g_RWe
        # self.g_WNi = g_WNi
        # self.g_WRi = g_WRi
        # self.g_NRi = g_NRi
        # self.g_NWi = g_NWi
        # self.res = nn.Parameter(tensor(res), requires_grad=True)
        self.g_RRe = nn.Parameter(tensor(g_RRe), requires_grad=True)
        self.g_RWe = nn.Parameter(tensor(g_RWe), requires_grad=True)
        self.g_WNi = nn.Parameter(tensor(g_WNi), requires_grad=True)
        self.g_WRi = nn.Parameter(tensor(g_WRi), requires_grad=True)
        self.g_NRi = nn.Parameter(tensor(g_NRi), requires_grad=True)
        self.g_NWi = nn.Parameter(tensor(g_NWi), requires_grad=True)
        self.g_NWi = nn.Parameter(tensor(g_NWi), requires_grad=True)

        self.setParams()

    def heaviside(self, X):
        """
        Heaviside step function
            X: variable to evaluate
        """
        # return (X >= 0).float()  # Not differentiable.
        steepness_coefficient = torch.tensor(5.0)
        return torch.sigmoid(steepness_coefficient * X)  # sigmoidal to make differentiable.

    def set_RK(self, RK_N, noise_input):
        """
        ODE functions - SRK iterations
            N: RK moment
            dt: time step
            noiseValues: table with all noise values for every time step
            i: time step
        """

        # Populations firing rates
        self.f_W[RK_N + 1] = self.f_W[0] + self.A[RK_N] * (
                self.F_W_max * 0.5 * (1 + torch.tanh((self.I_W(RK_N, noise_input) - self.beta_W) / self.alpha_W)) -
                self.f_W[RK_N]) / self.tau_W
        self.f_N[RK_N + 1] = self.f_N[0] + self.A[RK_N] * (self.F_N_max * 0.5 * (
                    1 + torch.tanh((self.I_N(RK_N, noise_input) + self.kappa * self.h[RK_N]) / self.alpha_N)) - self.f_N[RK_N]) / self.tau_N
        self.f_R[RK_N + 1] = self.f_R[0] + self.A[RK_N] * (
                self.F_R_max * 0.5 * (1 + torch.tanh((self.I_R(RK_N, noise_input) - self.beta_R) / self.alpha_R)) -
                self.f_R[RK_N]) / self.tau_R

        # Neurotransmitter concentrations
        self.c_WXi[RK_N + 1] = self.c_WXi[0] + self.A[RK_N] * (torch.tanh(self.f_W[RK_N + 1] / self.gamma_E) - self.c_WXi[RK_N]) / self.tau_E
        self.c_NXi[RK_N + 1] = self.c_NXi[0] + self.A[RK_N] * (torch.tanh(self.f_N[RK_N + 1] / self.gamma_G) - self.c_NXi[RK_N]) / self.tau_G
        self.c_RXe[RK_N + 1] = self.c_RXe[0] + self.A[RK_N] * (torch.tanh(self.f_R[RK_N + 1] / self.gamma_A) - self.c_RXe[RK_N]) / self.tau_A

        # Homeostatic force
        self.h[RK_N + 1] = self.h[0] + self.A[RK_N] * \
                           ((self.H_max - self.h[RK_N]) / self.tau_hw * self.heaviside(self.f_W[RK_N] - self.theta_W) -
                            self.h[RK_N] / self.tau_hs * self.heaviside(self.theta_W - self.f_W[RK_N]))

    def add_RK2(self, var):
        """
        RK2 iteration
            var: list of the RK moments to iterate
        """
        var[0] = (-3 * var[0] + 2 * var[1] + 4 * var[2] + 2 * var[3] + var[4]) / 6

    def add_RK1(self):
        """
        RK1 iteration
        """
        self.add_RK2(self.f_W)
        self.add_RK2(self.f_N)
        self.add_RK2(self.f_R)
        self.add_RK2(self.c_WXi)
        self.add_RK2(self.c_NXi)
        self.add_RK2(self.c_RXe)
        self.add_RK2(self.h)

    def I_W(self, RK_N, noise_input):
        """
        Wake input function
            N: RK moment
            noiseValues: table with all noise values for every time step
            i: time step
        """
        return self.g_NWi * self.c_NXi[RK_N] + self.g_RWe * self.c_RXe[RK_N] + noise_input  # + self.getWhiteGaussianNoise(noiseValues, i)

    def I_N(self, RK_N, noise_input):
        """
        NREM input function
            N: RK moment
            noiseValues: table with all noise values for every time step
            i: time step
        """
        return self.g_WNi * self.c_WXi[RK_N] + noise_input  # + self.getWhiteGaussianNoise(noiseValues, i)

    def I_R(self, RK_N, noise_input):
        """
        REM input function
            N: RK moment
            noiseValues: table with all noise values for every time step
            i: time step
        """
        return self.g_WRi * self.c_WXi[RK_N] + self.g_NRi * self.c_NXi[RK_N] + self.g_RRe * self.c_RXe[RK_N] + noise_input  # + self.getWhiteGaussianNoise(noiseValues, i)

    def initVal(self, value):
        """
        Initialise paramters
            value: initial value for the parameter
        """
        return [value, 0.0, 0.0, 0.0, 0.0]

    def setParams(self):
        """
        Setting initial parameters values for the model
        """
        # Membrane time in [s]
        self.tau_W = 1500e3
        self.tau_N = 600e3
        self.tau_R = 60e3

        # Neurotransmitter time constants in [s]
        self.tau_E = 25e3
        self.tau_G = 10e3
        self.tau_A = 10e3

        # Maximum firing rate in [Hz]
        self.F_W_max = 6.5
        self.F_N_max = 5.
        self.F_R_max = 5.

        # Sigmoid slope parameters in [aU]
        self.alpha_W = 0.5
        self.alpha_N = 0.175
        self.alpha_R = 0.13

        # Sigmoid threshold parameters in [aU]
        self.beta_W = -0.4
        self.beta_R = -0.9

        # Neurotransmitter release scaling in [s^-1]
        self.gamma_E = 5.
        self.gamma_G = 4.
        self.gamma_A = 2.

        # # Synaptic weights for neurotransmitter efficacy in [aU]
        # self.g_RRe = g_RRe 			# REM to REM
        # self.g_RWe = g_RWe			# REM to Wake
        # self.g_WNi = g_WNi 			# Wake to NREM
        # self.g_WRi = g_WRi 			# Wake to REM
        # self.g_NRi = g_NRi 			# NREM to REM
        # self.g_NWi = g_NWi 			# NREM to Wake

        # Sleep Homeostasis parameter
        self.H_max = 1.  # in [aU]
        self.theta_W = 2.  # in [s]
        self.tau_hw = 34830e3  # 580.5 min in [s]
        self.tau_hs = 30600e3  # 510 min in [s]
        self.kappa = 1.5  # in [aU]

        # SRK integration parameters
        self.A = [0.5, 0.5, 1.0, 1.0]

        # Wake-population input pulses parameters
        self.etaCpt = 0
        self.eta = 0

        # Declaration and initialization of variables
        self.f_W = self.initVal(6.)  # Wake promoting activity	in [Hz]
        self.f_N = self.initVal(1e-3)  # Sleep promoting activity	in [Hz]
        self.f_R = self.initVal(1e-3)  # REM promoting activity	in [Hz]
        self.c_WXi = self.initVal(0.9)  # Norephrine concentration	in [aU]
        self.c_NXi = self.initVal(1e-3)  # GABA concentration		in [aU]
        self.c_RXe = self.initVal(1e-3)  # Acetylcholin concentration in [aU]
        self.h = self.initVal(0.5)  # Homeostatic sleep drive	in [aU]

    # =================== from RunSim ======================
    def forward(self, noise_input):  #, noiseValues, i):
        """
        First calculate every ith RK moment.
        Has to be in order, 1th moment first
            noiseValues: table with all noise values for every time step
            i: time step
        """
        for y in range(4):
            self.set_RK(y, noise_input)  #, i)

        # Add all moments
        self.Sleep_Reg.add_RK1()
