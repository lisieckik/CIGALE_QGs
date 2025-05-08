"""
Stochastic star formation history model based on regulation model
==========================================================================

This module implements a star formation history (SFH) described as a stochastic process
regulated by 5 parameters (sigmaReg, tauEq, tauIn, sigmaDyn, tauDyn, Wan+24, MNRAS;
Iyer+24, ApJ).

"""

import numpy as np
from pcigale.sed_modules import SedModule
import os



__category__ = "SFH"

def get_tarr(ageMax, n_tarr = 8):
    edges = ageMax - np.append(np.linspace(ageMax, 30, n_tarr), [10,0]).astype(int)
    centers = []
    for i in range(len(edges) - 1):
        centers.append((edges[i + 1] - edges[i]) / 2 + edges[i])
    centers = np.array(centers)
    return centers, edges

class SFHStohastic_Regulator(SedModule):
    """Stochastic star formation history model based on regulation model.
    See Iyer+24 and Wan+24 for details.
    sigmaReg: the amount of overall variance, unitless,
    contrary to self-regulation (higher value - larger changes in the same time);
    tauEq: equilibrium timescale, Myr;
    tauIn: inflow correlation timescale (includes 2pi factor), Myr;
    sigmaDyn: giant molecular cloud dynamical variability, unitless;
    tauDyn: dynamical lifetime of GMC, Myr.


    It takes additional ~ 5 MB per 1e4 models
    """

    parameters = {
        "age_form": (
            "cigale_list(dtype=int, minvalue=0.)",
            "Look-back time, since the galaxy formed, started forming stars in Myr. The "
            "precision is 1 Myr.",
            2000.
        ),
        "nModels": (
            "cigale_list(dtype=int, minvalue=0.)",
            "Number of random SFH generated using given parameters "
            "(per each combination of parameters)."
            "One number only! (int)",
            100
        ),
        "nLevels": (
            "cigale_list(dtype=int, minvalue=0.)",
            "Number of SFR bins for SFH. age_form is divied into nLevels linear steps",
            6
        ),
        "sigmaReg": (
            "cigale_list(dtype=float, minvalue=0.)",
            "The amount of overall variance, unitless,"
            "higher values - larger changes in the same time (float).",
            1
        ),
        "tauEq": (
            "cigale_list(dtype=int, minvalue=0.)",
            "Equilibrium timescale, Myr,"
            "precision is 1 Myr.",
            500.
        ),
        "tauIn": (
            "cigale_list(dtype=int, minvalue=0.)",
            "Inflow correlation timescale, Myr,"
            "precision is 1 Myr.",
            150.
        ),
        "sigmaDyn": (
            "cigale_list(dtype=float, minvalue=0.)",
            "giant molecular cloud dynamical variability, unitless(float).",
            0.24
        ),
        "tauDyn": (
            "cigale_list(dtype=int, minvalue=0.)",
            "Dynamical lifetime of GMC, Myr,"
            "precision is 1 Myr.",
            5.
        ),
        "sfr_A": (
            "cigale_list(minvalue=0.)",
            "Multiplicative factor controlling the SFR if normalise is False. "
            "For instance without any burst/quench: SFR(t)=sfr_A×t×exp(-t/τ)/τ²",
            1.
        ),
        "normalise": (
            "boolean()",
            "Normalise the SFH to produce one solar mass.",
            True
        )
    }

    def _init_code(self):

        self.age_form = int(self.parameters["age_form"])
        self.nLevels = int(self.parameters["nLevels"])
        self.nModel = int(self.parameters["nModels"])

        self.sigmaReg = float(self.parameters["sigmaReg"])
        self.tauEq = int(self.parameters["tauEq"])
        self.tauIn = int(self.parameters["tauIn"])
        self.sigmaDyn = float(self.parameters["sigmaDyn"])
        self.tauDyn = int(self.parameters["tauDyn"])
        sfr_A = float(self.parameters["sfr_A"])


        if isinstance(self.parameters["normalise"], str):
            normalise = self.parameters["normalise"].lower() == 'true'
        else:
            normalise = bool(self.parameters["normalise"])


        ### Build a new SFH, using already caculated stohastic values. ###

        # Find and open stohastic values for this model

        try:
            sfrValues = np.load('out/SFHs/SFH_%i_%i_%.4f_%i_%i_%.4f_%i.npy' % (
                self.age_form, self.nLevels, self.sigmaReg, self.tauEq,
                self.tauIn, self.sigmaDyn, self.tauDyn))[:, self.nModel]
        except Exception as err2:
            sfrValues = np.ones([self.nLevels+1])


        if len(sfrValues) != self.nLevels + 1:
            sfrValues = np.ones([self.nLevels+1])

        # start with finding when SFR will change
        centers, edges = get_tarr(self.age_form, n_tarr=self.nLevels)
        tarr = np.arange(self.age_form)


        # Prepare SFR table
        self.sfr = np.zeros(tarr.shape)
        for i in range(len(centers)):
            self.sfr[edges[i]:edges[i + 1]] = 10**sfrValues[i]

        self.sfr_integrated = np.sum(self.sfr) * 1e6  ### Myr to Yr
        if normalise:
            self.sfr /= self.sfr_integrated
            self.sfr_integrated = 1.
        else:
            self.sfr *= sfr_A
            self.sfr_integrated *= sfr_A


    def process(self, sed):
        """
        Parameters
        ----------
        sed : pcigale.sed.SED object

        """

        sed.add_module(self.name, self.parameters)

        # Add the sfh and the output parameters to the SED.
        sed.sfh = self.sfr
        sed.add_info("sfh.integrated", self.sfr_integrated, True,
                     unit='solMass')
        sed.add_info("sfh.age_form", self.age_form, unit='Myr')
        sed.add_info("sfh.nLevels", self.nLevels)
        sed.add_info("sfh.sigmaReg", self.sigmaReg)
        sed.add_info("sfh.tauEq", self.tauEq)
        sed.add_info("sfh.tauIn", self.tauIn)
        sed.add_info("sfh.sigmaDyn", self.sigmaDyn)
        sed.add_info("sfh.tauDyn", self.tauDyn)


# CreationModule to be returned by get_module
Module = SFHStohastic_Regulator
