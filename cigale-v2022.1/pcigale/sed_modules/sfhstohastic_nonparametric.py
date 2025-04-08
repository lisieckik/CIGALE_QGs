"""Stochastic star formation history model with tstudent distribution
==========================================================================

This module implements a star formation history (SFH) described as a stochastic process
where the sfr change between steps is limited by tstudent distribution

"""

import numpy as np
from . import SedModule
import os


# def callmeforhelp():
#     result = inspect.getouterframes(inspect.currentframe(), 2)
#     result = str(result[1][1]).split('/sed_modules')[0]
#     return result

__category__ = "SFH"


class SFHStohastic_Nonparametric(SedModule):
    """Stochastic star formation history model with tstudent distribution
    It takes additional ~ 300 MB per 1e4 models
    """

    parameter_list = {
        "age_form": (
            "cigale_list(dtype=int, minvalue=0.)",
            "Look-back time, since the galaxy formed, started forming stars in Myr. The "
            "precision is 1 Myr.",
            2000.
        ),
        "nModels": (
            "cigale_list(dtype=int, minvalue=0.)",
            "Number of random SFH generated using given parameters."
            "One number only!",
            100
        ),
        "nLevels": (
            "cigale_list(dtype=int, minvalue=0.)",
            "Number of SFR bins for SFH. age_form is divied into nLevels+1 steps with log scale"
            "In Leja+19, they tested 4-14 bins with consistent results",
            6
        ),
        "lastBin": (
            "cigale_list(dtype=int, minvalue=0.)",
            "The width of the last bin (right before the observation) in Myr."
            "30 Myr allows to check for recent starburst. The "
            "precision is 1 Myr.",
            30.
        ),
        "scaleFactor": (
            "cigale_list(dtype=float, minvalue=0.)",
            "Scale factor controlls the width of the random distrivution."
            "Tachella+21b tested values:"
            "0.3 (continuity prior) and 1 (bursty continuity prior)",
            1.
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
        self.lastBin = int(self.parameters["lastBin"])
        self.nModel = int(self.parameters["nModels"])
        self.scaleFactor = float(self.parameters['scaleFactor'])
        sfr_A = float(self.parameters["sfr_A"])


        if isinstance(self.parameters["normalise"], str):
            normalise = self.parameters["normalise"].lower() == 'true'
        else:
            normalise = bool(self.parameters["normalise"])


        
        # start with finding when SFR will change
        tBin = np.logspace(np.log10(self.lastBin), np.log10(self.age_form - self.age_form/self.nLevels), self.nLevels-1).astype(int)[::-1]
        if abs(tBin[-2] - tBin[-1]) < self.lastBin:
            tBin = np.logspace(np.log10(abs(tBin[-2] - tBin[-1])/2), np.log10(self.age_form - self.age_form/self.nLevels), self.nLevels-1).astype(int)[::-1]
        tBin = np.append([self.age_form], tBin)
        tBin = np.append(tBin, [0])

        # Open the file contaning the changes (if its check/config take all zeros)
        try:
            sfrChange = np.load('out/SFHs/RandomChange/%i.npy'% (self.nLevels), allow_pickle=True)[self.nModel]
        except Exception as err2:
            sfrChange = np.zeros([self.nLevels])

        if len(sfrChange) != self.nLevels:
            sfrChange = np.zeros([self.nLevels])
        # Prepare SFR table
        self.sfr = np.zeros([self.age_form + 1]) + 1
        for change in range(len(sfrChange) - 1):
            self.sfr[tBin[change + 2]:tBin[change+1]] = self.sfr[tBin[change]] / 10 ** sfrChange[change]
        self.sfr = self.sfr[::-1]


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
        sed.add_info("sfh.lastBin", self.lastBin)
        sed.add_info("sfh.scaleFactor", self.scaleFactor)

# CreationModule to be returned by get_module
Module = SFHStohastic_Nonparametric
