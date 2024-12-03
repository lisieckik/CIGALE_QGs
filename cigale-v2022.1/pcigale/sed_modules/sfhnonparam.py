"""
Delayed tau model for star formation history with an optional burst/quench
==========================================================================

This module implements a star formation history (SFH) described as a delayed
rise of the SFR up to a maximum, followed by an exponential decrease. Optionally
a quenching or a bursting episode can be added. It is described in more detail
in Ciesla et al. (2017).

"""

import numpy as np
from . import SedModule
import os


# def callmeforhelp():
#     result = inspect.getouterframes(inspect.currentframe(), 2)
#     result = str(result[1][1]).split('/sed_modules')[0]
#     return result

__category__ = "SFH"


class SFHNonParam(SedModule):
    """Delayed tau model for Star Formation History with an optional burst or
    quench.

    This module sets the SED star formation history (SFH) proportional to time,
    with a declining exponential parametrised with a time-scale τ. Optionally
    a burst/quench can be added. In that case the SFR of that episode is
    constant and parametrised as a ratio of the SFR before the beginning of the
    episode. See Ciesla et al. (2017).

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
        sfr_A = float(self.parameters["sfr_A"])


        if isinstance(self.parameters["normalise"], str):
            normalise = self.parameters["normalise"].lower() == 'true'
        else:
            normalise = bool(self.parameters["normalise"])


        try:
            # If the SFH was calculated before, use the same one
            self.sfr = np.load('out/SFHs/SFH_%i_%i_%i_%i.npy'%(self.nModel, self.nLevels, self.lastBin,self.age_form))
        except Exception as err:
            # Else build a new one, but save it later for further analysis.
            # It can slow down small batches, but extremely useful for big fitting

            # start with finding when SFR will change
            tBin = np.logspace(np.log10(self.lastBin), np.log10(self.age_form), self.nLevels).astype(int)[::-1]
            tBin = np.append(tBin, [0])

            # Open the file contaning the changes (if its check/config take all zeros)
            try:
                sfrChange = np.load('out/SFHs/RandomChange/%i_%i.npy' % (self.nModel, self.nLevels))
            except Exception as err2:
                sfrChange = np.zeros([self.nLevels + 1])

            # Prepare SFR table
            self.sfr = np.zeros([self.age_form + 1]) + 1
            for change in range(len(sfrChange) - 1):
                self.sfr[tBin[change + 1]:tBin[change]] = self.sfr[tBin[change]] / 10 ** sfrChange[change]
            self.sfr = self.sfr[::-1]

            # Save SFR table (if its check/config skip)
            try:
                np.save('out/SFHs/SFH_%i_%i_%i_%i.npy'%(self.nModel, self.nLevels, self.lastBin,self.age_form), self.sfr)
            except:pass


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

# CreationModule to be returned by get_module
Module = SFHNonParam
