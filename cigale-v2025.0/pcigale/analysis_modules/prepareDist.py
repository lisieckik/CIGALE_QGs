import numpy as np
from os import rmdir
from scipy.stats import t as tstudent

def buildCovMatrix(timeArray,
                  sigmaReg, tauEq, tauFlow, sigmaDyn, tauDyn,):

    cov_matrix = np.zeros((len(timeArray), len(timeArray)))
    for i in range(len(cov_matrix)):
        for j in range(len(cov_matrix)):
            tau = np.abs(timeArray[i] - timeArray[j])
            if tauFlow == tauEq:
                c_reg = sigmaReg ** 2 * (1 + tau / tauEq) * (np.exp(-tau / tauEq))
            else:
                c_reg = sigmaReg ** 2 / (tauFlow - tauEq) * (
                            tauFlow * np.exp(-tau / tauFlow) - tauEq * np.exp(-tau / tauEq))
            c_gmc = sigmaDyn ** 2 * np.exp(-tau / tauDyn)
            cov_matrix[i, j] = c_reg + c_gmc
    return cov_matrix

def get_tarr(ageMax, n_tarr = 8):
    edges = ageMax - np.append(np.linspace(ageMax, 30, n_tarr), [10,0]).astype(int)
    if edges[1]<10:
        edges = ageMax - np.linspace(ageMax, 0, n_tarr+2).astype(int)
    centers = []
    for i in range(len(edges) - 1):
        centers.append((edges[i + 1] - edges[i]) / 2 + edges[i])
    centers = np.array(centers)
    return centers, edges

def prepareRandomDist(conf):
    sfhMod = conf['sed_modules'][0]
    stochasticType = sfhMod.split('_')[1]
    nLevels = conf['sed_modules_params'][sfhMod]['nLevels']
    nModels = conf['sed_modules_params'][sfhMod]['nModels']

    if stochasticType == "nonparametric":
        scaleFactor = conf['sed_modules_params'][sfhMod]['scaleFactor'][0]
        if isinstance(scaleFactor, list):
            scaleFactor = scaleFactor[0]
        for nL in nLevels:
            sfrChange = tstudent.rvs(2, size=(nL)*len(nModels), scale = float(scaleFactor))
            sfrChange = np.reshape(sfrChange, [len(nModels),nL])
            np.save('out/SFHs/RandomChange/%i.npy' % (nL), sfrChange, allow_pickle=True)
    elif stochasticType == "regulator":
        rmdir('out/SFHs/RandomChange')

        age_form = conf['sed_modules_params'][sfhMod]['age_form']
        sigmaReg = conf['sed_modules_params'][sfhMod]['sigmaReg']
        tauEq = conf['sed_modules_params'][sfhMod]['tauEq']
        tauFlow = conf['sed_modules_params'][sfhMod]['tauFlow']
        sigmaDyn = conf['sed_modules_params'][sfhMod]['sigmaDyn']
        tauDyn = conf['sed_modules_params'][sfhMod]['tauDyn']

        for age in age_form:
            for nL in nLevels:
                centers, edges = get_tarr(age, n_tarr=nL)
                mean_array = np.zeros(len(centers))
                for sR in sigmaReg:
                    for tE in tauEq:
                        for tI in tauFlow:
                            for sD in sigmaDyn:
                                for tD in tauDyn:
                                    matrix = buildCovMatrix(centers, sR, tE, tI, sD, tD)
                                    sfrPoints = np.random.multivariate_normal(mean_array, matrix, size=len(nModels)).T
                                    np.save('out/SFHs/SFH_%i_%i_%.4f_%i_%i_%.4f_%i.npy' % (
                                        age,nL, sR, tE, tI, sD, tD), sfrPoints, allow_pickle=True)

