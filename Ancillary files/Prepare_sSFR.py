import numpy as np
from astropy.io import fits
from astrop.table import Table
import astropy as u


def sSFH(gal, sfhFile, resultFile, overwrite = False):
    """
    Function for producing sSFR in cosmic time
    :param gal: astropy table record from results.fits, produced by cigale run
    :sfhFile: path to the sfhFile produced by cigale run
    :resultFile: path to the result file
    :overwrite: if you want to make a new file
    :return: two tables: realCosmicTime - spaced over 1Myr between the formation redshift and the observation redshift
                         ssfr - value of sSFR for each time step [1/yr]
    """
    mBest = gal['best.stellar.m_star']  # best fit M\odot
    mBayes = gal['bayes.stellar.m_star']  # best value from bayes stats M\odot
    z = gal['best.Universe.redshift']  # redshift of the grid
    timeRange = gal['bayes.sfh.age']  # age formation of the galaxy in Myr
    cosmicAge = cosmo.age(z).value * 1e3  # age at given redshit in Myr
    mRatio = mBayes / mBest  # ratio between bayes and best
    SFH = fits.open(sfhFile)[1].data

    xtime = SFH['time']  # time in Myr
    sfr = SFH['SFR'] * mRatio  # sfr in M/yr, multiplied by ratio
    # to make SFH produce bayes mass

    realCosmicTimeSpaced = (xtime * timeRange /
                            np.max(xtime) + cosmicAge - timeRange)
    # this is the scaled time
    # and shifted to match Universe age
    realCosmicTime = np.arange(round(realCosmicTimeSpaced[0], 0),
                                round(realCosmicTimeSpaced[-1], 0) + 1)
    # this is cosmic time spaced by 1Myr
    ### due to changing in spacing,
    ### I now simply build new array of SFR assigning the closest value
    newSFR = np.zeros([len(realCosmicTime)])
    for i in range(len(newSFR)):
        dx = abs(realCosmicTimeSpaced - realCosmicTime[i])
        ind = np.where(dx == np.min(dx))[0][0]
        newSFR[i] = sfr[ind]

    ### due to changing in spacing, we need to assure
    ### SFH will still produce the same mass, bayes mass
    totalMassNew = 1e6 * np.dot(SSPmass[0:len(newSFR)], newSFR[::-1])
    sfr = newSFR / (totalMassNew / mBayes)

    ### finally, produce mass evolution
    MG = np.array([1])
    for i in range(len(sfr)-1):
        i += 1
        estimatedMBayes = 1e6 * np.dot(SSPmass[0:i], sfr[0:i][::-1])
        MG = np.append(MG, [estimatedMBayes])

    ### save the results
    res = Table()
    res['sfr'] = sfr
    res['mass'] = MG
    res['cosmicTime'] = realCosmicTime
    res.units = ['Msolar/yr', 'Msolar', 'Myr']
    res.write(resultFile, overwrite = overwrite)

    ### get the ssfh
    ssfr = np.log10(sfr/MG)
    return realCosmicTime, ssfr

def findQT(realCosmicTime, ssfr):
    """
    Function to find the moment in time when the galaxy falls below the limits specified in Pacifici+16, and Lisiecki+25
    :param realCosmicTime: its the result from the sSFH function, table of cosmic time between formation and observation redshifts
    :sfhFile ssfr: its the result from the sSFH function, table of sSFR for each time step from realCosmicTime
    :return: two floats: QT - moment when galaxy is considered quiescent
                         SFRT - moment when galaxy is considered MS galaxy
    """
    ### calculate the limit for QGs for each time step
    tau = np.log10(0.2/(realCosmicTime*1e6))
    diff = ssfr - tau

    ### find where where sSFH crosses the limit, and if it goes up or down
    intersectionD = np.array([])
    intersectionU = np.array([])
    for i in range(len(ssfr)-2):
        i+=1
        if diff[i]*diff[i+1] < 0:
            if diff[i] > diff[i+1]:
                intersectionD = np.append(intersectionD, [realCosmicTime[i]])
            elif realCosmicTime[i] > 1:
                intersectionU = np.append(intersectionU, [realCosmicTime[i]])

    ### find the first time, when the galaxy is below the QGs limit for long enough (see Lisiecki+25 for details)
    intersectionU = np.append(intersectionU, [realCosmicTime[-1]])
    for i in range(len(intersectionD)):
        if intersectionU[i] - intersectionD[i] > 0.2*intersectionD[i] or intersectionU[i] == realCosmicTime[-1]:
            QT = intersectionD[i]
            break
    ### calculate the limit for MS for each time step
    tau = np.log10(1/(realCosmicTime*1e6))
    diff = ssfr - tau
    ### for nonparametric or deleyedBQ, the quenching proccess can be instantenious, if so, we return -999
    SFRT = -999
    ### find the moment when galaxy fall below MS limit before falling below QGs limit
    for i in range(len(ssfr) - 1):
        if realCosmicTime[i]<QT and diff[i] * diff[i + 1] < 0:
            SFRT = realCosmicTime[i]
            
    return QT, SFRT