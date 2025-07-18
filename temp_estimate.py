from scipy import optimize
import numpy as np
from scipy.interpolate import interp1d, NearestNDInterpolator # LinearNDInterpolator
import sys
sys.path.append('/Users/ashley/Documents/Research/Projects/thesis_work/paperIII/final_tests/general_scripts')
#from temp_estimate_functions import find_Tdust_new

# Model grid: vary pressure, cloud edge velocity dispersion, metallicity 
# 4 cases based on the chemical compositions:
# chem 1: H2 + CO, log Z > -2 (some O atoms remain in OI!!)
# chem 2: H2 + C + O , -3 < log Z < -2
# chem 3: H2 + C+ + O, -4 < log Z < -3
# chem 4: H + C+ + O, log Z < -4 
# fiducial model is chem 0 

def jeanscalnew(usern, userveldisp, userR, userlogZ, userk_rho, cosmicrays, remyruyer, H2formationheating, chem, coratio, cosmicCrocker):

    from scipy.interpolate import interp1d, NearestNDInterpolator
    from scipy import optimize

    #(* Constants *)
    u = 1.660538291 * 1e-24
    mH = 1.007825 * u
    mD = 2.01410178 * u
    mHe = 4.002602 * u
    me = 9.10938291 * 1e-28
    eleccharge = 4.8032 * 1e-10 
    alpha = 7.2973525698 * 1e-3 # \[Alpha] = alpha
    c = 29979245500
    eV = 1.602176487 * 1e-12
    keV = 1e3 * eV
    MeV = 1e6 * eV
    kB = 1.3806488 * 1e-16
    NA = 6.02214179 * 1e23
    hbar = 6.62606957 * 1e-27/(2 * np.pi) # \[HBar] = hbar; \[Pi] = np.pi
    mu_i = 0.61 * mH # \[Mu]i = mu_i
    G = 6.67384 * 1e-8
    sigmaT = 6.6524586 *1e-25 # \[Sigma]T = sigmaT
    Msun = 1.99 * 1e33
    Lsun = 3.84 * 1e33
    Rsun = 6.955 *1e10
    sigmaSB = 5.67 *1e-5 # \[Sigma]SB = sigmaSB
    Tn = 1/0.539
    Tn3 = 1/0.854
    yr = 365.25*24*3600
    au = 1.496*1e13
    pc = 3.086*1e18
    #(* below, cosmic ray pressure in the MW, based on cosmic ray energy \density of 1eV/cm^3, and PMWCR = 1/3 of the cosmic ray energy density \*)
    PMW = 3*1e5*kB
    PMWCR = 3481.36*kB
    kmpers = 1e5

    # new calculation, to limit the number of moving parts
    # fix delta, ne, and P
    # below, for C+ 
    # data taken from https://home.strw.leidenuniv.nl/~moldata/datafiles/c+.dat
    
    # Data points for C+ collision rates with H
    Cplusk10H_data = np.array([[20, 5.96e-10], [40, 6.79e-10], [60, 7.19e-10], [80, 7.42e-10],
                               [100, 7.58e-10], [140, 7.84e-10], [200, 8.17e-10], [300, 8.63e-10],
                               [400, 9.02e-10], [600, 9.67e-10], [800, 1.02e-9], [1000, 1.066e-9],
                               [1500, 1.158e-9], [2000, 1.231e-9]])
    
    # Data points for C+ collision rates with oH2
    Cplusk10oH2_data = np.array([[10, 5.29e-10], [20, 5.33e-10], [50, 5.37e-10], [100, 5.45e-10],
                                 [200, 5.62e-10], [300, 5.71e-10], [500, 5.79e-10]])
    
    # Data points for C+ collision rates with pH2
    Cplusk10pH2_data = np.array([[10, 4.36e-10], [20, 4.53e-10], [50, 4.63e-10], [100, 4.72e-10],
                                 [200, 5.13e-10], [300, 5.55e-10], [500, 6.01e-10]])
    
    # Interpolation functions
    Cplusk10H = interp1d(Cplusk10H_data[:, 0], Cplusk10H_data[:, 1], kind='linear', fill_value='extrapolate')
    Cplusk10oH2 = interp1d(Cplusk10oH2_data[:, 0], Cplusk10oH2_data[:, 1], kind='linear', fill_value='extrapolate')
    Cplusk10pH2 = interp1d(Cplusk10pH2_data[:, 0], Cplusk10pH2_data[:, 1], kind='linear', fill_value='extrapolate')
    
    #more accurate implementation of C+ cooling based on statistical equilibrium
    def CpluscoolBetter(usernH, userT, userxHI, userxH2, userxCplus, usermuH):
        # valid for both low and high densities
        # assume no other background radiation field
        # assume only two main quantum states
        # collisional rate coefficients for collisions between C+ and H at different temperatures
        # assume main collision partner is H
        
        # Constants
        CplusA10 = 2.3e-6  # Einstein A for C+
        Cplusg0 = 2  # statistical weight of the lower quantum state
        Cplusg1 = 4  # statistical weight of the upper quantum state
        CplusT = 91.21
        Cplusnu10 = 1900.5369e9
        
        # Rates for He are not given, so scale down rates of oH2 by sqrt(2) to approximate for rates for He
        Cplusk10He = Cplusk10oH2(userT) * np.sqrt(2)
        
        Cpluskbar10 = (userxHI * Cplusk10H(userT) + 0.75 * userxH2 * Cplusk10oH2(userT) +
                       0.25 * userxH2 * Cplusk10pH2(userT) + 0.1 * Cplusk10He)
        
        Cpluskbar01 = (Cplusg1 / Cplusg0) * Cpluskbar10 * np.exp(-CplusT / userT)
        
        Cplusx1 = userxCplus / (1 + (Cpluskbar10 * usernH + CplusA10) / (Cpluskbar01 * usernH))
        # based on statistical equilibrium: -k01n0n + k10n1n + A10n1n = 0 and n0+n1 = nC+ = 1e-4*n*Z/Zsolar
        
        # below, if C+ cooling is to be used
        cooling = -Cplusx1 * CplusA10 * kB * CplusT / (usermuH * mH)
        
        # optical depth calculation
        Cplusf10 = (CplusA10 * me * c**3 * Cplusg1) / (8 * np.pi**2 * eleccharge**2 * Cplusnu10**2 * Cplusg0)
        Cpluslambda10 = c / Cplusnu10
        
        return cooling, Cplusf10, Cpluslambda10
    
    # below, for O
    Og0 = 5
    Og1 = 3
    Og2 = 1
    OA10 = 8.91e-5
    OA20 = 1.34e-10
    OA21 = 1.75e-5
    Onu10 = 4744.77749e9
    Onu20 = 6804.84658e9
    Onu21 = 2060.06909e9
    OT10 = 227.712
    OT20 = 326.579
    OT21 = 326.579
    
    data_ok10H = np.array([[10, 7.02e-11], [20, 8.20e-11], [30, 9.06e-11], [40, 9.85e-11], [60, 1.14e-10], [80, 1.30e-10],
                           [110, 1.55e-10], [160, 1.94e-10], [220, 2.37e-10], [320, 2.997e-10], [450, 3.66e-10],
                           [630, 4.38e-10], [890, 5.16e-10], [1260, 5.96e-10], [1780, 6.77e-10], [2510, 7.63e-10],
                           [3550, 8.62e-10], [5010, 9.79e-10], [8000, 1.18e-9]])
    Ok10H = interp1d(data_ok10H[:, 0], data_ok10H[:, 1], kind=3, fill_value='extrapolate')
    
    data_ok20H = np.array([[10, 7.31e-11], [20, 6.95e-11], [30, 7.11e-11], [40, 7.42e-11], [60, 8.26e-11], [80, 9.22e-11],
                           [110, 1.08e-10], [160, 1.35e-10], [220, 1.68e-10], [320, 2.18e-10], [450, 2.73e-10],
                           [630, 3.35e-10], [890, 4.03e-10], [1260, 4.74e-10], [1780, 5.51e-10], [2510, 6.37e-10],
                           [3550, 7.37e-10], [5010, 8.49e-10], [8000, 1.02e-9]])
    Ok20H = interp1d(data_ok20H[:, 0], data_ok20H[:, 1], kind=3, fill_value='extrapolate')
    
    data_ok21H = np.array([[10, 1.23e-10], [20, 1.13e-10], [30, 1.10e-10], [40, 1.10e-10], [60, 1.11e-10], [80, 1.12e-10],
                           [110, 1.15e-10], [160, 1.22e-10], [220, 1.34e-10], [320, 1.58e-10], [450, 1.92e-10],
                           [630, 2.39e-10], [890, 3.04e-10], [1260, 3.92e-10], [1780, 5.06e-10], [2510, 6.48e-10],
                           [3550, 8.22e-10], [5010, 1.02e-9], [8000, 1.34e-9]])
    Ok21H = interp1d(data_ok21H[:, 0], data_ok21H[:, 1], kind=3, fill_value='extrapolate')
    
    data_ok10oH2 = np.array([[10, 1.32e-10], [20, 1.39e-10], [30, 1.44e-10], [40, 1.49e-10], [60, 1.59e-10], [80, 1.70e-10],
                             [110, 1.84e-10], [160, 2.06e-10], [220, 2.29e-10], [320, 2.61e-10], [450, 2.96e-10],
                             [630, 3.36e-10], [890, 3.85e-10], [1260, 4.41e-10], [1780, 5.03e-10], [2510, 5.72e-10],
                             [3550, 6.51e-10], [5010, 7.39e-10], [8000, 8.51e-10]])
    Ok10oH2 = interp1d(data_ok10oH2[:, 0], data_ok10oH2[:, 1], kind=3, fill_value='extrapolate')
    
    data_ok20oH2 = np.array([[10, 6.50e-11], [20, 7.60e-11], [30, 8.24e-11], [40, 8.79e-11], [60, 9.85e-11], [80, 1.08e-10],
                             [110, 1.22e-10], [160, 1.43e-10], [220, 1.63e-10], [320, 1.91e-10], [450, 2.22e-10],
                             [630, 2.59e-10], [890, 3.07e-10], [1260, 3.66e-10], [1780, 4.38e-10], [2510, 5.21e-10],
                             [3550, 6.17e-10], [5010, 7.27e-10], [8000, 8.73e-10]])
    Ok20oH2 = interp1d(data_ok20oH2[:, 0], data_ok20oH2[:, 1], kind=3, fill_value='extrapolate')
    
    data_ok21oH2 = np.array([[10, 2.64e-12], [20, 3.08e-12], [30, 3.31e-12], [40, 3.49e-12], [60, 3.83e-12], [80, 4.14e-12],
                             [110, 4.60e-12], [160, 5.35e-12], [220, 6.22e-12], [320, 7.61e-12], [450, 9.34e-12],
                             [630, 1.16e-11], [890, 1.47e-11], [1260, 1.88e-11], [1780, 2.43e-11], [2510, 3.15e-11],
                             [3550, 4.10e-11], [5010, 5.28e-11], [8000, 7.05e-11]])
    Ok21oH2 = interp1d(data_ok21oH2[:, 0], data_ok21oH2[:, 1], kind=3, fill_value='extrapolate')
    
    data_ok10pH2 = np.array([[10, 1.18e-10], [20, 1.28e-10], [30, 1.33e-10], [40, 1.37e-10], [60, 1.46e-10], [80, 1.54e-10],
                             [110, 1.64e-10], [160, 1.77e-10], [220, 1.88e-10], [320, 2.01e-10], [450, 2.17e-10],
                             [630, 2.37e-10], [890, 2.66e-10], [1260, 3.01e-10], [1780, 3.44e-10], [2510, 3.95e-10],
                             [3550, 4.57e-10], [5010, 5.30e-10], [8000, 6.32e-10]])
    Ok10pH2 = interp1d(data_ok10pH2[:, 0], data_ok10pH2[:, 1], kind=3, fill_value='extrapolate')
    
    data_ok20pH2 = np.array([[10, 8.19e-11], [20, 1.02e-10], [30, 1.12e-10], [40, 1.20e-10], [60, 1.32e-10], [80, 1.40e-10],
                             [110, 1.49e-10], [160, 1.56e-10], [220, 1.59e-10], [320, 1.60e-10], [450, 1.63e-10],
                             [630, 1.72e-10], [890, 1.93e-10], [1260, 2.25e-10], [1780, 2.68e-10], [2510, 3.22e-10],
                             [3550, 3.90e-10], [5010, 4.70e-10], [8000, 5.84e-10]])
    Ok20pH2 = interp1d(data_ok20pH2[:, 0], data_ok20pH2[:, 1], kind=3, fill_value='extrapolate')
    
    data_ok21pH2 = np.array([[10, 7.99e-14], [20, 1.59e-13], [30, 2.63e-13], [40, 3.97e-13], [60, 6.66e-13], [80, 8.69e-13],
                             [110, 1.05e-12], [160, 1.18e-12], [220, 1.23e-12], [320, 1.37e-12], [450, 1.75e-12],
                             [630, 2.61e-12], [890, 4.29e-12], [1260, 7.04e-12], [1780, 1.11e-11], [2510, 1.68e-11],
                             [3550, 2.45e-11], [5010, 3.48e-11], [8000, 5.09e-11]])
    Ok21pH2 = interp1d(data_ok21pH2[:, 0], data_ok21pH2[:, 1], kind=3, fill_value='extrapolate')
    
    data_ok10He = np.array([[10, 1.65e-11], [20, 1.86e-11], [30, 2.15e-11], [40, 2.46e-11], [60, 3.09e-11], [80, 3.71e-11],
                            [110, 4.56e-11], [160, 5.79e-11], [220, 7.03e-11], [320, 8.77e-11], [450, 1.07e-10],
                            [630, 1.32e-10], [890, 1.66e-10], [1260, 2.08e-10], [1780, 2.57e-10], [2510, 3.12e-10],
                            [3550, 3.75e-10], [5010, 4.46e-10], [8000, 5.58e-10]])
    Ok10He = interp1d(data_ok10He[:, 0], data_ok10He[:, 1], kind=3, fill_value='extrapolate')
    
    data_ok20He = np.array([[10, 2.80e-11], [20, 2.95e-11], [30, 3.33e-11], [40, 3.80e-11], [60, 4.81e-11], [80, 5.81e-11],
                            [110, 7.23e-11], [160, 9.30e-11], [220, 1.13e-10], [320, 1.40e-10], [450, 1.68e-10],
                            [630, 2.00e-10], [890, 2.43e-10], [1260, 2.97e-10], [1780, 3.60e-10], [2510, 4.29e-10],
                            [3550, 5.04e-10], [5010, 5.86e-10], [8000, 7.07e-10]])
    Ok20He = interp1d(data_ok20He[:, 0], data_ok20He[:, 1], kind=3, fill_value='extrapolate')
    
    data_ok21He = np.array([[10, 1.05e-13], [20, 1.63e-13], [30, 2.51e-13], [40, 3.65e-13], [60, 6.66e-13], [80, 1.06e-12],
                            [110, 1.80e-12], [160, 3.37e-12], [220, 5.63e-12], [320, 9.92e-12], [450, 1.58e-11],
                            [630, 2.39e-11], [890, 3.48e-11], [1260, 4.90e-11], [1780, 6.69e-11], [2510, 8.49e-11],
                            [3550, 1.18e-10], [5010, 1.53e-10], [8000, 2.14e-10]])
    Ok21He = interp1d(data_ok21He[:, 0], data_ok21He[:, 1], kind=3, fill_value='extrapolate')
    
    def OcoolBetter(usernH, userT, userxHI, userxH2, userxO, usermuH):
        Okbar10 = (userxHI*Ok10H(userT) + 0.75*userxH2*Ok10oH2(userT) + 0.25*userxH2*Ok10pH2(userT) + 0.1*Ok10He(userT))
        Okbar20 = (userxHI*Ok20H(userT) + 0.75*userxH2*Ok20oH2(userT) + 0.25*userxH2*Ok20pH2(userT) + 0.1*Ok20He(userT))
        Okbar21 = (userxHI*Ok21H(userT) + 0.75*userxH2*Ok21oH2(userT) + 0.25*userxH2*Ok21pH2(userT) + 0.1*Ok21He(userT))
        Okbar01 = (Og1/Og0)*Okbar10*np.exp(-OT10/userT)
        Okbar02 = (Og2/Og0)*Okbar20*np.exp(-OT20/userT)
        Okbar12 = (Og2/Og1)*Okbar21*np.exp(-OT21/userT)
        Ox1x0 = (usernH * (OA20*Okbar01 + OA21*(Okbar01 + Okbar02) + Okbar02*Okbar21*usernH + Okbar01*(Okbar20 + Okbar21)*usernH))/(OA10*(OA20 + OA21 + (Okbar20 + Okbar21)*usernH) + usernH*(OA21*Okbar10 + OA20*(Okbar10 + Okbar12) + Okbar12*Okbar20*usernH + Okbar10*(Okbar20 + Okbar21)*usernH))
        Ox2x0 = (usernH * (OA10*Okbar02 + Okbar01*Okbar12*usernH + Okbar02*(Okbar10 + Okbar12)*usernH))/(OA10*(OA20 + OA21 + (Okbar20 + Okbar21)*usernH) + usernH*(OA21*Okbar10 + OA20*(Okbar10 + Okbar12) + Okbar12*Okbar20*usernH + Okbar10*(Okbar20 + Okbar21)*usernH))
        Ox0 = userxO / (1 + Ox1x0 + Ox2x0)
        Ox1 = Ox1x0 * Ox0
        Ox2 = Ox2x0 * Ox0
        cooling = -(Ox1*OA10*OT10 + Ox2*OA20*OT20 + Ox2*OA21*OT21)*kB/(usermuH*mH)
        # optical depth calculation
        Of10 = OA10 * me * c**3 * Og1 / (8 * np.pi**2 * eleccharge**2 * Onu10**2 * Og0)
        Olambda10 = c / Onu10
        Of20 = OA20 * me * c**3 * Og2 / (8 * np.pi**2 * eleccharge**2 * Onu20**2 * Og0)
        Olambda20 = c / Onu20
        Of21 = OA21 * me * c**3 * Og2 / (8 * np.pi**2 * eleccharge**2 * Onu21**2 * Og1)
        Olambda21 = c / Onu21
        return cooling, Of10, Olambda10, Of20, Olambda20, Of21, Olambda21, Ox0, Ox1, Ox2
    
    # below, for C
    Cg0 = 1
    Cg1 = 3
    Cg2 = 5
    CA10 = 7.88e-8
    CA21 = 2.65e-7
    CA20 = 1.81e-14
    Cnu10 = 492.160651e9
    Cnu21 = 809.34197e9
    Cnu20 = 1301.50262e9
    CT10 = 23.620
    CT21 = 62.462
    CT20 = 62.462
    
    x_k10H = np.array([10, 20, 50, 100, 200])
    y_k10H = np.array([1.6e-10, 1.7e-10, 1.6e-10, 1.6e-10, 1.7e-10])
    Ck10H = interp1d(x_k10H, y_k10H, kind=3, fill_value='extrapolate')
    
    x_k20H = np.array([10, 20, 50, 100, 200])
    y_k20H = np.array([1e-10, 9.7e-11, 9.5e-11, 9.4e-11, 1e-10])
    Ck20H = interp1d(x_k20H, y_k20H, kind=3, fill_value='extrapolate')
    
    x_k21H = np.array([10, 20, 50, 100, 200])
    y_k21H = np.array([2.3e-10, 2.4e-10, 2.6e-10, 2.9e-10, 3.2e-10])
    Ck21H = interp1d(x_k21H, y_k21H, kind=3, fill_value='extrapolate')
    
    x_k10oH2 = np.array([10, 20, 50, 100, 200, 500, 1000, 1200])
    y_k10oH2 = np.array([7.3e-11, 8.1e-11, 7.6e-11, 7.1e-11, 7.1e-11, 8.0e-11, 8.6e-11, 8.5e-11])
    Ck10oH2 = interp1d(x_k10oH2, y_k10oH2, kind=3, fill_value='extrapolate')
    
    x_k20oH2 = np.array([10, 20, 50, 100, 200, 500, 1000, 1200])
    y_k20oH2 = np.array([5.2e-11, 5.8e-11, 6.3e-11, 6.9e-11, 7.9e-11, 1.0e-10, 1.1e-10, 1.1e-10])
    Ck20oH2 = interp1d(x_k20oH2, y_k20oH2, kind=3, fill_value='extrapolate')
    
    x_k21oH2 = np.array([10, 20, 50, 100, 200, 500, 1000, 1200])
    y_k21oH2 = np.array([9.2e-11, 1.1e-10, 1.3e-10, 1.5e-10, 1.8e-10, 2.5e-10, 2.8e-10, 2.8e-10])
    Ck21oH2 = interp1d(x_k21oH2, y_k21oH2, kind=3, fill_value='extrapolate')
    
    x_k10pH2 = np.array([10, 20, 50, 100, 200, 500, 1000, 1200])
    y_k10pH2 = np.array([9.6e-11, 1.0e-10, 7.7e-11, 6.7e-11, 6.7e-11, 7.6e-11, 8.1e-11, 8.0e-11])
    Ck10pH2 = interp1d(x_k10pH2, y_k10pH2, kind=3, fill_value='extrapolate')
    
    x_k20pH2 = np.array([10, 20, 50, 100, 200, 500, 1000, 1200])
    y_k20pH2 = np.array([9.6e-11, 9.9e-11, 9.2e-11, 8.6e-11, 8.5e-11, 9.6e-11, 1.0e-10, 1.0e-10])
    Ck20pH2 = interp1d(x_k20pH2, y_k20pH2, kind=3, fill_value='extrapolate')
    
    x_k21pH2 = np.array([10, 20, 50, 100, 200, 500, 1000, 1200])
    y_k21pH2 = np.array([1.8e-10, 1.8e-10, 1.8e-10, 1.8e-10, 1.9e-10, 2.3e-10, 2.6e-10, 2.6e-10])
    Ck21pH2 = interp1d(x_k21pH2, y_k21pH2, kind=3, fill_value='extrapolate')
    
    x_k10He = np.array([10, 20, 40, 100, 150])
    y_k10He = np.array([8.5e-12, 1.35e-11, 1.59e-11, 1.74e-11, 1.86e-11])
    Ck10He = interp1d(x_k10He, y_k10He, kind=3, fill_value='extrapolate')
    
    x_k20He = np.array([10, 20, 40, 100, 150])
    y_k20He = np.array([4.05e-11, 4.23e-11, 4.34e-11, 4.42e-11, 4.53e-11])
    Ck20He = interp1d(x_k20He, y_k20He, kind=3, fill_value='extrapolate')
    
    x_k21He = np.array([10, 20, 40, 100, 150])
    y_k21He = np.array([7.15e-11, 7.48e-11, 7.75e-11, 8.29e-11, 8.83e-11])
    Ck21He = interp1d(x_k21He, y_k21He, kind=3, fill_value='extrapolate')
    
    def CcoolBetter(usernH, userT, userxHI, userxH2, userxC, usermuH):
        Ckbar10 = (userxHI*Ck10H(userT) + 0.75*userxH2*Ck10oH2(userT) +
                   0.25*userxH2*Ck10pH2(userT) + 0.1*Ck10He(userT))
        Ckbar20 = (userxHI*Ck20H(userT) + 0.75*userxH2*Ck20oH2(userT) +
                   0.25*userxH2*Ck20pH2(userT) + 0.1*Ck20He(userT))
        Ckbar21 = (userxHI*Ck21H(userT) + 0.75*userxH2*Ck21oH2(userT) +
                   0.25*userxH2*Ck21pH2(userT) + 0.1*Ck21He(userT))
        Ckbar01 = (Cg1/Cg0)*Ckbar10*np.exp(-CT10/userT)
        Ckbar02 = (Cg2/Cg0)*Ckbar20*np.exp(-CT20/userT)
        Ckbar12 = (Cg2/Cg1)*Ckbar21*np.exp(-CT21/userT)
        Cx1x0 = (usernH * (CA20 * Ckbar01 + CA21 * (Ckbar01 + Ckbar02) +
                           Ckbar02 * Ckbar21 * usernH +
                           Ckbar01 * (Ckbar20 + Ckbar21) * usernH)) / (
                               CA10 * (CA20 + CA21 + (Ckbar20 + Ckbar21) * usernH) +
                               usernH * (CA21 * Ckbar10 + CA20 * (Ckbar10 + Ckbar12) +
                                         Ckbar12 * Ckbar20 * usernH +
                                         Ckbar10 * (Ckbar20 + Ckbar21) * usernH))
        Cx2x0 = (usernH * (CA10 * Ckbar02 + Ckbar01 * Ckbar12 * usernH +
                           Ckbar02 * (Ckbar10 + Ckbar12) * usernH)) / (
                               CA10 * (CA20 + CA21 + (Ckbar20 + Ckbar21) * usernH) +
                               usernH * (CA21 * Ckbar10 + CA20 * (Ckbar10 + Ckbar12) +
                                         Ckbar12 * Ckbar20 * usernH +
                                         Ckbar10 * (Ckbar20 + Ckbar21) * usernH))
        Cx0 = userxC / (1 + Cx1x0 + Cx2x0)
        Cx1 = Cx1x0 * Cx0
        Cx2 = Cx2x0 * Cx0
        cooling = -(Cx1 * CA10 * CT10 + Cx2 * CA20 * CT20 +
                    Cx2 * CA21 * CT21) * kB / (usermuH * mH)
        
        # Optical depth calculation
        Cf10 = CA10 * me * c**3 * Cg1 / (8 * np.pi**2 * eleccharge**2 * Cnu10**2 * Cg0)
        Clambda10 = c / Cnu10
        Cf20 = CA20 * me * c**3 * Cg2 / (8 * np.pi**2 * eleccharge**2 * Cnu20**2 * Cg0)
        Clambda20 = c / Cnu20
        Cf21 = CA21 * me * c**3 * Cg2 / (8 * np.pi**2 * eleccharge**2 * Cnu21**2 * Cg1)
        Clambda21 = c / Cnu21
        
        return (cooling, Cf10, Clambda10, Cf20, Clambda20, Cf21, Clambda21, Cx0, Cx1, Cx2)
    
    
    def remyruter(userZ):
        def metal(Z):
            return 12 + np.log10(Z * 4.9e-4)
        
        def remyruyterfit(Z):
            if metal(Z) <= 7.96:
                return 0.68 + 3.08 * (metal(1) - metal(Z))
            else:
                return 2.21 + 1 * (metal(1) - metal(Z))
            
        def dusttogasratio(Z):
            return 1 / (10 ** remyruyterfit(Z))
            
        return np.log10(dusttogasratio(userZ) * 162)


    # check if Psi_gd is smaller than dust when its optically thick or not
    planckmean = np.loadtxt("/Users/ashley/Documents/Research/Projects/thesis_work/paperIII/final_tests/general_scripts/semenov2003.txt", delimiter=",")
    planckmean = np.transpose(planckmean)
    planckmean_wavelength = planckmean[0]
    planckmean_opacity = planckmean[1]
    planckmeanopacity = interp1d(planckmean_wavelength, planckmean_opacity, kind=3, fill_value='extrapolate')
    
    # CO cooling grid generated from Despotic
    # input file grid_COcooling.out - only has sigma_v from 0.5 to 3 km/s
    # input file grid_COcooling_new.out - has sigma_v from 0.5 to 5 km/s
    
    # 4D grid - log nH, Tgas, log NCO, sigma_v/1e5
    # last two columns are cooling rate and optical depth of 1-0 line
    
    COfile = np.genfromtxt("/Users/ashley/Documents/Research/Projects/thesis_work/paperIII/final_tests/check_models/temperature_estimates/grid_COcooling_new.out")

    # Extracting the relevant columns from COfile
    log_nH = np.log10(COfile[:, 0])
    Tgas = COfile[:, 1]
    log_NCO = np.log10(COfile[:, 2])
    sigma_v = COfile[:, 3] / 1e5
    cooling_rate = COfile[:, 4]
    optical_depth = COfile[:, 5]
    
    #COcooling = LinearNDInterpolator(list(zip(log_nH,Tgas,log_NCO,sigma_v)), cooling_rate)
    COcooling = NearestNDInterpolator(list(zip(log_nH,Tgas,log_NCO,sigma_v)), cooling_rate)
    
    # Read cloudfile data
    cloudfile = np.genfromtxt('cldensities.tsv', delimiter='\t')

    # Extract columns from cloudfile
    logNH2 = cloudfile[:, 0]
    lognH2 = cloudfile[:, 1]

    # Create an interpolation function
    cloudNH2 = interp1d(lognH2, logNH2, kind=1, fill_value='extrapolate')

    def find_Tdust_new(usern, userveldisp, userR, loguserZ, userk_rho, cosmicrays, remyruyer):
        
        loguserdelta = loguserZ if remyruyer == 0 else remyruter(10**loguserZ)
        muH = 1.4
        if loguserZ >= -4:
            mu = 2.33
        elif loguserZ < -5:
            mu = 1.4
        else:
            mu = 2.33 + 0.93 * (4 + loguserZ)

        def rho(n):
            return n * mu * mH

        rho_ex = rho(usern)

        def nH(n):
            return rho(n) / (muH * mH)

        alphavir = 2.0
        capitalSigma = 10**(np.mean(cloudfile[:, 0])) * mu * mH
        P = (3.0 / 20.0) * np.pi * alphavir * G * capitalSigma**2

        def k_rho(n, R):
            #return 3 - (4 * rho(n) * R / capitalSigma)
            return userk_rho

        #k_rho_ex = k_rho(usern, userR)
        k_rho_ex = userk_rho
        #print('k_rho_ex')
        #print(k_rho_ex)
        #print('--')
        
        def M(R):
            return np.pi * capitalSigma * R**2
    
        M_ex = M(userR)

        psi = 2.5 * 10**14

        def L(n, R, epsilonM, epsilonL):
            return epsilonL * epsilonM * psi * np.sqrt(3 * G * rho(n) / (3 - k_rho(n, R))) * M(R)

        kappa_d0 = 0.27
        T0 = 144.0

        def alpha1(n, R, beta):
            return 2 * beta + 4 * (k_rho(n, R) - 1)

        def Rct(n, R, epsilonM, epsilonL, beta, delta):
            return ((L(n, R, epsilonM, epsilonL) / M(R)) * (M(R) / (np.pi * R**2))**((4 + beta) / beta) / (4 * 1.6 * sigmaSB) * ((3 - k_rho(n, R)) * delta * kappa_d0 / (4 * (k_rho(n, R) - 1) * T0**beta))**(4 / beta))**(10 * beta / (beta - 10 * alpha1(n, R, beta)))

        Rct_ex = Rct(usern, userR, 0.5, 0.75, 2, 10**loguserdelta)

        def Rch(Rct):
            return userR / Rct

        Rch_ex = Rch(Rct_ex)

        def Lt(n, R, epsilonM, epsilonL, beta, delta):
            return 1.6 * Rct(n, R, epsilonM, epsilonL, beta, delta)**(1.0 / 10.0)

        def kT(n, R, epsilonM, epsilonL, beta, delta):
            # for troubleshooting
            #print('kT')
            #print(k_rho(n, R))
            #print(Rct(n, R, epsilonM, epsilonL, beta, delta))
            #print('--')
            return (0.48 * k_rho(n, R)**0.005 / Rct(n, R, epsilonM, epsilonL, beta, delta)**(0.02 * k_rho(n, R)**1.09) + 0.1 * k_rho(n, R)**5.5 / Rct(n, R, epsilonM, epsilonL, beta, delta)**(0.7 * k_rho(n, R)**1.09))

        kT_ex = kT(usern, userR, 0.5, 0.75, 2, 10**loguserdelta)
        #print('kT_ex')
        #print(kT_ex)
        #print('--')

        def Tdust(n, R, epsilonM, epsilonL, beta, delta):
            return (((L(n, R, epsilonM, epsilonL) / M(R)) / (4 * sigmaSB * Lt(n, R, epsilonM, epsilonL, beta, delta)))**(k_rho(n, R) - 1 + beta * kT(n, R, epsilonM, epsilonL, beta, delta)) * ((3 - k_rho(n, R)) * delta * kappa_d0 / (4 * (k_rho(n, R) - 1) * T0**beta))**(4 * kT(n, R, epsilonM, epsilonL, beta, delta) - 2) * (M(R) / (np.pi * R**2))**((4 + beta) * kT(n, R, epsilonM, epsilonL, beta, delta) + k_rho(n, R) - 3))**(1 / alpha1(n, R, beta))

        Tdust_val = Tdust(usern, userR, 0.5, 0.75, 2, 10**loguserdelta)

        return usern, userveldisp / kmpers, userR, loguserZ, rho_ex, P / kB, capitalSigma, k_rho_ex, M_ex / Msun, np.log10(Rct_ex), Rch_ex, kT_ex, Tdust_val

    solution = find_Tdust_new(usern, userveldisp, userR, userlogZ, userk_rho, cosmicrays, remyruyer)

    # below, for fiducial model where chemical composition is varied
    # case = Which(userlogZ < -5, 4, -5 <= userlogZ < -4, 3.5, -4 <= userlogZ < -3, 3, -3 <= userlogZ < -2, 2, -2 <= userlogZ < -1, 1.5, userlogZ >= -1, 1)
    # below, for a singular chemical composition
    # case = allcase
    # [usern, userveldisp/(kmpers), userR, loguserZ, Rhoe, P/kB, Sigma, kRhoex, Mex/Msun, Log10[Rctex], Rchex, kTex, Tdustval]
    
    Redge = solution[2]
    Z = 10 ** solution[3]
    Rhoe = solution[4]
    P = kB * solution[5]
    #Sigma = solution[7]
    kRho = solution[7]
    #print('kRho')
    #print(kRho)
    #print('--')
    Rch =  solution[10] #10 ** solution[10]
    kT = solution[11]
    Tdust = solution[12]
    
    # which cosmic ray prescription to use? Crocker+2021 or Padovani+2009
    cosmicPadovani = 1 - cosmicCrocker
    loguserdelta = (
        np.log10(Z) if remyruyer == 0 else remyruter(Z)[0] #First(remyruter[Z])
    )
    # change the above line to change how dust to gas ratio evolves with metallicity

    minR = None
    if remyruyer == 0 and cosmicrays == 0 and userlogZ >= -1.5:
        minR = np.log10(0.1 * Rch)
    elif remyruyer == 0 and cosmicrays == 0 and userlogZ < -1.5:
        minR =  np.log10(0.00001 * Redge)
    elif remyruyer == 1 and cosmicrays == 0:
        minR =  np.log10(0.00001 * Redge)
    elif cosmicrays == 1 and userlogZ <= -1.5:
        minR =  np.log10(0.00001 * Redge)
    elif cosmicrays == 1 and userlogZ > -1.5:
        minR =  np.log10(0.1 * Rch)

    maxR = None
    if cosmicrays == 0:
        maxR =  np.log10(Redge)
    elif cosmicrays == 1:
        maxR =  np.log10(1 * Redge)

    nums = 50
    Rarray = np.logspace(minR,maxR,nums)

    # Rarray = 10 ** np.array([np.log10(Rch), np.log10(Redge)])  # Uncomment this line if you want to use this alternative Rarray calculation
    # new: for gamma galaxies, inner range for remyruyer=0 and cosmicrays=0 and userlogZ >= -1.5 increased from Rch to 0.1Rch
    # muH is same for all cases (1 + 4xHe, xHe = 0.1), but decide the value of mu based on the case

    muH = 1.4
    mu = (
        2.33 if userlogZ >= -4
        else 1.4 if userlogZ < -5
        else 2.33 + 0.93 * (4 + userlogZ) if -5 <= userlogZ < -4
        else 0  # Modify this condition according to your requirements
    )
    
    def Tdustarray(R):
        # for troubleshooting
        #print('Tdustarray')
        #print(Tdust)
        #print(Redge)
        #print(kT)
        #print('--')
        return Tdust * (R / Redge) ** (-kT)
    
    def Rhoarray(R):
        return Rhoe * (R / Redge) ** (-kRho)
    
    def nH(R):
        # for troubleshooting:
        #print('nH')
        #print(Rhoarray(R))
        #print('--')
        return Rhoarray(R) / (muH * mH)
    
    def n(R):
        return Rhoarray(R) / (mu * mH)
    
    # new stuff, added April 28 2023
    # radial variations in Sigma
    def NH2(R):
        return 10 ** (cloudNH2(np.log10(n(R))))

    def Sigma(R):
        return NH2(R) * mu * mH
    
    Sigmaex = [Sigma(Rarray[i]) for i in range(len(Rarray))]
    
    # cosmic ray heating
    # from Krumholz 2012
    # below, fraction of cosmic ray to gas pressure in any ISM
    # fCR based on 6th degree poly fit to Crocker et al 2021, fig. 8b
    def SigmaMsunpc2(R):
        return Sigma(R) * pc ** 2 / Msun
    
    def fCR(R):
        sigma_msun_pc2 = SigmaMsunpc2(R)
        return (
            0 if sigma_msun_pc2 > 10 ** 5
            else (
                    10 ** (
                        -0.0220143 * np.log10(sigma_msun_pc2) ** 7 +
                        0.38424923 * np.log10(sigma_msun_pc2) ** 6 -
                        2.71439452 * np.log10(sigma_msun_pc2) ** 5 +
                        9.93461456 * np.log10(sigma_msun_pc2) ** 4 -
                        20.00225763 * np.log10(sigma_msun_pc2) ** 3 +
                        21.65004378 * np.log10(sigma_msun_pc2) ** 2 -
                        12.36886449 * np.log10(sigma_msun_pc2) + 2.78117685
                    )
            )
        )

    qCR = 6.5 * eV
    zetaMW = 3e-16

    # alternate prescription from Padovani+2009, eq 27 and Table 4 (electron fit E00 
    # Padovani gives lesser cosmic ray ionization rate (zeta) at given column density, and is perhaps more suitable for molecular clouds
    
    def zeta(R):
        if cosmicrays == 0:
            return 0
        elif cosmicrays == 1 and cosmicCrocker == 1:
            return zetaMW * fCR(R) * P / PMWCR
        elif cosmicrays == 1 and cosmicPadovani == 1:
            return 2.6e-14 * (NH2(R) / 1e20) ** -0.805
        
    zetaex = [zeta(Rarray[i]) for i in range(len(Rarray))]
        
    def gammacosmic(R):
        return (qCR / (muH * mH)) * zeta(R)

    # use a H2 mass fraction of 1e-3 based on the results of Sharda et al 2019 for Pop III stars 
    #xH2 =Which[case \[Equal] 1, 0.749,case\[Equal]1.5,0.749,case\[Equal]2,0.749,case\[Equal]3,0.749,\case\[Equal]3.5,10^(-3 + \(userlogZ+5)(Log10[0.749]+3)),case\[Equal]4,0.001];
    
    gammacosmicex = [gammacosmic(Rarray[i]) for i in range(len(Rarray))]
    
    xH2min = (
        0.001 * 1.4 / 2.0 if chem == 0
        else 0.5 if 1 <= chem <= 3
        else 0.001 * 1.4 / 2.0 if chem == 4
        else 0  # Modify this condition according to your requirements
    )
    
    xH2max = (
        0.5 if chem <= 3
        else 0.001 * 1.4 / 2.0 if chem == 4
        else 0  # Modify this condition according to your requirements
    )
    
    xHInocosmicrays = None
    if chem == 0:
        if userlogZ < -5:
            xHInocosmicrays = 1 - 2 * xH2min
        elif userlogZ >= -4:
            xHInocosmicrays = 1 - 2 * xH2max
        elif -5 <= userlogZ < -4:
            xHInocosmicrays = (1 - 2 * xH2min) + 2 * (userlogZ + 5) * (xH2min - xH2max)
    elif chem >= 1:
        xHInocosmicrays = 1 - 2 * xH2max

    # below, heating due to H2 formation on dust
    # following eq 40 and 55-60 of Grassi et al 2014, as well as eq 5 6 
    # 7 13 of Cazaux & Spaans 2009
    
    Sdust = 1e5  # cm^2/g
    dustrho = 3.0  # assume dust density of 3 g/cm^3
    grainsize = 3.0 / (4.0 * Sdust * dustrho)  # grain size is 0.025 microns
    mgr = (np.pi * grainsize ** 2) / Sdust  # mass of each grain

    def T2d(Tg):
        return Tg / 100

    def stick(Tg, R):
        return (1 + 0.4 * np.sqrt((T2d(Tg) + Tdustarray(R)) / 100) + 0.2 * T2d(Tg) + 0.08 * (T2d(Tg))**2)**-1
    
    eP1 = 800.0
    eC1 = 7000.0
    eS1 = 200.0
    
    def eq57(Tg):
        return 4 * np.exp(-(eP1 - eS1) / (eP1 + Tg)) * (1 + np.sqrt((eC1 - eS1) / (eP1 - eS1)))**-2

    def epsilonC(Tg, R):
        return min(1, (1 - eq57(Tg)) / (1 + 0.25 * np.exp(-eS1 / Tdustarray(R)) * (1 + np.sqrt((eC1 - eS1) / (eP1 - eS1))))**2)
    
    eP2 = 700.0
    eC2 = 15000.0
    eS2 = -1000.0
    apc = 1.7e-10
    betaD = 4e9
    
    def f(Tg):
        return 2 * np.exp(-(eP2 - eS2) / (eP2 + Tg)) / (1 + np.sqrt((eC2 - eS2) / (eP2 - eS2)))**2
    
    def epsilonSi(Tg, R):
        return min(1, f(Tg) + (1 + 16 * (Tdustarray(R) / (eC2 - eS2)) * np.exp(-(eP2 / Tdustarray(R)) + betaD * apc * np.sqrt(eP2 - eS2)))**-1)
    
    def Rcazauxnocosmicrays(Tg, R):
        return 7.25e-15 * nH(R) * xHInocosmicrays() * nH(R) * np.sqrt(Tg / 100.0) * (1.75 * epsilonC(Tg, R) + 1.1 * epsilonSi(Tg, R)) * stick(Tg, R) * Z
    
    H2dustfactor = 0.34
    
    def gammaH2dustnocosmicrays(Tg, R):
        return Rcazauxnocosmicrays(Tg, R) * H2dustfactor * 4.48 * eV / (muH * mH * nH(R))

    # if cosmic rays included, then find xHI based on cosmic ray \
    # ionization, assuming cosmic rays provide the free hydrogen atoms \
    # needed for H2 formation on dust: zeta*nH*0.5*(1-xHI) = Rcazaux = \
    # Qcazaux*xHI*nH^2
    
    # Find Qcazaux for cosmic ray ionization
    
    def Qcazaux(Tg, R):
        return 7.25e-15 * np.sqrt(Tg / 100.0) * (1.75 * epsilonC(Tg, R) + 1.1 * epsilonSi(Tg, R)) * stick(Tg, R) * Z

    def xHIcosmicrays(Tg, R):
        return (zeta(R) / 2.0) / (zeta(R) / 2.0 + nH(R) * Qcazaux(Tg, R))
    
    def xHI(Tg, R):
        if cosmicrays == 0:
            return xHInocosmicrays
        elif cosmicrays == 1:
            return max(xHInocosmicrays, xHIcosmicrays(Tg, R))
        
    def Rcazauxcosmicrays(Tg, R):
        return zeta(R) * nH(R) * 0.5 * (1 - xHI(Tg, R))
    
    def gammaH2dustcosmicrays(Tg, R):
        return Rcazauxcosmicrays(Tg, R) * H2dustfactor * 4.48 * eV / (muH * mH * nH(R))

    def gammaH2dust(Tg, R):
        if cosmicrays == 0 and H2formationheating == 1:
            return gammaH2dustnocosmicrays(Tg, R)
        elif cosmicrays == 1 and H2formationheating == 1:
            return gammaH2dustcosmicrays(Tg, R)
        elif H2formationheating == 0:
            return 0

    # xH2 is always (1-xHI)/2
    
    def xH2(Tg, R):
        return (1 - xHI(Tg, R)) / 2.0

    fHD = 0.1 # from figure 4 of Omukai 2005. Upper limit is 0.1, typical value is 0.001 
    xHD = fHD * 1e-5 #  10^-5 is the D/H ratio 

    # metal number fractions 
    xCMW = 1.4e-4
    xOMW = 3.0e-4

    # use coratio = 100 only for amarsi 2019 data 
    amarsiCO = -0.0246137 * userlogZ**3 + 0.14229702 * userlogZ**2 + 0.60035716 * userlogZ - 0.09157322

    # use coratio = 103 only for nicholls 2017 empirical relation 
    logoh = userlogZ + 8.69 - 12
    nichollsCO = np.log10(10**(-0.8) + 10**(logoh + 2.72))

    # different cases for CO ratio: 
    #100 --> cubic best-fit to amarsi data using scipy curve_fit
    #101 --> upper bound for the best-fit 
    #102 --> lower bound for the best-fit 
    #< 100 --> just use the input value as the CO ratio

    coratioo = None
    if coratio < 100:
        coratioo = coratio
    elif coratio == 100:
        coratioo = amarsiCO
    elif coratio == 101:
        coratioo = amarsiCO + 0.20
    elif coratio == 102:
        coratioo = amarsiCO - 0.40
    elif coratio == 103:
        coratioo = nichollsCO
    elif coratio == 104:
        coratioo = nichollsCO + 0.40
    elif coratio == 105:
        coratioo = nichollsCO - 0.40
        
    xC = xCMW * 10**(userlogZ + coratioo)
    xO = xOMW * 10**userlogZ

    # trying radial variations in xCO at fixed metallicity, based on xHI and xH2
    # currently only implemented for metallicity >= -0.5
        
    def abCO(Tg, R):
        if userlogZ < -1:
            return 0
        elif -1 <= userlogZ < -0.5:
            return 2 * (1 + userlogZ)
        elif userlogZ >= -0.5:
            return 2 * xH2(Tg, R)

    def xCO(Tg, R):
        if chem == 0:
            return xC * abCO(Tg, R)
        elif chem == 1:
            return xC
        elif chem >= 2:
            return 0

    abCI = None
    if userlogZ < -4:
        abCI = 0
    elif -4 <= userlogZ < -3:
        abCI = 4 + userlogZ
    elif -3 <= userlogZ < -1:
        abCI = 1
    elif -1 <= userlogZ < -0.5:
        abCI = -(1 + 2 * userlogZ)
    elif userlogZ >= -0.5:
        abCI = 0

    xCI = None
    if chem == 0:
        xCI = xC * abCI
    elif chem == 1:
        xCI = 0
    elif chem == 2:
        xCI = xC
    elif chem > 2:
        xCI = 0

    def xCplus(Tg, R):
        return xC - xCI - xCO(Tg, R)

    # (when chem==1, all C is locked in CO. But some O atoms remain 
    # free because xO > xC. So, make sure that you account for the 
    # remaining O atoms by setting xOI = xO-xCO for chem==1 (I made 
    # this change in 2022 after Gen's comment on the C/O paper)

    def xOI(Tg, R):
        if chem == 0 or chem == 1:
            return xO - xCO(Tg, R)
        elif chem >= 2:
            return xO
    
    def gammacompress(Tg, R):
        return (kB * Tg / mu) * np.sqrt(32 * G * muH * nH(R) / (3 * np.pi * mH))

    #  below, if C+ cooling is to be used
    
    def funcCplus(Tg, R):
        out = CpluscoolBetter(nH(R), Tg, xHI(Tg, R), xH2(Tg, R), xCplus(Tg, R), muH) # [0]
        if len(out) > 0:
            return out[0]
        else:
            return out

    # below, if C cooling is to be used
    
    def funcC(Tg, R):
        out = CcoolBetter(nH(R), Tg, xHI(Tg, R), xH2(Tg, R), xCI, muH) # [0]
        if len(out) > 0:
            return out[0]
        else:
            return out

    # Below, full O cooling including all three quantum states
    
    def funcO(Tg, R):
        out = OcoolBetter(nH(R), Tg, xHI(Tg, R), xH2(Tg, R), xOI(Tg, R), muH) # [0]
        if len(out) > 0:
            return out[0]
        else:
            return out

    def coldensCO(Tg, R):
        return xCO(Tg, R) * Sigma(R) / (muH * mH)

    def funcCO(Tg, R):
        if xCO(Tg, R) == 0:
            return 0
        elif xCO(Tg, R) > 0:
            return -(10**(COcooling(np.log10(nH(R)), Tg, np.log10(coldensCO(Tg, R)), userveldisp/kmpers))) / (muH * mH)

    def lambdametal(Tg, R):
        return funcCplus(Tg, R) + funcC(Tg, R) + funcO(Tg, R) + funcCO(Tg, R)

    # dust-gas energy exchange psigd

    alphagd = 0.5
    Sgd = 1e5

    def psigd(Tg, R):
        #for troubleshooting:
        #print('psidg')
        #print(Tdustarray(R))
        #print(nH(R))
        #print(loguserdelta)
        #print(xHI(Tg, R))
        #print(xH2(Tg, R))
        #print('--')
        return (Tdustarray(R) - Tg) * 2 * alphagd * nH(R) * Sgd * ((10**loguserdelta) / 162) * kB * np.sqrt(8 * kB * Tg / (np.pi * mH)) * (xHI(Tg, R) / np.sqrt(1.0) + xH2(Tg, R) / np.sqrt(2.0) + 0.1 / np.sqrt(4.0))

    # H2 cooling

    def x3(Tg):
        return Tg / 1000

    def HR(Tg):
        return np.exp(-(0.13 / x3(Tg))**3) * (9.5 * 10**(-22) * x3(Tg)**3.76) / (1 + 0.12 * x3(Tg)**2.1) + 3 * 10**(-24) * np.exp(-0.51 / x3(Tg))

    def HV(Tg):
        return 6.7 * 10**(-19) * np.exp(-5.86 / x3(Tg)) + 1.6 * 10**(-18) * np.exp(-11.7 / x3(Tg))

    def h2n0(Tg, R):
        return (-103 + 97.59 * np.log10(Tg) - 48.05 * (np.log10(Tg))**2 + 10.8 * (np.log10(Tg))**3 - 0.9032 * (np.log10(Tg))**4) * nH(R)

    def lambdaH2thin(Tg, R):
        return (HR(Tg) + HV(Tg)) / (1 + (HR(Tg) + HV(Tg)) / h2n0(Tg, R))

    def lambdaH2(Tg, R):
        return -lambdaH2thin(Tg, R) * (xH2(Tg, R) / (muH * mH)) * min(1, (n(R) / (8 * 10**9))**(-0.45))

    def ncr(Tg, R):
        return (10**6) * (Tg**(-0.5)) * (1.6 * nH(R) * np.exp(-(400 / Tg)**2) + 1.4 * xH2(Tg, R) * nH(R) * np.exp(-12000 / (Tg + 1200)))

    def fchem(Tg, R):
        return (1 + ncr(Tg, R) / n(R))**-1

    def kH23b(Tg):
        if Tg <= 300:
            return (1.3 * 10**(-32)) * (Tg / 300)**-0.38
        elif Tg > 300:
            return (1.3 * 10**(-32)) * (Tg / 300)**-1.0

    def gammaH23b(Tg, R):
        if H2formationheating == 1:
            return 4.48 * eV * fchem(Tg, R) * (kH23b(Tg) / (muH * mH)) * (nH(R) * xHI(Tg, R))**2
        elif H2formationheating == 0:
            return 0

    # HD cooling 
    # coefficients from Lipovka 2005, as well as Grassi 2014
    HDc = np.zeros((5, 5))
    HDc[0, 0] = -42.56788
    HDc[0, 1] = 21.93385
    HDc[0, 2] = -10.19097
    HDc[0, 3] = 2.19906
    HDc[0, 4] = -0.17334
    HDc[1, 0] = 0.92433
    HDc[1, 1] = 0.77952
    HDc[1, 2] = -0.54263
    HDc[1, 3] = 0.11711
    HDc[1, 4] = -0.00835
    HDc[2, 0] = 0.54962
    HDc[2, 1] = -1.06447
    HDc[2, 2] = 0.62343
    HDc[2, 3] = -0.13768
    HDc[2, 4] = 0.0106
    HDc[3, 0] = -0.07676
    HDc[3, 1] = 0.11864
    HDc[3, 2] = -0.07366
    HDc[3, 3] = 0.01759
    HDc[3, 4] = -0.001482
    HDc[4, 0] = 0.00275
    HDc[4, 1] = -0.00366
    HDc[4, 2] = 0.002514
    HDc[4, 3] = -0.00066631
    HDc[4, 4] = 0.000061926

    def nHD(R):
        return xHD * nH(R)

    def lambdaHD(Tg, R):
        hd_sum = sum(HDc[j, i] * np.log10(Tg)**i * np.log10(nH(R))**j for i in range(5) for j in range(5))
        return -(10**(hd_sum + np.log10(nHD(R)))) / (nH(R) * muH * mH)

    gammaturb = (3 * np.sqrt(3) / 2) * (userveldisp**3) / userR
    
    def gastemp(Tg, R):
        # for troubleshooting:
        #print('gastemp')
        #print(gammacompress(Tg, R))
        #print(lambdametal(Tg, R))
        #print(psigd(Tg, R))
        #print(lambdaH2(Tg, R))
        #print(gammaH23b(Tg, R))
        #print(gammacosmic(R))
        #print(gammaH2dust(Tg, R))
        #print(lambdaHD(Tg, R))
        #print(gammaturb)
        #print('--')
        return gammacompress(Tg, R) + lambdametal(Tg, R) + psigd(Tg, R) + lambdaH2(Tg, R) + gammaH23b(Tg, R) + gammacosmic(R) + gammaH2dust(Tg, R) + lambdaHD(Tg, R) + gammaturb

    #Print[Plot[Log10[Abs[gastemp[Tg,Rarray[[3]]]]],{Tg,100,180}]]
    #Tgasguess = Which[case\[Equal]1,90,case\[Equal]1.5,93,case\[Equal]2,97,case\[Equal]3,370,case\[Equal]3.5,370,case\[Equal]4,250]
    #change Tgasguess for -4 \[LessEqual] userlogZ < -2 to 166 when using remy ruyer
    #change Tgasguess for userlogZ >= -2 to 150 when using chem==4

    Tgasguess = None
    if userlogZ < -5 and chem <= 3:
        Tgasguess = 800
    elif -5 <= userlogZ < -4 and chem <= 3:
        Tgasguess = 370
    elif -4 <= userlogZ < -3 and chem <= 3:
        Tgasguess = 300
    elif -3 <= userlogZ < -2 and chem <= 3 and cosmicrays == 0:
        Tgasguess = 140
    elif -3 <= userlogZ < -2 and chem <= 3 and cosmicrays == 1:
        Tgasguess = 400
    elif userlogZ >= -2 and chem <= 3:
        Tgasguess = 110
    elif userlogZ < -4 and chem == 4:
        Tgasguess = 400
    elif -4 <= userlogZ < -2 and chem == 4:
        Tgasguess = 130
    elif userlogZ >= -2 and chem == 4:
        Tgasguess = 200
        
    Tgmax = None
    if chem <= 3:
        Tgmax = 2000
    elif chem == 4:
        Tgmax = 10000
        
    nex = n(Rarray)
    Tdustex = Tdustarray(Rarray)
    #NH2ex = NH2(Rarray)

    Tgas = []
    for r in Rarray:
        #try:
        Tg = optimize.root(lambda t: gastemp(t, r), x0=Tgasguess, method='linearmixing')
        Tgas.append(float(Tg.x))
        #except:
        #    Tg_arr = [gastemp(t, r) for t in np.arange(2.725,Tgmax,1.)]
        #    Tg = Tg_arr[np.argmin(Tg_arr)]
        #    print(Tg_arr)
        #    Tgas.append(Tg)
        
    Tgas = np.array(Tgas)
        
    gammacompressex = gammacompress(Tgas, Rarray)
    gammacosmicex = gammacosmic(Rarray)
    lambdametalex = [lambdametal(Tgas[i], Rarray[i]) for i in range(len(Rarray))]
    return Rarray, nex, Tdustex, Tgas

