#%%
# Import packages

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import pandas as pd
import astropy.units as units
import os # Used for os.system, consider pathlib for better practice
import warnings

from scipy import optimize
from scipy.interpolate import interp1d, NearestNDInterpolator

# Local imports - ensure these files (check_temp.py, constants.py,
# PDF_model_functions.py, emissivity_model_functions.py)
# are in your project directory or accessible via PYTHONPATH.
from check_temp import find_Tdust_new, gastemp
# Explicitly importing used constants from constants.py
from constants import mH, pc, G, kmpers, gcm2 # Note: 'mu' from constants.py is not used due to local redefinition.
from PDF_model_functions import (
    LN_width, PL_amplitude, normalization, lognormal, power_law,
    combine_PL_LN, transition_density, mean_log_density,
    mass_frac_above_crit, sound_speed, calc_scrit_PN11,
    calc_SFR_composite, calc_slopes, calc_fdense_Burk2019
)
from emissivity_model_functions import expectation_value_mass_weighted_linear
from check_temp import (
    gammacompress, lambdametal, psigd, lambdaH2, gammaH23b,
    gammacosmic, gammaH2dust, lambdaHD, gammaturb
)

# Matplotlib settings for LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.serif'] = 'Computer Modern Roman'
plt.rcParams['font.family'] = 'serif'

#####################
# MATCHING PAPER II #
#####################

#%%
def save_grid(df_out):
    # VARIABLE LIMITS
    temp_min = 10.
    temp_max = 300.
    alpha_min = 1.5
    alpha_max = 2.5
    min_mach = 2
    max_mach = 100
    nmin = 10.
    nmax = 1e8

    n_ref = 10**4.5
    b=0.5
    mu = 2.33 # This 'mu' is used, overriding any 'mu' from constants.py
    beta = np.inf

    Sigma_arr = df_out['Sigma'].values
    rs_arr = df_out['scale'].values/2.
    sigv_arr = df_out['Vdisp'].values
    avir_arr = df_out['alphavir'].values
    Galaxy = df_out['Galaxy'].values
    index_orig = df_out['Unnamed: 0'].values

    i=0
    f = open("model_grid.tsv", "w")
    # add:  sample
    f.write("Galaxy\tSigma\trs\tsigv\tavir\talpha\ttemp\tnsrc\tSigma_SFR\ttdep\teff\tindex\n")
    f.close()

    ## Extracting the relevant columns from COfile
    COfile = np.genfromtxt("grid_COcooling_new.out")
    log_nH = np.log10(COfile[:, 0])
    Tgas = COfile[:, 1]
    log_NCO = np.log10(COfile[:, 2])
    sigma_v = COfile[:, 3] / 1e5
    cooling_rate = COfile[:, 4]
    #optical_depth = COfile[:, 5]
    COcooling = NearestNDInterpolator(list(zip(log_nH,Tgas,log_NCO,sigma_v)), cooling_rate)

    for gal, Sigma, rs, sigv, avir, idx in zip(Galaxy, Sigma_arr, rs_arr, sigv_arr, avir_arr, index_orig):
        keep_criteria=False

        # Narayanan+2014
        Z = 1.
        # MASS OF SPHERE WITH CONSTANT DENSITY FROM SURFACE DENSITY, SIZE
        Mc_const = 4*np.pi*Sigma*units.M_sun/units.pc**2*(rs*units.pc)**2
        Mc_const = Mc_const.to(units.M_sun)
        rho_surface = Mc_const/(4/3*np.pi)/(rs*units.pc)**3
        rho_surface = rho_surface.to(units.kg/units.cm**3).value
        n_surface = rho_surface/(2.33*(const.m_p+const.m_e)).value
        Mc_const = Mc_const.value
        n_src = n_surface

        # these stay the same:
        userlogZ = 0
        Z = 10 ** userlogZ
        cosmicrays = 1
        remyruyer = 0
        H2formationheating = 1
        chem = 0
        coratio = 0
        cosmicCrocker = 0
        cosmicPadovani = 1 - cosmicCrocker

        # model inputs:
        kmpers = 1e5 # convert km/s to cm/s
        gcm2 = 0.000208836 # convert Msun / pc^2 to g / cm^2
        pc = 3.086*1e18 # convert pc to cm
        userSigma = Sigma * gcm2
        userveldisp = sigv * kmpers
        #useravir = avir
        useravir = 1.

        alpha = 2.
        kappa = 3./alpha

        # BASED ON Sharda+2023 MODEL
        #P = (3 / 20) * np.pi * useravir * G * userSigma**2
        #rho_ex = P / (userveldisp)**2 # edge density Sharda+2022
        #userR = (3 - kappa) * userSigma / (4 * rho_ex) # radius Sharda+2022
        #n_ex = rho_ex / (mH * mu)
        #fcorr = (3. - kappa) / 3.

        # BASED ON OBS
        rho_ex = rho_surface * units.kg.to(units.g)
        n_ex = rho_ex / (mH * mu)
        userR = rs * pc
        usern = n_ex

        ############
        # est temp #
        ############

        solution = find_Tdust_new(usern, userR, userlogZ, kappa, remyruyer, userSigma)
        Redge = userR
        Rhoe = rho_ex
        kRho = kappa
        Rch =  solution[2]
        kT = solution[3]
        Tdust = solution[4]
        loguserdelta = solution[5]

        # make coarse arrays
        minR = np.log10(0.001 * pc)
        maxR = np.log10(rs * pc)
        Rarray = np.logspace(minR,maxR,10)
        dens_H2 = n_ex*(Rarray/Redge)**-kRho

        # heating / cooling arrays
        gammacompress_arr = []
        gammaH23b_arr = []
        gammacosmic_arr = []
        gammaH2dust_arr = []
        gammaturb_arr = []
        lambdametal_arr = []
        lambdaH2_arr = []
        lambdaHD_arr = []
        psigd_arr = []

        # Initial guess
        Tgasguess = 30.

        Tgas = []
        for r, n  in zip(Rarray,dens_H2):

            # Corrected arguments for gastemp to match check_temp.py definition
            Tg = optimize.root(lambda t: gastemp(t, r, COcooling, Rhoe, Redge, kappa, coratio, chem, cosmicrays, cosmicCrocker, cosmicPadovani, userSigma, Tdust, kT, userveldisp, loguserdelta, H2formationheating, userR, usern), x0=Tgasguess, method='linearmixing')
            Tgas.append(float(Tg.x))

            # heating
            gammacompress_arr.append(gammacompress(Tg.x, r, Rhoe, Redge, kappa))
            gammaH23b_arr.append(gammaH23b(Tg.x, r, H2formationheating, chem, cosmicrays, cosmicCrocker, cosmicPadovani, userSigma, Rhoe, Redge, kRho, Tdust, kT))
            gammacosmic_arr.append(gammacosmic(r, cosmicrays, cosmicCrocker, cosmicPadovani, userSigma, Redge, kRho))
            gammaH2dust_arr.append(gammaH2dust(Tg.x, r, chem, cosmicrays, cosmicCrocker, cosmicPadovani, userSigma, Rhoe, Redge, kRho, Tdust, kT, H2formationheating))
            # Corrected arguments for gammaturb: now passing r, userR, and usern
            gammaturb_arr.append(gammaturb(userveldisp, r, userR, usern))

            # cooling
            lambdametal_arr.append(lambdametal(Tg.x, r, COcooling, coratio, userlogZ, chem,Rhoe,Redge,kappa,cosmicrays,cosmicCrocker,cosmicPadovani,userSigma, Tdust, kT, userveldisp))
            lambdaH2_arr.append(lambdaH2(Tg.x, r, chem, cosmicrays, cosmicCrocker, cosmicPadovani, userSigma, Rhoe, Redge, kappa, Tdust, kT))
            lambdaHD_arr.append(lambdaHD(Tg.x, r,Rhoe,Redge,kRho))
            psigd_arr.append(psigd(Tg.x, r,chem, cosmicrays, cosmicCrocker, cosmicPadovani, userSigma, Rhoe, Redge, kRho, Tdust, kT, loguserdelta))

        Tgas_arr = np.array(Tgas)
        Tmin = np.nanmin(Tgas_arr)
        ind = np.argmin(Tgas_arr)
        nmin = dens_H2[ind]
        ind = dens_H2 < nmin
        Tgas_arr[ind] = Tmin
        Tgas_arr[Tgas_arr < 2.73] = 2.73
        Tgas_interp = interp1d(np.log10(dens_H2), Tgas_arr, kind=1, fill_value='extrapolate')

        # ESTIMATE MEAN TEMP
        temp = 30.
        temp_last = temp
        dens_last = n_src
        dt = 999
        niter = 0
        maxniter = 100
        alpha_s = 3./kappa

        N_src = (userSigma / (mu *mH))
        n_src = n_surface
        while (dt > 0.001) & (niter < maxniter):

            c_s = sound_speed(temp_last)
            mach = sigv/c_s

            # PDF
            sigma_s = LN_width(b, mach)
            sigma_n = b**2*mach**2
            #sigma_n = np.sqrt(np.exp(sigma_s**2)-1)
            s_t = transition_density(alpha_s,sigma_s)
            s_0 = mean_log_density(sigma_s)
            Rarray = np.logspace(minR,maxR,500)
            n_array = n_src*(Rarray/Redge)**-kRho
            s_array = np.log(n_array / n_src)
            n_t = n_src*np.exp(s_t)

            C = PL_amplitude(alpha_s,sigma_s)
            N = normalization(C,s_t,alpha_s,sigma_s)
            LN = lognormal(N,sigma_s,s_array,s_0)
            PL = power_law(N,C,alpha_s,s_array)
            PDF = combine_PL_LN(s_t,PL,LN,s_array)

            Tgas_arr = Tgas_interp(np.log10(n_array))
            Tgas_arr[Tgas_arr < 2.73] = 2.73
            temp = expectation_value_mass_weighted_linear(n_array, Tgas_arr, PDF/(n_array))
            n_pdf = expectation_value_mass_weighted_linear(n_array, n_array, PDF/(n_array))

            c_s = sound_speed(temp)
            mach = sigv/c_s
            fcorr = 1

            dt = abs(1 - temp_last/temp)
            dn = abs(1 - dens_last/n_src)
            dens_last = n_src
            temp_last = temp

            niter =  niter+1

        # Define LARGE ARRAYS
        Rarray = np.logspace(minR,maxR,500)
        n_array = n_src*(Rarray/(rs * pc))**-kRho
        N_array = N_src*(Rarray/(rs * pc))**-(kRho-1)
        s_array = np.log(n_array / n_src)
        eta = np.log(N_array/N_src)

        #################
        # INTERPOLATION #
        #################

        # HEATING
        gammacompress_interp = interp1d(np.log10(dens_H2), gammacompress_arr, kind=1, fill_value='extrapolate')
        gammaH23b_interp = interp1d(np.log10(dens_H2), gammaH23b_arr, kind=1, fill_value='extrapolate')
        gammacosmic_interp = interp1d(np.log10(dens_H2), gammacosmic_arr, kind=1, fill_value='extrapolate')
        gammaH2dust_interp = interp1d(np.log10(dens_H2), gammaH2dust_arr, kind=1, fill_value='extrapolate')
        gammaturb_interp = interp1d(np.log10(dens_H2), gammaturb_arr, kind=1, fill_value='extrapolate')

        gammacompress_arr = gammacompress_interp(np.log10(n_array))
        gammaH23b_arr = gammaH23b_interp(np.log10(n_array))
        gammacosmic_arr = gammacosmic_interp(np.log10(n_array))
        gammaH2dust_arr = gammaH2dust_interp(np.log10(n_array))
        gammaturb_arr = gammaturb_interp(np.log10(n_array))

        # COOLING + GAS-DUST EXCHANGE
        lambdametal_interp = interp1d(np.log10(dens_H2), lambdametal_arr, kind=1, fill_value='extrapolate')
        lambdaH2_interp = interp1d(np.log10(dens_H2), lambdaH2_arr, kind=1, fill_value='extrapolate')
        lambdaHD_interp = interp1d(np.log10(dens_H2), lambdaHD_arr, kind=1, fill_value='extrapolate')
        psigd_interp = interp1d(np.log10(dens_H2), psigd_arr, kind=1, fill_value='extrapolate')

        lambdametal_arr = lambdametal_interp(np.log10(n_array))
        lambdaH2_arr = lambdaH2_interp(np.log10(n_array))
        lambdaHD_arr = lambdaHD_interp(np.log10(n_array))
        psigd_arr = psigd_interp(np.log10(n_array))

        #####################################
        # PLOT HEATING / COOLING BY DENSITY #
        #####################################

        size = plt.figaspect(1.2)
        fig = plt.figure(figsize=size/1.5)
        #grid = gs.GridSpec(1,2) # This line was commented out, and gs was unused.
        ax = fig.add_subplot(111)
        # Heating
        ax.plot(np.log10(n_array),np.log10(np.array(gammacompress_arr)),label=r'$\Gamma_\mathrm{compress}$')
        ax.plot(np.log10(n_array),np.log10(np.array(gammaH23b_arr)),label=r'$\Gamma_\mathrm{H23b}$')
        ax.plot(np.log10(n_array),np.log10(np.array(gammacosmic_arr)),label=r'$\Gamma_\mathrm{cosmic}$')
        ax.plot(np.log10(n_array),np.log10(np.array(gammaH2dust_arr)),label=r'$\Gamma_\mathrm{H2dust}$')
        ax.plot(np.log10(n_array),np.log10(np.array(gammaturb_arr)),label=r'$\Gamma_\mathrm{turb}$')
        # Cooling
        ax.plot(np.log10(n_array),np.log10(abs(np.array(lambdametal_arr))),label='$\Lambda_\mathrm{metal}$',ls='dashed')
        ax.plot(np.log10(n_array),np.log10(abs(np.array(lambdaH2_arr))),label='$\Lambda_\mathrm{H2}$',ls='dashed')
        ax.plot(np.log10(n_array),np.log10(abs(np.array(lambdaHD_arr))),label='$\Lambda_\mathrm{HD}$',ls='dashed')
        ax.plot(np.log10(n_array),np.log10(abs(np.array(psigd_arr))),label='$\Psi_\mathrm{gd}$',ls='dotted')
        # Format plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r'log $n_{H_2}$ (cm$^-3$)')
        ax.set_ylabel(r'$\Lambda\,\mathrm{or}\,\Gamma$')
        plt.savefig('figures/models/heating_cooling{}.png'.format(i),bbox_inches='tight')
        plt.close()

        # SOUND SPEED & MACH NUMBER
        c_s = sound_speed(temp)
        mach = sigv/c_s
        alpha_s, alpha_N, alpha_n, alpha_eta = calc_slopes(kappa)

        # 3D + 2D - LOG + LINEAR STD
        sigma_s = LN_width(b, mach)
        #sigma_n = b**2*mach**2 # This line was commented out
        sigma_n = np.sqrt(np.exp(sigma_s**2)-1)
        #fcorr = 2./sigma_n # This line was commented out
        A = 0.11
        sigma_eta = np.sqrt(A)*sigma_s
        sigma_N = np.sqrt(np.exp(sigma_eta**2))

        # CHARACTERISTIC DENSITIES
        N_surface = userSigma / (mu * mH)
        Mc = Mc_const/fcorr

        # TRANSITION DENSITY
        eta_t = 0.5*(2*abs(alpha_eta)-1)*sigma_eta**2
        N_t = N_src*np.exp(eta_t)
        s_t = transition_density(alpha_s,sigma_s)
        n_t = n_src*np.exp(s_t)
        r_t = (N_t/n_t)*units.cm.to(units.pc)

        ##################
        # CONSTRUCT PDFS #
        ##################

        # 3D
        s_0 = mean_log_density(sigma_s)
        C = PL_amplitude(alpha_s,sigma_s)
        N = normalization(C,s_t,alpha_s,sigma_s)
        LN = lognormal(N,sigma_s,s_array,s_0)
        PL = power_law(N,C,alpha_s,s_array)
        pdf = combine_PL_LN(s_t,PL,LN,s_array)

        # 2D
        eta_0 = -0.5*sigma_eta**2
        C_N = PL_amplitude(alpha_eta,sigma_eta)
        N_N = normalization(C_N,eta_t,alpha_eta,sigma_eta)
        LN_N = lognormal(N_N,sigma_eta,eta,eta_0)
        PL_N = power_law(N_N,C_N,alpha_eta,eta)
        PDF_N = combine_PL_LN(eta_t,PL_N,LN_N,eta)

        n_ref = 10**4.5
        s_crit = calc_scrit_PN11(mach,np.inf,avir)
        s_fix = np.log(n_ref/n_src)
        fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)


        s_array_large = np.linspace(-10,20,500)
        LN_large = lognormal(N,sigma_s,s_array_large,s_0)
        PL_large = power_law(N,C,alpha_s,s_array_large)
        PDF_large = combine_PL_LN(s_t,PL_large,LN_large,s_array_large)
        ffix = mass_frac_above_crit(s_fix,PDF_large,s_array_large)
        f_t = mass_frac_above_crit(s_t,PDF_large,s_array_large)

        pdf_log = np.log10(PDF/n_array)
        pdf_max = np.nanmax(pdf_log)
        pdf_min = np.nanmin(pdf_log)
        pdf_norm = (pdf_log - pdf_min)/(pdf_max - pdf_min)

        # PLOT TEMP & PDF
        plt.clf()
        plt.plot(np.log10(dens_H2),Tgas)
        plt.plot(np.log10(n_array),Tgas_arr)
        plt.plot(np.log10(n_array),(pdf_norm * 200),color='gray',alpha=0.5,ls='solid',label=r'Scaled PDF')
        plt.vlines(np.log10(n_t),0,250,color='gray',alpha=0.5,ls='dashed',label=r'$n_t$')
        plt.vlines(np.log10(n_src),0,250,color='gray',alpha=0.5,ls='dotted',label=r'$n_0$')
        plt.ylim(0,250)
        plt.xlim(np.nanmin(np.log10(n_array)),np.nanmax(np.log10(n_array)))
        plt.annotate(text = 'Tdust = {:.2f}'.format(Tdust), xy=(0.01,0.95),xycoords='axes fraction')
        plt.annotate(text = 'Tgas = {:.2f}'.format(temp), xy=(0.01,0.9),xycoords='axes fraction')
        plt.annotate(text = r'log $n_0$ = {:.2f}'.format(np.log10(n_src)), xy=(0.01,0.85),xycoords='axes fraction')
        plt.xlabel(r'log $n_{H_2}$ (cm$^-3$)')
        plt.ylabel(r'$T_\mathrm{gas}$ (K)')
        plt.legend(loc='upper right')
        plt.savefig('figures/models/temp_vs_dens_pdf{}.png'.format(i),bbox_inches='tight')
        plt.close()

        ##############################
        # SAVING PDF(S) TO DATAFRAME #
        ##############################

        df_pdf = pd.DataFrame()
        # DENSITY ARRAYS
        df_pdf['s_array'] = s_array
        df_pdf['eta'] = eta
        df_pdf['log n_arr'] = np.log10(n_array)
        df_pdf['log N_arr'] = np.log10(N_array)
        # RADII
        df_pdf['r_arr_pc'] = Rarray / pc

        # 3D PDF MODEL ARRAYS
        df_pdf['LN'] = LN
        df_pdf['PL'] = PL
        df_pdf['PDF'] = PDF
        # 2D PDF MODEL ARRAYS
        df_pdf['LN_eta'] = LN_N
        df_pdf['PL_eta'] = PL_N
        df_pdf['PDF_eta'] = PDF_N

        # SLOPES
        df_pdf['kappa'] = kappa
        df_pdf['alpha_s'] = alpha_s
        df_pdf['alpha_N'] = alpha_N
        df_pdf['alpha_n'] = alpha_n
        df_pdf['alpha_eta'] = alpha_eta

        # PDF INPUTS AND NORMALIZATION CONSTANTS
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = sigv
        df_pdf['mach'] = mach
        df_pdf['N_surface'] = N_surface
        df_pdf['N_src'] = N_src
        df_pdf['n_src'] = n_src
        df_pdf['Mc_Msun'] = Mc_const
        # 3D PDF
        df_pdf['sigma_s'] = sigma_s
        df_pdf['sigma_n'] = sigma_n
        df_pdf['s_0'] = s_0
        df_pdf['C norm'] = C
        df_pdf['N norm'] = N
        # 2D PDF
        df_pdf['eta_0'] = eta_0
        df_pdf['N_N norm'] = C_N
        df_pdf['N_N norm'] = N_N
        df_pdf['sigma_eta'] = sigma_eta
        df_pdf['sigma_N'] = sigma_N
        # FRACTIONS
        df_pdf['fcrit'] = fcrit
        df_pdf['f_above_fix'] = ffix
        df_pdf['f_t'] = f_t
        # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
        # EVEN CALCULATED FOR LN-ONLY MODELS
        df_pdf['eta_t'] = eta_t
        df_pdf['s_t'] = s_t
        df_pdf['N_t cm-2'] = N_t
        df_pdf['log n_t'] = np.log10(n_t)
        df_pdf['r_t pc'] = r_t
        # DENSITY CORRECTION
        df_pdf['fcorr'] = fcorr
        # CRITICAL DENSITY
        df_pdf['s_crit'] = s_crit

        # VIRIAL PARAMETER
        df_pdf['alpha_vir'] = avir

        # SOURCE SIZE
        df_pdf['rs_pc'] = rs

        # TEMP
        df_pdf['temp_mean'] = temp
        df_pdf['temp'] = Tgas_arr
        df_pdf['temp_dust'] = Tdust
        df_pdf['c_s'] = c_s

        # HEATING
        df_pdf['gammacompress'] = np.array(gammacompress_arr)
        df_pdf['gammaH23b'] = np.array(gammaH23b_arr)
        df_pdf['gammacosmic'] = np.array(gammacosmic_arr)
        df_pdf['gammaH2dust'] = np.array(gammaH2dust_arr)
        df_pdf['gammaturb'] = np.array(gammaturb_arr)

        # COOLING + GAS-DUST EXCHANGE
        df_pdf['lambdametal'] = np.array(lambdametal_arr)
        df_pdf['lambdaH2'] = np.array(lambdaH2_arr)
        df_pdf['lambdaHD'] = np.array(lambdaHD_arr)
        df_pdf['psigd'] = np.array(psigd_arr)

        df_pdf.to_csv('output/pdf{:.0f}.tsv'.format(i),sep='\t')

        s_t = transition_density(alpha,sigma_s)
        eff = calc_SFR_composite(sigma_s, alpha, s_t, 0.05) # alpha = 2.
        sigma_n = np.sqrt(np.exp(sigma_s**2)-1)
        A = 0.11
        sigma_eta = np.sqrt(A)*sigma_s
        sigma_N = np.sqrt(np.exp(sigma_eta**2)-1)
        ks = sigma_n/(sigma_N+sigma_n)

        rho = n_src * mu * mH * units.g / units.cm**3
        tff = (np.sqrt(3.*np.pi/(32*const.G))*np.sqrt(1/(rho))).to(units.yr)
        Sigma_SFR = (eff*(Sigma*units.M_sun/units.pc**2)/tff).to(units.M_sun/units.kpc**2/units.yr)
        tdep = ((Sigma*units.M_sun/units.pc**2)/Sigma_SFR).to(units.Myr)

        temp_criteria = (temp > temp_min) & (temp < temp_max)
        dens_criteria = (n_src > nmin) & (n_src < nmax)
        tdep_criteria = (tdep > 1e-1*units.Myr) & (tdep < 1e10*units.Myr)
        sfr_criteria = (Sigma_SFR > 10**-4*units.M_sun/units.kpc**2/units.yr) & (Sigma_SFR < 10**5*units.M_sun/units.kpc**2/units.yr)
        eff_criteria = (eff > 0.001) & (eff < 1.)
        keep_criteria = True #tdep_criteria & sfr_criteria & eff_criteria & temp_criteria #& dens_criteria

        print(gal,rs,temp,n_src,sigv,alpha,Sigma_SFR.to(units.M_sun/units.kpc**2/units.yr).value,tdep.to(units.Myr).value,eff)

        if keep_criteria:
            f = open("model_grid.tsv", "a")
            # add:  sample
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(gal,Sigma, rs, sigv, avir, alpha,temp,n_src,Sigma_SFR.to(units.M_sun/units.kpc**2/units.yr).value,tdep.to(units.Myr).value,eff,idx))
            f.close()

            print('{} Success'.format(i))

        else:
            print(rs,temp,n_src,alpha,Sigma_SFR.to(units.M_sun/units.kpc**2/units.yr).value,tdep.to(units.Myr).value,eff)
            print('Skip {}'.format(i))

        i=i+1

#%%
df_out = pd.read_csv('model_subset300.csv')[:5]

os.system('rm output/pdf*')
os.system('rm figures/models/cooling*')
os.system('rm figures/models/heating*')
os.system('rm figures/models/temp*')
save_grid(df_out)
