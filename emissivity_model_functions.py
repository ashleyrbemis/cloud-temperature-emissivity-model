from glob import glob
from astropy.io import ascii
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd
import sys
from timeit import default_timer as timer
import astropy.constants as const
import astropy.units as u
import scipy.stats as stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from PDF_model_functions import sound_speed, LN_width, PL_amplitude, transition_density, normalization, mean_log_density, power_law, lognormal, combine_PL_LN, calc_fdense_Burk2019,LN_width_alfven, conservation_shift
from PDF_model_functions import get_PDFs, calc_avir, calc_scrit_KM05, calc_SFR_composite, calc_SFR_composite_multiff, mass_frac_above_crit, calc_SFR_lognormal_PN11, calc_SFR_lognormal_multiff, calc_scrit_PN11, find_effective_ex_density, expectation_value_mass_weighted, expectation_value_mass_weighted_linear

# FINE
def parse_radex_table(tab_radex):
    co_out = tab_radex.loc[tab_radex['mol'] == 'co']
    hcn_out = tab_radex.loc[tab_radex['mol'] == 'hcn']
    hcop_out = tab_radex.loc[tab_radex['mol'] == 'hco+']
    return co_out, hcn_out, hcop_out

def zero_to_nan(array):
    array[array == 0.0] = np.nan
    return array

def parse_4lines(new_tab):
    jup = []
    for j in new_tab['JUP'].values:
        try:
            jup.append(np.int(j))
        except:
            jup.append(-100)
    jup = np.array(jup,dtype='int')
    l10 = new_tab.loc[jup == 1]
    l21 = new_tab.loc[jup == 2]
    l32 = new_tab.loc[jup == 3]
    l43 = new_tab.loc[jup == 4]
    return l10, l21, l32, l43

def parse_1line(new_tab,jup):
    jup_arr = pd.to_numeric(new_tab['JUP'].values,errors='coerce')
    l10 = new_tab.loc[jup_arr == jup]
    return l10

def read_radex_1line(tab_radex):
    
    # J = 1-0 lines
    flux_co = tab_radex['FLUX_Kkm/s_co'].values
    flux_hcn = tab_radex['FLUX_Kkm/s_hcn'].values
    flux_hcop = tab_radex['FLUX_Kkm/s_hco+'].values
    tex_co = tab_radex['T_EX_K_co'].values
    tex_hcn = tab_radex['T_EX_K_hcn'].values
    tex_hcop = tab_radex['T_EX_K_hco+'].values
    tau_co = tab_radex['TAU_co'].values
    tau_hcn = tab_radex['TAU_hcn'].values
    tau_hcop = tab_radex['TAU_hco+'].values
    popup_co = tab_radex['POP_UP_co'].values
    popup_hcn = tab_radex['POP_UP_hcn'].values
    popup_hcop = tab_radex['POP_UP_hco+'].values
    
    out = (flux_co,
           flux_hcn,
           flux_hcop,
           tex_co,
           tex_hcn,
           tex_hcop,
           tau_co,
           tau_hcn,
           tau_hcop,
           popup_co,
           popup_hcn,
           popup_hcop)
    
    return out

def calc_beta(tau):
    return (1.5/tau)*(1 - 2/tau**2 + (2/tau + 2/tau**2)*np.exp(-tau))

def calc_XHCN_pop(pop_up,tau):
    beta = calc_beta(tau)
    return 6.3/4*1e19/beta/pop_up

def calc_XCO_pop(pop_up,tau):
    beta = calc_beta(tau)
    return 1.79*1e18/beta/pop_up

def calc_X_HCN(tex,density):
    density = density/1e3
    return np.array(2.05*1e20*np.sqrt(density)*(np.exp(4.25362/tex)-1.))


def calc_X_HCN_corr(tex,density,alpha_vir,tau):
    density = density/1e3
    escape_part = 1/(2/tau + 1)
    corr = np.sqrt(alpha_vir)/escape_part
    return np.array(2.05*1e20*np.sqrt(density)*(np.exp(4.25362/tex)-1.))*corr

def calc_X_CO(tex,density):
    density = density/1e3
    return np.array(1.58*1e20*np.sqrt(density)*(np.exp(5.53211/tex) - 1.))

def calc_X_CO_corr(tex,density,alpha_vir,tau):
    density = density/1e3
    escape_part = 1/(2/tau + 1)
    corr = np.sqrt(alpha_vir)/escape_part
    return np.array(1.58*1e20*np.sqrt(density)*(np.exp(5.53211/tex)-1.))*corr

def alpha_mol(X):
    return X*u.pc.to(u.cm)**2*(2.33*(const.m_p+const.m_e))/(u.M_sun.to(u.kg))/u.kg

def radex_file_base(N_,t_,dv_):
    return 'NH2{:.2e}cm-2_temp{:.0f}K_dv{:.1f}kms'.format(N_,t_,dv_).replace('.','p')

def format_tau(tau):
    i = np.argwhere(tau > 100)
    if len(i) > 100:
        tau[i] = np.nan
    return tau
        
    
def remove_outliers(df):

    # mask tex > tkin
    tex_hcn = df['T_EX_K_hcn'].values
    tex_co = df['T_EX_K_hcn'].values
    tau_hcn = df['TAU_hcn'].values
    tau_co = df['TAU_hcn'].values
    tkin = df['temp_K'].values[0]

    df.loc[tex_hcn < 0] = np.nan
    df.loc[tex_co < 0] = np.nan
    df.loc[tau_hcn <= 0] = np.nan
    df.loc[tau_co <= 0] = np.nan
    #df.loc[tex_hcn > tkin] = np.nan
    #df.loc[tex_co > tkin] = np.nan

    hcn = df['FLUX_Kkm/s_hcn'].values/10**df['log N_arr'].values
    co = df['FLUX_Kkm/s_hcn'].values/10**df['log N_arr'].values

    ind = np.log10(hcn) > -20
    df.loc[ind] = np.nan
    ind = np.log10(co) > -19
    df.loc[ind] = np.nan
    
    return df

def zscore_chunk(df,cut=3.):

    df_ = df.apply(np.log10)
    df_ = df_.apply(stats.zscore)
    df_ = df_.apply(lambda x: x - np.roll(x,-1))
    dn = df_['nH2_cm-3'].values

    co = df_['FLUX_Kkm/s_co'].values
    hcn = df_['FLUX_Kkm/s_hcn'].values
    hcop = df_['FLUX_Kkm/s_hcop'].values
    #print(co)

    zco = np.nan_to_num(stats.zscore(co))
    zhcn = np.nan_to_num(stats.zscore(hcn))
    zhcop = np.nan_to_num(stats.zscore(hcop))

    i = (abs(zco/np.nanmedian(zco)) < cut) & (abs(zhcn/np.nanmedian(zhcn)) < cut) & (abs(zhcop/np.nanmedian(hcop)) < cut)
    
    return df.loc[i]


def isolation_forest(df_,n_jobs=6):
    
    X = np.log10(df_['FLUX_Kkm/s_hcn'].values)
    i0 = ~np.isnan(X) & ~np.isinf(X)
    df_ = df_.loc[i0]
    X = X[i0].reshape(-1,1)
    clf = IsolationForest(n_jobs=n_jobs,bootstrap=True).fit(X)
    y_pred = clf.predict(X)
    in_hcn = y_pred != -1
    df_ = df_.loc[in_hcn]

    X = np.log10(df_['FLUX_Kkm/s_co'].values)
    i1 = ~np.isnan(X) & ~np.isinf(X)
    df_ = df_.loc[i1]
    X = X[i1].reshape(-1,1)
    clf = IsolationForest(n_jobs=n_jobs,bootstrap=True).fit(X)
    y_pred = clf.predict(X)
    in_co = y_pred != -1
    df_ = df_.loc[in_co]

    X = np.log10(df_['FLUX_Kkm/s_hcop'].values)
    i2 = ~np.isnan(X) & ~np.isinf(X)
    X = X[i2].reshape(-1,1)
    df_ = df_.loc[i2]
    clf = IsolationForest(n_jobs=n_jobs,bootstrap=True).fit(X)
    y_pred = clf.predict(X)
    in_hcop = y_pred != -1
    df_ = df_.loc[in_hcop]
    
    return df_

def calc_emiss_and_SF_props(df_pdf, df_radex,corr=1.,xco=1e-4,xhcn=1e-8,xhcop=1e-8,
                            lognormal=False,beta=np.inf,jup=4,tol_dex=0.1,remove_out=False):

    n_arr = df_radex['nH2_cm-3'].values
    df_radex['log n_arr'] = np.log10(n_arr)
    df_radex = df_radex.sort_values('log n_arr')
    df = pd.merge_asof(df_pdf,df_radex,on='log n_arr',tolerance=tol_dex) # ,how='outer
    df = df.dropna(axis=0)
    df = df.apply(pd.to_numeric,errors='coerce')

    if remove_out==True:
        df = remove_outliers(df)

    if len(df) == 0:
        return

    l1 = parse_1line(df,1)
    cols = [c_ for c_ in l1.columns if np.all([s_ not in c_ for s_ in ['Unnamed']])]
    l1 = l1[cols]
    
    try:
        l2 = parse_1line(df,2)
        l2 = l2[cols]
    except:
        pass
    try:
        l3 = parse_1line(df,3)
        l3 = l3[cols]
    except:
        pass
    try:
        l4 = parse_1line(df,4)
        l4 = l4[cols]
    except:
        pass
    
    if (len(l1) == 0):
        return

    l1_out = funcs_one_transition(l1,1,ext='10',corr=corr,
                                  xco=xco,xhcn=xhcn,xhcop=xhcop,
                                  lognorm=lognormal)

    try:
        l2_out = funcs_one_transition(l2,2,ext='21',
                                      xco=xco,xhcn=xhcn,xhcop=xhcop,
                                      lognorm=lognormal)
    except:
        #print('pass 2')
        pass

    try:
        l3_out = funcs_one_transition(l3,3,ext='32',corr=corr,
                                      xco=xco,xhcn=xhcn,xhcop=xhcop,
                                      lognorm=lognormal)
        #print('pass 3')
    except:
        pass

    try:
        l4_out = funcs_one_transition(l4,4,ext='43',corr=corr,
                                      xco=xco,xhcn=xhcn,xhcop=xhcop,
                                      lognorm=lognormal)
        #print('pass 4')
    except:
        pass

    try:
        new_tab = l1_out.copy()
    except:
        return None
    
    try:
        new_tab = pd.merge(l1_out,l2_out,how='outer') # l1_out.append(l2_out)
    except:
        pass

    try:
        new_tab = pd.merge(new_tab,l3_out,how='outer') #new_tab.append(l3_out)
    except:
        pass

    try:
        new_tab = pd.merge(new_tab,l4_out,how='outer')
    except:
        pass

    #new_tab.to_csv(outf,sep='\t')
    
    return new_tab
    

def funcs_one_transition(l1,jup,ext='',xco=1.4e-4,xhcn=1e-8,xhcop=1e-8,corr=1.,
                         lognorm=False,ff_lim=0.99):

    (flux_co,flux_hcn,flux_hcop,\
     tex_co,tex_hcn,tex_hcop,\
     tau_co,tau_hcn,tau_hcop,\
     popup_co,popup_hcn,popup_hcop) = read_radex_1line(l1)

    n_array = 10**l1['log n_arr'].values
    N_array = 10**l1['log N_arr'].values
    #l1.loc[n_array > 1e8] = np.nan
    NH2 = l1['Nmol_cm-2_co'].values/xco
    hcn =  l1['FLUX_Kkm/s_hcn'].values
    co =  l1['FLUX_Kkm/s_co'].values
    
    s_array = l1['s_array'].values
    r_arr = l1['r_arr_pc'].values
    #r_norm = l1['r_norm_pc'].values[0]
    #nnorm = l1['n_norm cm-3'].values[0]
    r_s = l1['rs_pc'].values[0]
    s_t = l1['s_t'].values[0]
    sigma_s = l1['sigma_s'].values[0]
    mach = l1['mach'].values[0]
    alpha_vir = l1['alpha_vir'].values[0]

    s_0 = l1['s_0'].values[0]
    n_0 = l1['n_src'].values[0]
    N_0 = l1['N_src'].values[0]
    n_t = n_0*np.exp(s_t)
    dn = np.roll(n_array,-1)-n_array
    l1['alpha_vir'] = alpha_vir
    rho_src = n_0*2.33*(const.m_p+const.m_e)*u.cm**-3

    emissivity_co_weighted = flux_co/NH2
    emissivity_hcn_weighted = flux_hcn/NH2
    emissivity_hcop_weighted = flux_hcop/NH2

    if lognorm == True:
        s_crit = np.exp(10**4.5/n_0)
        n_crit = n_0*np.exp(s_crit)
        pdf_weight = l1['LN'].values
        fdense = mass_frac_above_crit(s_crit,pdf_weight,s_array)
        #SFR_ff = 0.01*calc_SFR_lognormal(sigma_s, s_crit, eff = corr)
        #SFR_mff = 0.01*calc_SFR_lognormal_multiff(sigma_s, s_crit, eff = corr)
        SFR_ff = calc_SFR_lognormal(sigma_s, s_crit, eff = corr)
        SFR_mff = calc_SFR_lognormal_multiff(sigma_s, s_crit, eff = corr)
        l1['pdf_LN'] = pdf_weight
        l1['s_crit_LN'] = s_crit
        l1['n_crit_LN'] = n_crit
        l1['SFR_ff_LN'] = SFR_ff
        l1['SFR_mff_LN'] = SFR_mff
        l1['fd_LN'] = fdense
        #print('test')
        icrit = np.argmin(abs(n_array - n_crit))
        r_crit = r_arr[icrit]
        l1['r_crit_pc_LN'] = r_crit

        
        # INTEGRATING MOLECULAR EMISSIVITIES
        co_integrated = expectation_value_mass_weighted(s_array, emissivity_co_weighted, pdf_weight) #,i0=ico)
        hcn_integrated = expectation_value_mass_weighted(s_array, emissivity_hcn_weighted,pdf_weight) #,i0=ihcn)
        hcop_integrated = expectation_value_mass_weighted(s_array, emissivity_hcop_weighted,pdf_weight) #,i0=ihcop)
        
        # EFFECTIVE EXCITATION DENSITY
        ico, seff_co = find_effective_ex_density(s_array,emissivity_co_weighted*pdf_weight*np.exp(s_array),lim=ff_lim)
        ihcn, seff_hcn = find_effective_ex_density(s_array,emissivity_hcn_weighted*pdf_weight*np.exp(s_array),lim=ff_lim)
        ihcop, seff_hcop = find_effective_ex_density(s_array,emissivity_hcop_weighted*pdf_weight*np.exp(s_array),lim=ff_lim)
        icld,seff_tot = find_effective_ex_density(s_array, pdf_weight*np.exp(s_array),lim=ff_lim)
        
        neff_co = n_0*np.exp(seff_co)
        neff_hcn = n_0*np.exp(seff_hcn)
        neff_hcop = n_0*np.exp(seff_hcop)
        
        # r-weight
        co_r = r_arr[ico]
        hcn_r = r_arr[ihcn]
        hcop_r = r_arr[ihcop]
        r_cld_LN = r_arr[icld]
        
        # tau
        #tau_co  = format_tau(tau_co)
        #tau_hcn  = format_tau(tau_hcn)
        #tau_hcop  = format_tau(tau_hcop)
        
        co_tau = expectation_value_mass_weighted(s_array, tau_co, pdf_weight*emissivity_co_weighted)
        hcn_tau = expectation_value_mass_weighted(s_array, tau_hcn, pdf_weight*emissivity_hcn_weighted)
        hcop_tau = expectation_value_mass_weighted(s_array, tau_hcop, pdf_weight*emissivity_hcop_weighted)
        
        # tex
        co_tex = expectation_value_mass_weighted(s_array, tex_co, pdf_weight*emissivity_co_weighted) #,i0=ico)
        hcn_tex = expectation_value_mass_weighted(s_array, tex_hcn, pdf_weight*emissivity_hcn_weighted)#,i0=ihcn)
        hcop_tex = expectation_value_mass_weighted(s_array, tex_hcop, pdf_weight*emissivity_hcop_weighted)#,i0=ihcop)
        
        l1['neff'+ext+'_hcn_LN'] = neff_hcn
        l1['neff'+ext+'_hco+_LN'] = neff_hcop
        l1['neff'+ext+'_co_LN'] = neff_co
        
        l1['emiss'+ext+'_hcn_K_kms_cm-2_LN'] = hcn_integrated
        l1['emiss'+ext+'_hco+_K_kms_cm-2_LN'] = hcop_integrated
        l1['emiss'+ext+'_co_K_kms_cm-2_LN'] = co_integrated
        
        l1['tau'+ext+'_hcn_weight_LN'] = hcn_tau
        l1['tau'+ext+'_hco+_weight_LN'] = hcop_tau
        l1['tau'+ext+'_co_weight_LN'] = co_tau
        
        l1['tex'+ext+'_hcn_weight_LN'] = hcn_tex
        l1['tex'+ext+'_hco+_weight_LN'] = hcop_tex
        l1['tex'+ext+'_co_weight_LN'] = co_tex
        
        l1['r'+ext+'_hcn_LN'] = hcn_r
        l1['r'+ext+'_hco+_LN'] = hcop_r
        l1['r'+ext+'_co_LN'] = co_r
        l1['r_cloud_LN'] = r_cld_LN
        
        l1['i'+ext+'_hcn_LN'] = ihcn
        l1['i'+ext+'_hco+_LN'] = ihcop
        l1['i'+ext+'_co_LN'] = ico
        
        l1['ff'+ext+'_LN'] = hcn_r**2/co_r**2
        l1['ff_co_sfr'+ext+'_LN'] = co_r**2/r_crit**2

    s_crit = calc_scrit_PN11(mach,np.inf,  alpha_vir)
    n_crit = n_0*np.exp(s_crit)
    
    fdense = l1['fcrit'].values[0]
    alpha = l1['alpha_s'].values[0]
    pdf_weight = l1['PDF'].values/n_array
    #SFR_ff = 0.01*calc_SFR_composite(sigma_s, alpha, s_t, eff = 1)
    #SFR_mff = 0.01*calc_SFR_composite_multiff(sigma_s, alpha, s_t, eff = 1)
    SFR_ff = calc_SFR_composite(sigma_s, alpha, s_t, eff = corr)
    SFR_mff = calc_SFR_composite_multiff(sigma_s, alpha, s_t, eff = corr)
    l1['pdf_norm'] = pdf_weight
    l1['s_crit_KM'] = s_crit
    l1['n_crit_KM'] = n_crit
    l1['SFR_ff'] = SFR_ff
    l1['SFR_mff'] = SFR_mff
    l1['fd'] = fdense
    icrit = np.argmin(abs(n_array - n_crit))
    r_crit = r_arr[icrit]
    l1['r_crit_pc'] = r_crit
    kappa = l1['kappa'].values[0]
    try:
        ks = l1['ks'].values[0]
    except:
        ks = kappa
    
        
    # INTEGRATING MOLECULAR EMISSIVITIES
    co_integrated = expectation_value_mass_weighted_linear(n_array, emissivity_co_weighted, pdf_weight)
    hcn_integrated = expectation_value_mass_weighted_linear(n_array, emissivity_hcn_weighted, pdf_weight)
    hcop_integrated = expectation_value_mass_weighted_linear(n_array, emissivity_hcop_weighted, pdf_weight)

    l1['emiss'+ext+'_hcn_K_kms_cm-2'] = hcn_integrated
    l1['emiss'+ext+'_hco+_K_kms_cm-2'] = hcop_integrated
    l1['emiss'+ext+'_co_K_kms_cm-2'] = co_integrated

    # INTEGRATING MOLECULAR INTENSITY
    Ico_integrated = expectation_value_mass_weighted_linear(n_array, flux_co, pdf_weight)
    Ihcn_integrated = expectation_value_mass_weighted_linear(n_array, flux_hcn, pdf_weight)
    Ihcop_integrated = expectation_value_mass_weighted_linear(n_array, flux_hcop, pdf_weight)

    l1['inten'+ext+'_hcn_K_kms'] = Ihcn_integrated
    l1['inten'+ext+'_hco+_K_kms'] = Ihcop_integrated
    l1['inten'+ext+'_co_K_kms'] = Ico_integrated

    # MASS-WEIGHTED RADIUS
    co_r = expectation_value_mass_weighted_linear(n_array, r_arr, emissivity_co_weighted*pdf_weight) #,i0=ico)
    hcn_r = expectation_value_mass_weighted_linear(n_array, r_arr, emissivity_hcn_weighted*pdf_weight) #,i0=ihcn)
    hcop_r = expectation_value_mass_weighted_linear(n_array, r_arr, emissivity_hcop_weighted*pdf_weight) #,i0=ihcop)
    r_cld = expectation_value_mass_weighted_linear(n_array, r_arr, pdf_weight) #,i0=ihcop)

    l1['r'+ext+'_hcn_mass_wt'] = hcn_r
    l1['r'+ext+'_hco+_mass_wt'] = hcop_r
    l1['r'+ext+'_co_mass_wt'] = co_r
    l1['r_cloud_mass_wt'] = r_cld

    # EFFECTIVE EXCITATION DENSITY & RADIUS
    ico, seff_co = find_effective_ex_density(s_array,emissivity_co_weighted*pdf_weight*np.exp(s_array),lim=ff_lim)
    ihcn, seff_hcn = find_effective_ex_density(s_array,emissivity_hcn_weighted*pdf_weight*np.exp(s_array),lim=ff_lim)
    ihcop, seff_hcop = find_effective_ex_density(s_array,emissivity_hcop_weighted*pdf_weight*np.exp(s_array),lim=ff_lim)
    icld,seff_tot = find_effective_ex_density(s_array,pdf_weight*np.exp(s_array),lim=ff_lim)

    l1['i'+ext+'_hcn'] = ihcn
    l1['i'+ext+'_hco+'] = ihcop
    l1['i'+ext+'_co'] = ico
    
    neff_co = n_0*np.exp(seff_co)
    neff_hcn = n_0*np.exp(seff_hcn)
    neff_hcop = n_0*np.exp(seff_hcop)

    l1['neff'+ext+'_hcn'] = neff_hcn
    l1['neff'+ext+'_hco+'] = neff_hcop
    l1['neff'+ext+'_co'] = neff_co

    #co_r = r_arr[ico]
    #hcn_r = r_arr[ihcn]
    #hcop_r = r_arr[ihcop]
    #r_cld = r_arr[icld]

    l1['r'+ext+'_hcn'] = hcn_r
    l1['r'+ext+'_hco+'] = hcop_r
    l1['r'+ext+'_co'] = co_r
    l1['r_cloud'] = r_cld

    # FILLING FACTORS
    l1['ff'+ext] = hcn_r**2/co_r**2
    l1['ff_co_sfr'+ext] = co_r**2/r_crit**2
    
    # OPTICAL DEPTH
    #tau_co  = format_tau(tau_co)
    #tau_hcn  = format_tau(tau_hcn)
    #tau_hcop  = format_tau(tau_hcop)
        
    co_tau = expectation_value_mass_weighted_linear(n_array, tau_co, pdf_weight*emissivity_co_weighted)
    hcn_tau = expectation_value_mass_weighted_linear(n_array, tau_hcn, pdf_weight*emissivity_hcn_weighted)
    hcop_tau = expectation_value_mass_weighted_linear(n_array, tau_hcop, pdf_weight*emissivity_hcop_weighted)

    #co_tau = expectation_value_mass_weighted_linear(n_array, tau_co, pdf_weight)
    #hcn_tau = expectation_value_mass_weighted_linear(n_array, tau_hcn, pdf_weight)
    #hcop_tau = expectation_value_mass_weighted_linear(n_array, tau_hcop, pdf_weight)

    l1['tau'+ext+'_hcn_weight'] = hcn_tau
    l1['tau'+ext+'_hco+_weight'] = hcop_tau
    l1['tau'+ext+'_co_weight'] = co_tau
    
    # EXCITATION TEMPERATURE
    co_tex = expectation_value_mass_weighted_linear(n_array, tex_co, pdf_weight*emissivity_co_weighted)#,i0=ico)
    hcn_tex = expectation_value_mass_weighted_linear(n_array, tex_hcn, pdf_weight*emissivity_hcn_weighted)#,i0=ihcn)
    hcop_tex = expectation_value_mass_weighted_linear(n_array, tex_hcop, pdf_weight*emissivity_hcop_weighted)#,i0=ihcop)
    
    #co_tex = expectation_value_mass_weighted_linear(n_array, tex_co, pdf_weight)
    #hcn_tex = expectation_value_mass_weighted_linear(n_array, tex_hcn, pdf_weight)
    #hcop_tex = expectation_value_mass_weighted_linear(n_array, tex_hcop, pdf_weight)

    l1['tex'+ext+'_hcn_weight'] = hcn_tex
    l1['tex'+ext+'_hco+_weight'] = hcop_tex
    l1['tex'+ext+'_co_weight'] = co_tex

    C = PL_amplitude(alpha,sigma_s)
    N = normalization(C,s_t,alpha,sigma_s)
    s_array_large = np.linspace(-10,20,500)
    LN_large = lognormal(N,sigma_s,s_array_large,s_0)
    PL_large = power_law(N,C,alpha,s_array_large)
    PDF_large = combine_PL_LN(s_t,PL_large,LN_large,s_array_large)
    scut = np.log(10**1.5/n_0)
    f1p5 = mass_frac_above_crit(scut,PDF_large,s_array_large)
    scut = np.log(10**2.5/n_0)
    f2p5 = mass_frac_above_crit(scut,PDF_large,s_array_large)
    scut = np.log(10**3.5/n_0)
    f3p5 = mass_frac_above_crit(scut,PDF_large,s_array_large)
    scut = np.log(10**4.5/n_0)
    f4p5 = mass_frac_above_crit(scut,PDF_large,s_array_large)
    scut = np.log(10**5.5/n_0)
    f5p5 = mass_frac_above_crit(scut,PDF_large,s_array_large)
    
    l1['f5p5'] = float(f5p5)
    l1['f4p5'] = float(f4p5)
    l1['f3p5'] = float(f3p5)
    l1['f2p5'] = float(f2p5)
    l1['f1p5'] = float(f1p5)

    return l1

def calc_slope(x0,y0,x1,y1):
    return (y1-y0)/(x1-x0)

def remove_outliers2(df_):
    
    # CHECK FOR DIVERGENCE
    s_arr = df_['s_array']
    tau_hcn = df_['TAU_hcn']
    i = np.argmin(tau_hcn)
    y1 = s_arr.iloc[i]
    x1 = tau_hcn.iloc[i]
    try:
        x0 = s_arr.iloc[i-1]
        y0 = tau_hcn.iloc[i-1]
    except:
        x0 = x1
        y0 = y1
    i0 = calc_slope(x0,y0,x1,y1)
    try:
        x0 = s_arr.iloc[i+1]
        y0 = tau_hcn.iloc[i+1]
    except:
        x0 = x1
        y0 = y1
    x0 = s_arr.iloc[i+1]
    y0 = tau_hcn.iloc[i+1]
    i1 = calc_slope(x0,y0,x1,y1)
    if ((np.isnan(i0) or np.isnan(i1))) and ((i == 0.) or (i == len(df_))):
        # mask adjacent
        i = np.arange(i-5,i+6,1).astype('int')
        i_full = np.arange(0,len(df_)+1,1).astype('int')
        i = (i > len(s_arr)) | (i < 0)
        i_keep = list(set(i_full)-set(i))
        df_ = df_.iloc[i_keep]
    
    tau_co = df_['TAU_co']
    i = np.argmin(tau_co)
    x0 = s_arr.iloc[i-1]
    y0 = tau_co.iloc[i-1]
    y1 = s_arr.iloc[i]
    x1 = tau_co.iloc[i]
    i0 = calc_slope(x0,y0,x1,y1)
    x0 = s_arr.iloc[i+1]
    y0 = tau_co.iloc[i+1]
    i1 = calc_slope(x0,y0,x1,y1)
    if ((np.isnan(i1) or np.isnan(i0))) and ((i == 0.) or (i == len(df_))):
        i = np.arange(i-5,i+6,1).astype('int')
        i_full = np.arange(0,len(df_)+1,1).astype('int')
        i_keep = list(set(i_full)-set(i))
        df_ = df_.iloc[i_keep]
        
    return df_
