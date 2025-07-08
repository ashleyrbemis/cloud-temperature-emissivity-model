# CONSTRUCTING THE LN + PL PDF FOLLOWING THE FORMALISM OF BURKHART+2019
import math
import sys
import numpy as np
import astropy.constants as const
from astropy import units as u
import scipy
from scipy.special import erf
import math
import pandas as pd
from scipy import stats as st
#from temp_estimates import temp_grav, temp_cr, temp_turb
import matplotlib.pyplot as plt
from scipy.special import hyp2f1

# ELEMENT BY ELEMENT FUNCTIONS FOR MATRICES
mult_elem = np.vectorize(lambda i,j: i*j)
div_elem = np.vectorize(lambda i,j: i/j)
sub_elem = np.vectorize(lambda i,j: i-j)

def parse_pdf_file(tab):
    pdf = tab['PDF'].values
    s_array = tab['s_array'].values
    ind = tab['radex_grid_index'].values[0]
    s_t = tab['s_t'].values[0]
    sigma_s = tab['sigma_s'].values[0]
    s_0 = tab['s_0'].values[0]
    fdense = tab['fcrit'].values[0]
    mach = tab['mach'].values[0]
    vdisp = tab['vdisp'].values[0]
    temp = tab['temp'].values[0]
    alpha = tab['alpha'].values[0]
    b = tab['b'].values[0]
    return pdf, s_array, ind, s_t, sigma_s, s_0, fdense, mach, vdisp, temp, alpha, b

def calc_sigma_sb(b, v_disp, c_s, beta):
    i = np.inf(beta)
    sigma_s = LN_width(b, v_disp, c_s)
    sigma_s[i] = LN_width_alfven(b[i], v_disp[i], c_s[i], beta[i])
    return sigma_s

def conservation_shift(N,C,alpha,s_t,sigma_s):
    s_t = np.array(s_t)
    sigma_s = np.array(sigma_s)
    f0 = (np.exp(s_t*(1-alpha))*N*C)/(alpha-1)
    f1_ = (sigma_s**2 - 2*s_t)/(2*np.sqrt(2)*sigma_s)
    f1 = (N/2)*erf(f1_)
    return np.log(f0 + f1)

def normalization(C,s_t,alpha,sigma_s):
    # CALCULATES NORMALIZATION OF LN+PL N-PDF
    f1 = C*np.exp(-s_t*alpha)/alpha
    f2 = 1./2
    r = (2*s_t + sigma_s**2)/(2*np.sqrt(2)*sigma_s)
    f3 = (1./2)*erf(r)
    return (f1 + f2 + f3)**-1

def transition_density(alpha,sigma_s):
    return (alpha - 1./2)*sigma_s**2

def PL_amplitude(alpha,sigma_s):
    C = np.exp((1./2)*(alpha - 1)*alpha*sigma_s**2)/(sigma_s*np.sqrt(2*np.pi))
    return C

def sound_speed(temp):
    c_s = np.sqrt(const.k_B.value * temp / (2.33 * 1.635575*1e-27))
    return np.round(c_s/1e3,2)

def mean_log_density(sigma_s):
    return - (1./2) * sigma_s**2

def logarithmic_density(n, rho_0, mu = 2.33): # rho
    mH2 = mu*(const.m_p+const.m_e).value
    rho = n*mH2
    if isinstance(rho_0,(list,tuple,np.ndarray)):
        r1 = np.matrix(rho)
        r2 = np.matrix(rho_0)
        r = r1.T/r2
    else:
        r = rho / rho_0
    return np.log(r)

def fbeta_PN11(beta):
    return (1 + 0.925*beta**(-3./2))**(-2./3)/(1 + 1/beta)**2

def power_law(N,C,alpha,s):
    PL = N*C*np.exp(-alpha*s)
    return PL

def calc_fdense_Burk2019(sigma_s,alpha):
    C = PL_amplitude(alpha,sigma_s)
    top = C*np.exp(sigma_s**2*(1-alpha)*(alpha-0.5))
    z = (2*sigma_s**2*(alpha - 0.5) - sigma_s**2)/(2*np.sqrt(2)*sigma_s)
    bottom = (alpha - 1)/2*(1 + erf(z)) + C*np.exp(sigma_s**2*(alpha-0.5)*(1-alpha))
    return top/bottom

def LN_width(b, mach):
    return np.sqrt(np.log(1 + b**2 * (mach)**2))

def LN_width_alfven(b, mach, beta):
    return np.sqrt(np.log(1 + b**2 * (mach)**2*(beta/(beta+1))))

def lognormal(N,sigma_s,s,s_0):
    LN = (N/np.sqrt(2*np.pi*sigma_s**2))*np.exp(-(s-s_0)**2/(2*sigma_s**2))
    return LN

def combine_PL_LN(s_t,PL,LN,s_array):
    ind_PL = s_array >= s_t
    ind_LN = s_array < s_t

    PDF = np.zeros_like(LN)
    PDF[ind_LN] = LN[ind_LN]
    PDF[ind_PL] = PL[ind_PL]
    return PDF


def calc_big_PDF(alpha, b, beta, mach, n_src, size=100,
                 smin=-5,smax=10,n_ref=10**3.5,
                 mass_conserving=False,lognorm=False,
                 s_crit=False,s_const=False,alpha_vir=1.):

    try:
        mach = np.vstack(mach)
    except:
        pass
    try:
        alpha = np.vstack(alpha)
    except:
        pass
    try:
        alpha_vir = np.vstack(alpha_vir)
    except:
        pass
    try:
        beta = np.vstack(beta)
    except:
        pass
    try:
        b = np.vstack(b)
    except:
        pass
    try:
        n_src = np.vstack(n_src)
    except:
        pass

    if isinstance(beta,np.ndarray):
        sigma_s = LN_width_alfven(b, mach, beta)
    else:
        if np.isfinite(beta):
            sigma_s = LN_width_alfven(b, mach, beta)
        else:
            sigma_s = LN_width(b, mach)

    C = PL_amplitude(alpha,sigma_s)
    s_t = transition_density(alpha,sigma_s)
    N = normalization(C,s_t,alpha,sigma_s)
    s_0 = mean_log_density(sigma_s)

    s_array = np.linspace(smin,smax,size)
    if isinstance(mach,np.ndarray):
        s_array = np.tile(s_array,len(mach)).reshape(len(mach),size)
    
    if mass_conserving == True:
        s_s = conservation_shift(N, C, alpha, s_t, sigma_s)
        s_array = s_array - np.nan_to_num(s_s)

    PL = power_law(N,C,alpha,s_array)
    LN = lognormal(N,sigma_s,s_array,s_0)
    PDF = combine_PL_LN(s_t,PL,LN,s_array)
    if lognorm:
        if s_crit == True:
            #print('CALCULATING SCRIT')
            s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
        if s_const == True:
            #print('FIXED SCRIT')
            n_crit = 10**4.5
            s_crit = np.log(n_crit/n_src)
        LN = lognormal(1.,sigma_s,s_array,s_0)
        pdf_trap = (LN + np.roll(LN,-1))/2
        pdf_trap[-1] = np.nan
        ds = np.abs(s_array - np.roll(s_array,-1))
        ds[-1] = np.nan
        sum_tot = np.nansum(pdf_trap*ds*s_array)
        LN = LN/sum_tot
        f3p5 = []
        s_ = np.log(n_ref/n_src)
        for st_, LN_, s_ in zip(s_,pdf_trap,s_array):
            f3p5.append(mass_frac_above_crit(st_,LN_,s_))
        f3p5 = np.array(f3p5)
        fdense = []
        for st_, LN_, s_ in zip(s_crit,pdf_trap,s_array):
            fdense.append(mass_frac_above_crit(st_,LN_,s_))
        fdense = np.array(fdense)
    else:
        f3p5 = []
        s_ = np.log(n_ref/n_src)
        for st_, LN_, s_ in zip(s_,PDF,s_array):
            f3p5.append(mass_frac_above_crit(st_,LN_,s_))
        f3p5 = np.array(f3p5)
        if (s_crit == True):
            s_crit = calc_scrit_PN11(mach, np.inf, alpha_vir)
            fdense = []
            for st_, LN_, s_ in zip(s_crit,PDF,s_array):
                fdense.append(mass_frac_above_crit(st_,LN_,s_))
            fdense = np.array(fdense)
        if s_const == True:
            n_crit = 10**4.5
            s_crit = np.log(n_crit/n_src)
            fdense = []
            for st_, LN_, s_ in zip(s_crit,PDF,s_array):
                fdense.append(mass_frac_above_crit(st_,LN_,s_))
            fdense = np.array(fdense)
        elif (s_const == False):
            fdense = calc_fdense_Burk2019(sigma_s,alpha)
            
    return s_array, PL, LN, PDF, s_t, sigma_s, s_0, fdense, f3p5

def calc_scrit_KM05(mach, alpha_vir, theta_t = 1.):
    return np.log(np.pi**2/15*theta_t**2*alpha_vir*mach**2)

def calc_scrit_HC11(mach,alpha_vir,y_cut = 0.1,beta=np.inf):
    rth_crit = np.pi**2/5*y_cut**-2*alpha_vir*mach**2*(1+1/beta)
    rturb_crit = np.pi**2/15*y_cut**-1*alpha_vir
    return np.log(rth_crit + rturb_crit)

def calc_scrit_PN11(mach, beta, alpha_vir, theta = 0.35):
    if np.isinf(beta):
        scrit = np.log(0.067*theta**(-2)*alpha_vir*mach**2*fbeta_PN11(beta))
    else:
        scrit =  np.log(0.067*theta**(-2)*alpha_vir*mach**2)
    return scrit

def calc_avir(sigma_v,rs,rho_src):
    # sigma_v = 1D vel dispersion
    Mc = 4./3*np.pi*(rs*u.pc)**3*rho_src*u.kg/u.cm**3
    avir = 5*(sigma_v*u.km/u.s)**2*rs*u.pc/(const.G*Mc)
    return avir.to('').value

def calc_SFR_lognormal_PN11(sigma_s, s_crit, eff = 1.):
    #BURKHART + 2018
    # PN11, KM05 if theta_t == 1; also depends on s_crit
    sfr_ff = 0.5*(1+erf((sigma_s**2-2*s_crit)/(2*np.sqrt(2)*sigma_s)))*np.exp(s_crit/2.)
    return abs(sfr_ff)*eff

def calc_SFR_lognormal_KM05(sigma_s, s_crit, eff = 1.,theta_t=1):
    #BURKHART + 2018
    # PN11, KM05 if theta_t == 1; also depends on s_crit
    sfr_ff = 0.5/theta_t*(1+erf((sigma_s**2-2*s_crit)/(2*np.sqrt(2)*sigma_s)))
    return abs(sfr_ff)*eff
    
def calc_SFR_composite(sigma_s, alpha, s_crit, eff = 1.):
    #BURKHART + 2018
    # COMPOSITE LN+PL SFR
    C = PL_amplitude(alpha,sigma_s)
    s_t = transition_density(alpha,sigma_s)
    N = normalization(C,s_t,alpha,sigma_s)
    sfr_ff = np.exp(s_crit/2.)*N*(0.5*erf((sigma_s**2-2*s_crit)/(np.sqrt(8*sigma_s**2))) - 0.5*erf((sigma_s**2-2*s_t)/(np.sqrt(8*sigma_s**2))) +C*np.exp(s_t*(1-alpha))/(alpha-1))
    return abs(sfr_ff)*eff

def calc_SFR_lognormal_multiff(sigma_s, s_crit, theta_t = 1.,eff = 1.):
    #BURKHART + 2018
    # MULTI-FF COMPOSITE LN+PL SFR
    sfr_ff = (eff/(2*theta_t))*(1+erf((sigma_s**2-s_crit)/(np.sqrt(2)*sigma_s))*np.exp(3./8*sigma_s**2))  
    return abs(sfr_ff)

def calc_SFR_composite_multiff(sigma_s, alpha, s_crit, eff = 1.):
    #BURKHART + 2018
    # MULTI-FF COMPOSITE LN+PL SFR
    C = PL_amplitude(alpha,sigma_s)
    s_t = transition_density(alpha,sigma_s)
    N = normalization(C,s_t,alpha,sigma_s)
    sfr_ff = np.exp(3.*(sigma_s**2)/8.)*N*eff/2.*(erf((sigma_s**2-s_crit)/(np.sqrt(2*sigma_s**2))) - erf((sigma_s**2-s_t)/(np.sqrt(2*sigma_s**2)))) + N*eff*C*np.exp(s_t*(1.5-alpha)/(alpha-1.5))
    return abs(sfr_ff)

def mass_frac_above_crit(s_t,pdf_s,s_array):

    ind = s_array > s_t
    pdf_mask = np.copy(pdf_s)
    pdf_mask[ind == False] = np.nan

    pdf_trap = (pdf_s + np.roll(pdf_s,1))/2
    pdf_trap[0] = np.nan # [1]
    
    ds = np.abs(s_array - np.roll(s_array,1))
    ds[0] = np.nan #[1]
    
    sum_above = np.nansum(pdf_mask*ds*np.exp(s_array))
    sum_tot = np.nansum(pdf_trap*ds*np.exp(s_array))
    
    return np.abs(sum_above/sum_tot)

def mass_frac_above_crit_linear(n_t,pdf_n,n_array):

    ind = n_array > n_t
    pdf_mask = np.copy(pdf_n)
    pdf_mask[ind == False] = np.nan

    pdf_tmask = (pdf_mask + np.roll(pdf_mask,-1))/2
    pdf_tmask[-1] = np.nan

    pdf_trap = (pdf_n + np.roll(pdf_n,-1))/2
    pdf_trap[-1] = np.nan
    
    dn = np.abs(n_array - np.roll(n_array,-1))
    dn[-1] = np.nan
    
    sum_above = np.nansum(pdf_tmask*dn*n_array)
    sum_tot = np.nansum(pdf_trap*dn*n_array)
    
    return np.abs(sum_above/sum_tot)

def expectation_value_mass_weighted(s_array,arr,pdf_s,i0=0):

    pdf_trap = (pdf_s + np.roll(pdf_s,-1))/2
    pdf_trap[-1] = np.nan

    trap = (arr + np.roll(arr,-1))/2
    trap[-1] = np.nan
    
    ds = np.abs(s_array - np.roll(s_array,-1))
    ds[-1] = np.nan
    
    sum_above = np.nansum(pdf_trap[i0:]*trap[i0:]*np.exp(s_array[i0:])*ds[i0:])
    sum_tot = np.nansum(pdf_trap[i0:]*np.exp(s_array[i0:])*ds[i0:])
    
    return np.abs(sum_above/sum_tot)

def expectation_value_mass_weighted_linear(n_array,arr,pdf_n,i0=0):

    pdf_trap = (pdf_n + np.roll(pdf_n,-1))/2
    pdf_trap[-1] = np.nan

    trap = (arr + np.roll(arr,-1))/2
    trap[-1] = np.nan
    
    dn = np.abs(n_array - np.roll(n_array,-1))
    dn[-1] = np.nan
    
    sum_above = np.nansum(pdf_trap[i0:]*trap[i0:]*n_array[i0:]*dn[i0:])
    sum_tot = np.nansum(pdf_trap[i0:]*dn[i0:]*n_array[i0:])
    
    return np.abs(sum_above/sum_tot)

def expectation_value_volume_weighted_linear(n_array,arr,pdf_n,i0=0):

    pdf_trap = (pdf_n + np.roll(pdf_n,-1))/2
    pdf_trap[-1] = np.nan

    trap = (arr + np.roll(arr,-1))/2
    trap[-1] = np.nan
    
    dn = np.abs(n_array - np.roll(n_array,-1))
    dn[-1] = np.nan
    
    sum_above = np.nansum(pdf_trap[i0:]*trap[i0:]*dn[i0:])
    sum_tot = np.nansum(pdf_trap[i0:]*dn[i0:])
    
    return np.abs(sum_above/sum_tot)

def find_effective_ex_density(s_array,pdf,lim=0.1):
    
    pdf_trap = (pdf + np.roll(pdf,-1))/2
    pdf_trap[-1] = np.nan

    ds = np.abs(s_array - np.roll(s_array,-1))
    ds[-1] = np.nan

    sum_tot = np.nansum(pdf_trap*ds)
    
    for i in np.flip(np.arange(len(s_array))):
        sum_above = np.nansum(pdf_trap[i:-1]*ds[i:-1])
        if (sum_above/sum_tot >= lim):
            return i, s_array[i]

def calc_slopes(kappa):
    alpha_s = 3/kappa 
    alpha_N = (1+kappa)/abs(1-kappa)
    alpha_n = (3/kappa + 1)
    alpha_eta = abs(2/(1-kappa))
    return alpha_s, alpha_N, alpha_n, alpha_eta

def lognormal_quantile(mu,sigma,p):
    return np.exp(mu**2+np.sqrt(2*sigma**2)/erf(2*p-1))




def get_PDFs(sigma_v,
             temp,
             Sigma,
             rs,
             kappa=1.5,
             size=1000,
             b=0.4,
             beta=np.inf,
             n_ref = 10**4.5,
             mass_conserving=False,
             lognorm=False,
             s_crit=False,
             s_const=False,
             alpha_vir = 1.,
             correct_dens=False,
             user_nsrc = -1,
             r_ref = 1.):


    N_surface = Sigma*u.M_sun/u.pc**2/(2.33*(const.m_p+const.m_e))
    N_surface = N_surface.to(u.cm**-2).value
    
    sigma_v = np.sqrt(3)*sigma_v
    c_s = sound_speed(temp)
    mach = sigma_v/c_s
    alpha_s, alpha_N, alpha_n, alpha_eta = calc_slopes(kappa)

    if isinstance(beta,np.ndarray):
        sigma_s = LN_width_alfven(b, mach, beta)
    else:
        if np.isfinite(beta):
            sigma_s = LN_width_alfven(b, mach, beta)
        else:
            sigma_s = LN_width(b, mach)

    # 3D + 2D - LOG + LINEAR STD
    sigma_n = np.sqrt(np.exp(sigma_s**2)-1)
    A = 0.11
    sigma_eta = np.sqrt(A)*sigma_s
    sigma_N = np.sqrt(np.exp(sigma_eta**2)-1)
    ks = sigma_n/(sigma_N+sigma_n)
    kappa_N = 3./alpha_eta

    Mc_const = 4*np.pi*Sigma*u.M_sun/u.pc**2*(rs*u.pc)**2
    Mc_const = Mc_const.to(u.M_sun)
    rho_src = Mc_const/(4/3*np.pi)/(rs*u.pc)**3
    rho_src = rho_src.to(u.kg/u.cm**3).value
    n_surface = rho_src/(2.33*(const.m_p+const.m_e)).value
    Mc_const = Mc_const.value
    
    n_src = n_surface*sigma_n**2/2
    N_src = N_surface*sigma_N**2/2
    Mc = 4/3*np.pi*(n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
    Mc = Mc.to(u.M_sun).value
    fcorr = Mc_const/Mc
 
    if isinstance(user_nsrc,(list,np.ndarray)):
        n_src = user_nsrc
    elif isinstance(user_nsrc,int):
        if (user_nsrc != -1):
            n_src = user_nsrc

    rho_surface = n_surface*(2.33*(const.m_p+const.m_e))*u.cm**-3
    rho_surface = rho_surface.to(u.kg/u.cm**3).value

    '''
    if user_nsrc != -1:
        n_src = user_nsrc
    
    # calculate mean density
    #rho_ref = rho_src*(rs/r_ref)**-kappa*(1+rs/r_ref)**(ks - kappa)
    rho_ref = rho_src*(rs/r_ref)**-kappa*(1+rs/r_ref)**(kappa-ks)
    # calculate mass of double power law using the hypergeometric function
    c0 = 3-kappa
    c1 = ks - kappa
    c2 = 4-kappa
    c3 = -rs/r_ref
    Mc = 4*np.pi*rho_ref*u.kg/u.cm**3*(rs*u.pc)**3*(rs/r_ref)**-kappa/(3-kappa)*hyp2f1(c0,c1,c2,c3)
    Mc = (Mc).to(u.M_sun).value
    
    fcorr = Mc_const / Mc
    '''

    if correct_dens == True:
        if lognorm:
            fcorr = (1-ks/3)/(1-ks/5)
        if isinstance(n_surface,(list,np.ndarray)):
            n_src = np.hstack(n_surface)/np.hstack(fcorr)
            N_src = np.hstack(N_surface)/np.hstack(fcorr)
            #alpha_vir = alpha_vir*np.hstack(fcorr)
        else:
            n_src = n_surface/fcorr
            N_src = N_surface/fcorr
            #alpha_vir = alpha_vir*fcorr
    else:
        if isinstance(Sigma,(list,np.ndarray)):
            N_src = np.hstack(N_surface)
            n_src = np.hstack(n_surface)
        else:
            N_src = N_surface
            n_src = n_surface

    r0 = (N_src/n_src)*u.cm.to(u.pc)

    # TRANSITION DENSITY
    eta_t = 0.5*(2*abs(alpha_eta)-1)*sigma_eta**2
    N_t = N_src*np.exp(eta_t)
    s_t = transition_density(alpha_s,sigma_s)
    n_t = n_src*np.exp(s_t)
    r_t = (N_t/n_t)*u.cm.to(u.pc)
    
    try:
        C = ((n_t/n_surface)*(r_t/rs)**(kappa/(ks-kappa)))
        C2 = ((N_t/N_surface)*(r_t/rs)**((kappa+1)/(ks-kappa)))
    except:
        C=0.
        C2 = 0.
    if lognorm:
        C=0
        C2=0.
    rnorm = abs(rs - C*r_t)/abs(C-1)
    rnorm2 = abs(rs - C2*r_t)/abs(C2-1)
    nnorm = n_t*(r_t/rnorm)**-kappa*(1+r_t/rnorm)**(kappa-ks)
    #Nnorm = nnorm*u.cm**-3*rnorm*u.pc
    #Nnorm = Nnorm.to(u.cm**-2).value
    Nnorm = N_t*(r_t/rnorm)**-(kappa+1)*(1+r_t/rnorm)**(kappa-ks)
    #Nnorm = N_t*(r_t/rnorm2)**-(kappa+1)*(1+r_t/rnorm2)**(kappa-ks)
    #print(rnorm2,Nnorm)

    #rnorm = rs*r_t/r_t
    #nnorm = n_surface #n_t

    # 3D
    if isinstance(n_src,(list,np.ndarray)):
        sigma_s = sigma_s.reshape(-1,1)
        s_t = s_t.reshape(-1,1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)

    # 2D
    if isinstance(sigma_eta,(list,np.ndarray)):
        sigma_eta = sigma_eta.reshape(-len(sigma_eta),1)
        eta_t = eta_t.reshape(-len(eta_t),1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)
        alpha_eta = alpha_eta.reshape(-1,1)

    # INIT DENSITY ARRAYS
    s_0 = mean_log_density(sigma_s)
    if isinstance(n_src,(list,np.ndarray)):
        s_0 = s_0.reshape(-1,1)
    s_array = np.squeeze(np.linspace(s_0-sigma_s*10,s_0+sigma_s*10,size).T)
    eta_0 = -0.5*sigma_eta**2
    if isinstance(n_src,(list,np.ndarray)):
        eta_0 = eta_0.reshape(-1,1)
    eta = np.squeeze(np.linspace(eta_0-sigma_eta*10,eta_0+sigma_eta*10,size).T)
    
    # 3D
    #s_0 = mean_log_density(sigma_s)
    C = PL_amplitude(alpha_s,sigma_s)
    N = normalization(C,s_t,alpha_s,sigma_s)
    LN = lognormal(N,sigma_s,s_array,s_0)
    PL = power_law(N,C,alpha_s,s_array)
    if lognorm == False:
        PDF = combine_PL_LN(s_t,PL,LN,s_array)
    elif lognorm == True:
        PDF = LN

    i0 = np.squeeze((s_array > s_0) & (np.log10(PDF) < -5))
    i1 = np.squeeze((s_array < s_0) & (np.log10(PDF) < -5))
    if isinstance(s_0,(list,np.ndarray)):
        s_max = []
        eta_max = []
        for i,s,eta_ in zip(i0,s_array,eta):
            s_max.append(s[i][0])
            eta_max.append(eta_[i][0])
        s_max = np.array(s_max).reshape(-1,1)
        eta_max = np.array(eta_max).reshape(-1,1)
        
        s_min = []
        eta_min = []
        for i,s,eta_ in zip(i1,s_array,eta):
            s_min.append(s[i][-1])
            eta_min.append(eta_[i][-1])
        s_min = np.array(s_min).reshape(-1,1)
        eta_min = np.array(eta_min).reshape(-1,1)
    else:
        s_max = s_array[i0][0]
        s_min = s_array[i1][-1]
        eta_min = eta[i0][0]
        eta_max = eta[i1][-1]
    #print(s_max)

    

    if lognorm:
        nmin = 1e-1
        nmax = 1e11
        
        if isinstance(n_src,(list,np.ndarray)):
            # assumes r0 cons
            n_src = n_src.reshape(-1,1)
            N_src = N_src.reshape(-1,1)
            N_surface = N_surface.reshape(-1,1)
            ks = ks.reshape(-1,1)
            if isinstance(rs,(list,np.ndarray)):
                rs = rs.reshape(-1,1) #.flatten()
            n_array = np.logspace(np.log10(nmin),np.log10(nmax),size)
            r = rs*(n_array/n_src)**(1/-ks)
            n_array = (n_src)*(r/rs)**(-ks)
            N_array = (N_src)*(r/rs)**(-ks+1)
            s_array = np.log(n_array/(n_src))
            eta = np.log(N_array/N_src)
            n_src = n_src.flatten()
            N_src = N_src.flatten()
            N_surface = N_surface.flatten()
            ks = ks.flatten()
        else:
            n_array = np.logspace(np.log10(nmin),np.log10(nmax),size)
            r = rs*(n_array/n_src)**(-1/ks)
            n_array = n_src*(r/rs)**(-ks)
            N_array = N_src*(r/rs)**(-ks+1)
            s_array = np.log(n_array/n_src)
            eta = np.log(N_array/N_src)

    
    if lognorm == False:
        nmin = 1e-2
        nmax = 1e10
        
        if isinstance(n_src,(list,np.ndarray)):
            nnorm = nnorm.flatten()
            rnorm = rnorm.flatten()
            ks = ks.flatten()
            if isinstance(kappa,(np.ndarray,list)):
                kappa = kappa.flatten()
            r = np.logspace(6,-6,1000).reshape(-1,1)
            n0_ = n_t*(r/r_t)**-kappa
            n1_ = n_surface*(r/rs)**-ks
            n_arr = 1/(1/n0_ + 1/n1_)
            i0 = np.argmin(abs(1-n_arr/nmin),axis=0)
            i1 = np.argmin(abs(1-n_arr/nmax),axis=0)
            rmax = r[i0].flatten()
            rmin = r[i1].flatten()
            r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
            nnorm = nnorm.reshape(-1,1)
            rnorm = rnorm.reshape(-1,1)
            if isinstance(kappa,(np.ndarray,list)):
                kappa = kappa.reshape(-1,1)
            ks = ks.reshape(-1,1)
            n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
            imax = np.argmax(n_array,axis=1)
            imin = np.argmin(n_array,axis=1)
            rmax = r[np.arange(len(imin)),imin]
            rmin = r[np.arange(len(imax)),imax]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
            while np.any(abs(np.log10(np.nanmax(n_array,axis=1)/nmax)) > 0.1) or np.any(abs(np.log10(np.nanmin(n_array,axis=1)/nmin)) > 0.1):
                n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
                imax = np.argmax(n_array,axis=1)
                imin = np.argmin(n_array,axis=1)
                cmax = np.nanmax(n_array,axis=1)/nmax
                cmin = np.nanmin(n_array,axis=1)/nmin
                rmax = r[np.arange(len(imin)),imin]
                rmin = r[np.arange(len(imax)),imax]
                i0 = abs(np.log10(cmax)) > 0.1
                i1 = abs(np.log10(cmin)) > 0.1
                rmax[i1] = (rmax*(cmin**0.33))[i1]
                rmin[i0] = (rmin*(cmax**0.33))[i0]
                r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
        
        else:
            r = np.logspace(6,-6,1000)
            n0_ = n_t*(r/r_t)**-kappa
            n1_ = n_src*(r/rs)**-ks
            n_arr = 1/(1/n0_ + 1/n1_)            
            i0 = np.argmin(abs(1-n_arr/nmin))
            i1 = np.argmin(abs(1-n_arr/nmax))
            rmax = r[i0]
            rmin = r[i1]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size)
            #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(ks-kappa)
            n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
            imax = np.argmax(n_array) # removed, axis=1
            imin = np.argmin(n_array)
            rmax = r[imin] #np.arange(len(imin)),imin]
            rmin = r[imax] #np.arange(len(imax)),imax]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size)
            while np.any(abs(np.log10(np.nanmax(n_array)/nmax)) > 1.) or np.any(abs(np.log10(np.nanmin(n_array)/nmin)) > 1.):
                # removed axis=1
                #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(ks-kappa)
                n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
                imax = np.argmax(n_array) #,axis=1)
                imin = np.argmin(n_array) #,axis=1)
                cmax = np.nanmax(n_array)/nmax # removed axis=1
                cmin = np.nanmin(n_array)/nmin # removed axis=1
                #print(cmin,cmax)
                rmax = r[imin] #[np.arange(len(imin)),imin]
                rmin = r[imax] #[np.arange(len(imax)),imax]
                i0 = np.argmin(abs(1-n_array/nmin)) # 
                i1 =  np.argmin(abs(1-n_array/nmax)) # abs(np.log10(cmin)) > 1.
                rmax = (rmax*(cmin**0.33)) # [i1]
                rmin = (rmin*(cmax**0.33)) # [i0]
                r = np.logspace(np.log10(rmax),np.log10(rmin),size)

        if isinstance(n_src,(list,np.ndarray)):
            Nnorm = Nnorm.reshape(-1,1)
            n_src = n_src.reshape(-1,1)
            N_src = N_src.reshape(-1,1)
            N_surface = N_surface.reshape(-1,1)
            rnorm2 = rnorm2.reshape(-1,1)

        n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
        N_array = Nnorm*(r/rnorm2)**-(kappa+1)*(1+r/rnorm2)**(kappa-ks)
        #N_array = Nnorm*(r/rnorm)**-(kappa+1)*(1+r/rnorm)**(kappa-ks)
        #print(N_array)
        #N_array = n_array*rnorm*(u.cm**-3*u.pc).to(u.cm**-2)
        N_array = n_array*r0.reshape(-1,1)*(u.cm**-3*u.pc).to(u.cm**-2)
        s_array = np.log(n_array/n_src)
        eta = np.log(N_array/N_src)
        

        '''
        # TESTING
        ks = abs(np.log(n_surface/n_t)/np.log(rs/r_t))
        print(ks)
        
        if isinstance(n_t,(np.ndarray,list)):
            n_t = np.vstack(n_t)
            N_t = np.vstack(N_t)
            r_t = np.vstack(r_t)
            n_surface = np.vstack(n_surface)
            N_surface = np.vstack(N_surface)
            n_src = np.vstack(n_src)
            N_src = np.vstack(N_src)
            ks = np.vstack(ks)
           
        if isinstance(rs,(np.ndarray,list)):
            rs = np.vstack(rs)

        if isinstance(kappa,(np.ndarray,list)):
            kappa = np.vstack(kappa)
            
        r = np.tile(np.logspace(3,-3,size),len(n_src)).reshape(len(n_src),size)
        n_PL = n_t*(r/r_t)**-kappa
        n_env = n_surface*(r/rs)**-ks
        n_array = n_env
        n_array[r < r_t] = n_PL[r < r_t]
        s_array = np.log(n_array/n_src)
        #print(n_array)

        N_array = n_array*rs*u.pc.to(u.cm)
        #N_PL = N_t*(r/r_t)**-(kappa+1)
        #N_env = N_surface*(r/rs)**-(ks+1)
        #N_array = N_env
        #N_array[r < r_t] = N_PL[r < r_t]
        eta = np.log(N_array/N_src)
        '''
        
           
    if isinstance(n_src,(list,np.ndarray)):
        sigma_s = sigma_s.reshape(-1,1)
        s_t = s_t.reshape(-1,1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)

    # 3D
    s_0 = mean_log_density(sigma_s)
    C = PL_amplitude(alpha_s,sigma_s)
    N = normalization(C,s_t,alpha_s,sigma_s)
    LN = lognormal(N,sigma_s,s_array,s_0)
    PL = power_law(N,C,alpha_s,s_array)
    if lognorm == False:
        PDF = combine_PL_LN(s_t,PL,LN,s_array)
    elif lognorm == True:
        PDF = LN
    
    
    if mass_conserving:
        s_s = conservation_shift(N,C,alpha_s,s_t,sigma_s)
        s_array = s_array+np.nan_to_num(s_s)
        
    # 2D
    if isinstance(sigma_eta,(list,np.ndarray)):
        sigma_eta = sigma_eta.reshape(-len(sigma_eta),1)
        eta_t = eta_t.reshape(-len(eta_t),1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_eta = alpha_eta.reshape(-1,1)

    # 2D
    eta_0 = -0.5*sigma_eta**2
    C_N = PL_amplitude(alpha_eta,sigma_eta)
    N_N = normalization(C_N,eta_t,alpha_eta,sigma_eta)
    LN_N = lognormal(N_N,sigma_eta,eta,eta_0)
    PL_N = power_law(N_N,C_N,alpha_eta,eta)
    PDF_N = combine_PL_LN(eta_t,PL_N,LN_N,eta)

    if mass_conserving:
        eta_s = conservation_shift(N_N,C_N,alpha_eta,eta_t,sigma_eta)
        eta = eta-np.nan_to_num(eta_s)

    if lognorm:
        s_fix = np.log(n_ref/n_src)
        if s_crit == 'KM05':
            s_crit = calc_scrit_KM05(mach, alpha_vir)
            if isinstance(s_crit,(np.ndarray,list)):
                fcrit = []
                #s_crit = s_crit.reshape(-len(s_crit),1)
                for st_, LN_, s_ in zip(s_crit,LN,s_array):
                    fcrit.append(mass_frac_above_crit(st_,LN_,s_))
                    #plt.plot(s_,LN_)
                fcrit = np.array(fcrit)
            else:
                fcrit = mass_frac_above_crit(s_crit,LN,s_array)
        elif s_crit == 'PN11':
            #theta = np.sqrt(1/(np.pi**2/15/0.067))
            s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
            if isinstance(s_crit,(np.ndarray,list)):
                fcrit = []
                #s_crit = s_crit.reshape(-len(s_crit),1)
                for st_, LN_, s_ in zip(s_crit,LN,s_array):
                    fcrit.append(mass_frac_above_crit(st_,LN_,s_))
                    #plt.plot(s_,LN_)
                fcrit = np.array(fcrit)
            else:
                fcrit = mass_frac_above_crit(s_crit,LN,s_array)
        if isinstance(s_fix,(np.ndarray,list)):
            ffix = []
            #s_fix = s_fix.reshape(-len(s_fix),1)
            for st_, LN_, s_ in zip(s_fix,LN,s_array):
                ffix.append(mass_frac_above_crit(st_,LN_,s_))
            ffix = np.array(ffix)
        else:
            ffix = mass_frac_above_crit(s_fix,LN,s_array)
        if s_const == True:
            s_crit = s_fix
            fcrit = ffix
                
            
    elif lognorm == False:
        #s_choice = s_crit
        s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
        s_fix = np.log(n_ref/n_src)
        fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)
        ffix = mass_frac_above_crit(s_fix,PDF,s_array)
        f_t = mass_frac_above_crit(s_t,PDF,s_array)
        if isinstance(s_crit,(np.ndarray,list)):
            fcrit = calc_fdense_Burk2019(sigma_s.reshape(-len(sigma_s),1),alpha_s)
            fcrit = np.array(fcrit)
        if isinstance(s_fix,(np.ndarray,list)):
            ffix = []
            s_fix = s_fix.reshape(-len(s_fix),1)
            for sf_,PDF_, s_ in zip(s_fix,PDF,s_array):
                ffix.append(mass_frac_above_crit(sf_,PDF_,s_))
            ffix = np.array(ffix)
        if isinstance(s_t,(np.ndarray,list)):
            f_t = []
            s_t = s_t.reshape(-len(s_t),1)
            for sf_,PDF_, s_ in zip(s_t,PDF,s_array):
                f_t.append(mass_frac_above_crit(sf_,PDF_,s_))
            f_t = np.array(f_t)
        else:
            fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)
            ffix = mass_frac_above_crit(s_fix,PDF,s_array)
            f_t = mass_frac_above_crit(s_t,PDF,s_array)
        if (s_const == True):
            fcrit = ffix

    ##############################
    # SAVING PDF(S) TO DATAFRAME #
    ##############################

    #plt.plot((s_array[100].T),np.log10(PDF[100].T))
    #plt.show()
        
    df_pdf = pd.DataFrame()
    if isinstance(n_src,(list,np.ndarray)):
        nmodel = np.arange(1,len(n_src.flatten())+1)
    elif isinstance(n_src,(list,np.ndarray)) == False:
        nmodel = 0.
    else:
        print('CANNOT PRODUCE MODEL INDEX')
        return


    # DENSITY ARRAYS
    df_pdf['s_array'] = s_array.flatten()
    df_pdf['eta'] = eta.flatten()
    df_pdf['log n_arr'] = np.log10(n_array.flatten())
    df_pdf['log N_arr'] = np.log10(N_array.flatten())
    # RADII
    df_pdf['r_arr_pc'] = r.flatten()
    

    # 3D PDF MODEL ARRAYS
    df_pdf['LN'] = LN.flatten()
    df_pdf['PL'] = PL.flatten()
    df_pdf['PDF'] = PDF.flatten()
    # 2D PDF MODEL ARRAYS
    df_pdf['LN_eta'] = LN_N.flatten()
    df_pdf['PL_eta'] = PL_N.flatten()
    df_pdf['PDF_eta'] = PDF_N.flatten()

    # SLOPES
    if isinstance(alpha_s,(list,np.ndarray)) == False:
        df_pdf['kappa'] = kappa
        df_pdf['alpha_s'] = alpha_s
        df_pdf['alpha_N'] = alpha_N
        df_pdf['alpha_n'] = alpha_n
        df_pdf['alpha_eta'] = alpha_eta
    elif isinstance(alpha_s,(list,np.ndarray)):
        df_pdf['kappa'] = np.repeat(kappa,size)
        df_pdf['alpha_s'] = np.repeat(alpha_s.flatten(),size)
        df_pdf['alpha_N'] = np.repeat(alpha_N.flatten(),size)
        df_pdf['alpha_n'] = np.repeat(alpha_n.flatten(),size)
        df_pdf['alpha_eta'] = np.repeat(alpha_eta.flatten(),size)
        df_pdf['model'] = np.repeat(nmodel,size)
    else:
        print('ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return
        
    # PDF INPUTS AND NORMALIZATION CONSTANTS
    if isinstance(sigma_s,(list,np.ndarray)) == False:
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = sigma_v
        df_pdf['mach'] = mach
        df_pdf['N_surface'] = N_surface
        df_pdf['N_src'] = N_src
        df_pdf['n_src'] = n_src
        df_pdf['Mc_Msun'] = Mc
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
        if lognorm == False:
            df_pdf['f_t'] = f_t
            # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
            # EVEN CALCULATED FOR LN-ONLY MODELS
            df_pdf['eta_t'] = eta_t
            df_pdf['s_t'] = s_t
            df_pdf['N_t cm-2'] = N_t
            df_pdf['log n_t'] = np.log10(n_t)
            df_pdf['r_t pc'] = r_t
        # SPATIAL SLOPE OF OUTER PL
        df_pdf['ks'] = ks
        # DENSITY CORRECTION
        df_pdf['fcorr'] = fcorr
        # CRITICAL DENSITY
        df_pdf['s_crit'] = s_crit
        # NORMALIZATION DENSITY & CORRESPONDING VALUES
        df_pdf['r_norm_pc'] = rnorm
        df_pdf['N_norm cm-2'] = Nnorm
        df_pdf['n_norm cm-3'] = nnorm
    elif isinstance(sigma_s,(list,np.ndarray)):
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = np.repeat(sigma_v.flatten(),size)
        df_pdf['mach'] = np.repeat(mach.flatten(),size)
        df_pdf['N_src'] = np.repeat(N_src.flatten(),size)
        df_pdf['n_src'] = np.repeat(n_src.flatten(),size)
        df_pdf['Mc_Msun'] = np.repeat(Mc.flatten(),size)
        # 3D PDF
        df_pdf['sigma_s'] = np.repeat(sigma_s.flatten(),size)
        df_pdf['sigma_n'] = np.repeat(sigma_n.flatten(),size)
        df_pdf['s_0'] = np.repeat(s_0.flatten(),size)
        df_pdf['C norm'] = np.repeat(C.flatten(),size)
        df_pdf['N norm'] = np.repeat(N.flatten(),size)
        # 2D PDF
        df_pdf['sigma_eta'] = np.repeat(sigma_eta.flatten(),size)
        df_pdf['sigma_N'] = np.repeat(sigma_N.flatten(),size)
        df_pdf['eta_0'] = np.repeat(eta_0.flatten(),size)
        df_pdf['N_N norm'] = np.repeat(C_N.flatten(),size)
        df_pdf['N_N norm'] = np.repeat(N_N.flatten(),size)
        # FRACTIONS
        df_pdf['fcrit'] = np.repeat(fcrit.flatten(),size)
        df_pdf['f_above_fix'] = np.repeat(ffix.flatten(),size)
        # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
        # EVEN CALCULATED FOR LN-ONLY MODELS
        df_pdf['eta_t'] = np.repeat(eta_t.flatten(),size)
        df_pdf['s_t'] = np.repeat(s_t.flatten(),size)
        df_pdf['N_t cm-2'] = np.repeat(N_t.flatten(),size)
        df_pdf['log n_t'] = np.log10(np.repeat(n_t.flatten(),size))
        df_pdf['r_t pc'] = np.repeat(r_t.flatten(),size)
        # SPATIAL SLOPE OF OUTER PL
        df_pdf['ks'] = np.repeat(ks.flatten(),size)
        # DENSITY CORRECTION
        df_pdf['fcorr'] = np.repeat(fcorr.flatten(),size)
        # CRITICAL DENSITY
        df_pdf['s_crit'] = np.repeat(s_crit.flatten(),size)
        # NORMALIZATION DENSITY & CORRESPONDING VALUES
        df_pdf['r_norm_pc'] = np.repeat(rnorm.flatten(),size)
        df_pdf['N_norm cm-2'] = np.repeat(Nnorm.flatten(),size)
        df_pdf['n_norm cm-3'] = np.repeat(nnorm.flatten(),size)
    else:
        print('ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # WE CAN SET VIRIAL PARAMETER MANUALLY
    # VIRIAL PARAMETER
    if isinstance(alpha_vir,(list,np.ndarray)) == False:
        df_pdf['alpha_vir'] = alpha_vir
    elif isinstance(alpha_vir,(list,np.ndarray)):
        df_pdf['alpha_vir'] = np.repeat(alpha_vir.flatten(),size)
    else:
        print('VIRIAL PARAMETER ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # SOURCE SIZE CAN BE A SINGLE VALUE EVEN IF THERE ARE MUTIPLE MODELS
    if isinstance(rs,(list,np.ndarray)) == False:
        df_pdf['rs_pc'] = rs
    elif isinstance(rs,(list,np.ndarray)):
        df_pdf['rs_pc'] = np.repeat(rs.flatten(),size)
    else:
        print('SOURCE SIZE ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # TEMP CAN BE A SINGLE VALUE EVEN IF THERE ARE MUTIPLE MODELS
    if isinstance(temp,(list,np.ndarray)) == False:
        df_pdf['temp'] = temp
        df_pdf['c_s'] = c_s
    elif isinstance(temp,(list,np.ndarray)):
        df_pdf['temp'] = np.repeat(temp.flatten(),size)
        df_pdf['c_s'] = np.repeat(c_s.flatten(),size)
    else:
        print('TEMP ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return
                
    return df_pdf

def get_PDFs2(sigma_v,
             temp,
             Sigma,
             rs,
             kappa=1.5,
             size=1000,
             b=0.4,
             beta=np.inf,
             n_ref = 10**4.5,
             mass_conserving=False,
             lognorm=False,
             s_crit=False,
             s_const=False,
             alpha_vir = 1.,
             correct_dens=False,
             user_nsrc = -1,
             r_ref = 1.):


    N_surface = Sigma*u.M_sun/u.pc**2/(2.33*(const.m_p+const.m_e))
    N_surface = N_surface.to(u.cm**-2).value
    
    sigma_v = np.sqrt(3)*sigma_v
    c_s = sound_speed(temp)
    mach = sigma_v/c_s
    alpha_s, alpha_N, alpha_n, alpha_eta = calc_slopes(kappa)

    if isinstance(beta,np.ndarray):
        sigma_s = LN_width_alfven(b, mach, beta)
    else:
        if np.isfinite(beta):
            sigma_s = LN_width_alfven(b, mach, beta)
        else:
            sigma_s = LN_width(b, mach)

    # 3D + 2D - LOG + LINEAR STD
    sigma_n = np.sqrt(np.exp(sigma_s**2)-1)
    A = 0.11
    sigma_eta = np.sqrt(A)*sigma_s
    sigma_N = np.sqrt(np.exp(sigma_eta**2)-1)
    ks = 2*sigma_n/(sigma_N+sigma_n)
    kappa_N = 3./alpha_eta

    Mc_const = 4*np.pi*Sigma*u.M_sun/u.pc**2*(rs*u.pc)**2
    Mc_const = Mc_const.to(u.M_sun)
    rho_src = Mc_const/(4/3*np.pi)/(rs*u.pc)**3
    rho_src = rho_src.to(u.kg/u.cm**3).value
    n_surface = rho_src/(2.33*(const.m_p+const.m_e)).value
    Mc_const = Mc_const.value
    
    n_src = n_surface*sigma_n**2/2
    N_src = N_surface*sigma_N**2/2
    Mc = 4/3*np.pi*(n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
    Mc = Mc.to(u.M_sun).value
    fcorr = Mc_const/Mc
 
    if isinstance(user_nsrc,(list,np.ndarray)):
        n_src = user_nsrc
    elif isinstance(user_nsrc,int):
        if (user_nsrc != -1):
            n_src = user_nsrc

    rho_surface = n_surface*(2.33*(const.m_p+const.m_e))*u.cm**-3
    rho_surface = rho_surface.to(u.kg/u.cm**3).value

    if correct_dens == True:
        if lognorm:
            fcorr = (1-ks/3)/(1-ks/5)
        if isinstance(n_surface,(list,np.ndarray)):
            n_src = np.hstack(n_surface)/np.hstack(fcorr)
            N_src = np.hstack(N_surface)/np.hstack(fcorr)
            #alpha_vir = alpha_vir*np.hstack(fcorr)
        else:
            n_src = n_surface/fcorr
            N_src = N_surface/fcorr
            #alpha_vir = alpha_vir*fcorr
    else:
        if isinstance(Sigma,(list,np.ndarray)):
            N_src = np.hstack(N_surface)
            n_src = np.hstack(n_surface)
        else:
            N_src = N_surface
            n_src = n_surface

    r0 = (N_src/n_src)*u.cm.to(u.pc)

    # TRANSITION DENSITY
    eta_t = 0.5*(2*abs(alpha_eta)-1)*sigma_eta**2
    N_t = N_src*np.exp(eta_t)
    s_t = transition_density(alpha_s,sigma_s)
    n_t = n_src*np.exp(s_t)
    r_t = (N_t/n_t)*u.cm.to(u.pc)

    try:
        C = ((n_t/n_surface)*(r_t/rs)**(kappa/(ks-kappa)))
        C2 = ((N_t/N_surface)*(r_t/rs)**((kappa+1)/(ks-kappa)))
    except:
        C=0.
        C2 = 0.
    if lognorm:
        C=0
        C2=0.
    rnorm = abs(rs - C*r_t)/abs(C-1)
    rnorm2 = abs(rs - C2*r_t)/abs(C2-1)
    nnorm = n_t*(r_t/rnorm)**-kappa*(1+r_t/rnorm)**(kappa-ks)
    #Nnorm = nnorm*u.cm**-3*rnorm*u.pc
    #Nnorm = Nnorm.to(u.cm**-2).value
    Nnorm = N_t*(r_t/rnorm)**-(kappa+1)*(1+r_t/rnorm)**(kappa-ks)
    #Nnorm = N_t*(r_t/rnorm2)**-(kappa+1)*(1+r_t/rnorm2)**(kappa-ks)
    #print(rnorm2,Nnorm)

    #rnorm = rs*r_t/r_t
    #nnorm = n_surface #n_t

    # 3D
    if isinstance(n_src,(list,np.ndarray)):
        sigma_s = sigma_s.reshape(-1,1)
        s_t = s_t.reshape(-1,1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)

    # 2D
    if isinstance(sigma_eta,(list,np.ndarray)):
        sigma_eta = sigma_eta.reshape(-len(sigma_eta),1)
        eta_t = eta_t.reshape(-len(eta_t),1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)
        alpha_eta = alpha_eta.reshape(-1,1)

    

    if lognorm:
        nmin = 1e-1
        nmax = 1e11
        
        if isinstance(n_src,(list,np.ndarray)):
            # assumes r0 cons
            n_src = n_src.reshape(-1,1)
            N_src = N_src.reshape(-1,1)
            N_surface = N_surface.reshape(-1,1)
            ks = ks.reshape(-1,1)
            if isinstance(rs,(list,np.ndarray)):
                rs = rs.reshape(-1,1) #.flatten()
            n_array = np.logspace(np.log10(nmin),np.log10(nmax),size)
            r = rs*(n_array/n_src)**(1/-ks)
            n_array = (n_src)*(r/rs)**(-ks)
            N_array = (N_src)*(r/rs)**(-ks+1)
            s_array = np.log(n_array/(n_src))
            eta = np.log(N_array/N_src)
            n_src = n_src.flatten()
            N_src = N_src.flatten()
            N_surface = N_surface.flatten()
            ks = ks.flatten()
        else:
            n_array = np.logspace(np.log10(nmin),np.log10(nmax),size)
            r = rs*(n_array/n_src)**(-1/ks)
            n_array = n_src*(r/rs)**(-ks)
            N_array = N_src*(r/rs)**(-ks+1)
            s_array = np.log(n_array/n_src)
            eta = np.log(N_array/N_src)

    
    if lognorm == False:
        nmin = 1e-2
        nmax = 1e10
        
        if isinstance(n_src,(list,np.ndarray)):
            nnorm = nnorm.flatten()
            rnorm = rnorm.flatten()
            ks = ks.flatten()
            if isinstance(kappa,(np.ndarray,list)):
                kappa = kappa.flatten()
            r = np.logspace(6,-6,1000).reshape(-1,1)
            n0_ = n_t*(r/r_t)**-kappa
            n1_ = n_surface*(r/rs)**-ks
            n_arr = 1/(1/n0_ + 1/n1_)
            i0 = np.argmin(abs(1-n_arr/nmin),axis=0)
            i1 = np.argmin(abs(1-n_arr/nmax),axis=0)
            rmax = r[i0].flatten()
            rmin = r[i1].flatten()
            r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
            nnorm = nnorm.reshape(-1,1)
            rnorm = rnorm.reshape(-1,1)
            if isinstance(kappa,(np.ndarray,list)):
                kappa = kappa.reshape(-1,1)
            ks = ks.reshape(-1,1)
            n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
            imax = np.argmax(n_array,axis=1)
            imin = np.argmin(n_array,axis=1)
            rmax = r[np.arange(len(imin)),imin]
            rmin = r[np.arange(len(imax)),imax]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
            while np.any(abs(np.log10(np.nanmax(n_array,axis=1)/nmax)) > 0.1) or np.any(abs(np.log10(np.nanmin(n_array,axis=1)/nmin)) > 0.1):
                n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
                imax = np.argmax(n_array,axis=1)
                imin = np.argmin(n_array,axis=1)
                cmax = np.nanmax(n_array,axis=1)/nmax
                cmin = np.nanmin(n_array,axis=1)/nmin
                rmax = r[np.arange(len(imin)),imin]
                rmin = r[np.arange(len(imax)),imax]
                i0 = abs(np.log10(cmax)) > 0.1
                i1 = abs(np.log10(cmin)) > 0.1
                rmax[i1] = (rmax*(cmin**0.33))[i1]
                rmin[i0] = (rmin*(cmax**0.33))[i0]
                r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
        
        else:
            r = np.logspace(6,-6,1000)
            n0_ = n_t*(r/r_t)**-kappa
            n1_ = n_src*(r/rs)**-ks
            n_arr = 1/(1/n0_ + 1/n1_)            
            i0 = np.argmin(abs(1-n_arr/nmin))
            i1 = np.argmin(abs(1-n_arr/nmax))
            rmax = r[i0]
            rmin = r[i1]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size)
            #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(ks-kappa)
            n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
            imax = np.argmax(n_array) # removed, axis=1
            imin = np.argmin(n_array)
            rmax = r[imin] #np.arange(len(imin)),imin]
            rmin = r[imax] #np.arange(len(imax)),imax]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size)
            while np.any(abs(np.log10(np.nanmax(n_array)/nmax)) > 1.) or np.any(abs(np.log10(np.nanmin(n_array)/nmin)) > 1.):
                # removed axis=1
                #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(ks-kappa)
                n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
                imax = np.argmax(n_array) #,axis=1)
                imin = np.argmin(n_array) #,axis=1)
                cmax = np.nanmax(n_array)/nmax # removed axis=1
                cmin = np.nanmin(n_array)/nmin # removed axis=1
                #print(cmin,cmax)
                rmax = r[imin] #[np.arange(len(imin)),imin]
                rmin = r[imax] #[np.arange(len(imax)),imax]
                i0 = np.argmin(abs(1-n_array/nmin)) # 
                i1 =  np.argmin(abs(1-n_array/nmax)) # abs(np.log10(cmin)) > 1.
                rmax = (rmax*(cmin**0.33)) # [i1]
                rmin = (rmin*(cmax**0.33)) # [i0]
                r = np.logspace(np.log10(rmax),np.log10(rmin),size)

        if isinstance(n_src,(list,np.ndarray)):
            Nnorm = Nnorm.reshape(-1,1)
            n_src = n_src.reshape(-1,1)
            N_src = N_src.reshape(-1,1)
            N_surface = N_surface.reshape(-1,1)
            rnorm2 = rnorm2.reshape(-1,1)

        #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
        #N_array = Nnorm*(r/rnorm2)**-(kappa+1)*(1+r/rnorm2)**(kappa-ks)
        ##N_array = Nnorm*(r/rnorm)**-(kappa+1)*(1+r/rnorm)**(kappa-ks)
        ##print(N_array)
        ##N_array = n_array*rnorm*(u.cm**-3*u.pc).to(u.cm**-2)
        #N_array = n_array*r0.reshape(-1,1)*(u.cm**-3*u.pc).to(u.cm**-2)
        #s_array = np.log(n_array/n_src)
        #eta = np.log(N_array/N_src)
        

        # TESTING
        #ks = abs(np.log(n_surface/n_t)/np.log(rs/r_t))
        #print(ks)
        
        if isinstance(n_t,(np.ndarray,list)):
            n_t = np.vstack(n_t)
            N_t = np.vstack(N_t)
            r_t = np.vstack(r_t)
            n_surface = np.vstack(n_surface)
            N_surface = np.vstack(N_surface)
            n_src = np.vstack(n_src)
            N_src = np.vstack(N_src)
            ks = np.vstack(ks)
           
        if isinstance(rs,(np.ndarray,list)):
            rs = np.vstack(rs)

        if isinstance(kappa,(np.ndarray,list)):
            kappa = np.vstack(kappa)

        if isinstance(n_src,(np.ndarray,list)):
            r = np.tile(np.logspace(3,-3,size),len(n_src)).reshape(len(n_src),size)
        else:
            r = np.logspace(3,-3,size)
            
        n_PL = n_t*(r/r_t)**-kappa
        n_env = n_t*(r/r_t)**-ks
        n_array = n_env
        n_array[r < r_t] = n_PL[r < r_t]
        s_array = np.log(n_array/n_src)
        #print(n_array)

        N_PL = N_t*(r/r_t)**-(kappa-1)
        if ks < 1:
            N_env = N_t*(r/r_t)**(ks-1)
        else:
            N_env = N_t*(r/r_t)**-(ks-1)
        N_array = N_env
        N_array[r < r_t] = N_PL[r < r_t]
        eta = np.log(N_array/N_src)

        # FIND TRUE RS
        ind = np.argmin(abs(1-n_array/n_surface))
        rs = r[ind]
        #print(rs)
        
           
    if isinstance(n_src,(list,np.ndarray)):
        sigma_s = sigma_s.reshape(-1,1)
        s_t = s_t.reshape(-1,1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)

    # 3D
    s_0 = mean_log_density(sigma_s)
    C = PL_amplitude(alpha_s,sigma_s)
    N = normalization(C,s_t,alpha_s,sigma_s)
    LN = lognormal(N,sigma_s,s_array,s_0)
    PL = power_law(N,C,alpha_s,s_array)
    if lognorm == False:
        PDF = combine_PL_LN(s_t,PL,LN,s_array)
    elif lognorm == True:
        PDF = LN
    
    
    if mass_conserving:
        s_s = conservation_shift(N,C,alpha_s,s_t,sigma_s)
        s_array = s_array+np.nan_to_num(s_s)
        
    # 2D
    if isinstance(sigma_eta,(list,np.ndarray)):
        sigma_eta = sigma_eta.reshape(-len(sigma_eta),1)
        eta_t = eta_t.reshape(-len(eta_t),1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_eta = alpha_eta.reshape(-1,1)

    # 2D
    eta_0 = -0.5*sigma_eta**2
    C_N = PL_amplitude(alpha_eta,sigma_eta)
    N_N = normalization(C_N,eta_t,alpha_eta,sigma_eta)
    LN_N = lognormal(N_N,sigma_eta,eta,eta_0)
    PL_N = power_law(N_N,C_N,alpha_eta,eta)
    PDF_N = combine_PL_LN(eta_t,PL_N,LN_N,eta)

    if mass_conserving:
        eta_s = conservation_shift(N_N,C_N,alpha_eta,eta_t,sigma_eta)
        eta = eta-np.nan_to_num(eta_s)

    if lognorm:
        s_fix = np.log(n_ref/n_src)
        if s_crit == 'KM05':
            s_crit = calc_scrit_KM05(mach, alpha_vir)
            if isinstance(s_crit,(np.ndarray,list)):
                fcrit = []
                #s_crit = s_crit.reshape(-len(s_crit),1)
                for st_, LN_, s_ in zip(s_crit,LN,s_array):
                    fcrit.append(mass_frac_above_crit(st_,LN_,s_))
                    #plt.plot(s_,LN_)
                fcrit = np.array(fcrit)
            else:
                fcrit = mass_frac_above_crit(s_crit,LN,s_array)
        elif s_crit == 'PN11':
            #theta = np.sqrt(1/(np.pi**2/15/0.067))
            s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
            if isinstance(s_crit,(np.ndarray,list)):
                fcrit = []
                #s_crit = s_crit.reshape(-len(s_crit),1)
                for st_, LN_, s_ in zip(s_crit,LN,s_array):
                    fcrit.append(mass_frac_above_crit(st_,LN_,s_))
                    #plt.plot(s_,LN_)
                fcrit = np.array(fcrit)
            else:
                fcrit = mass_frac_above_crit(s_crit,LN,s_array)
        if isinstance(s_fix,(np.ndarray,list)):
            ffix = []
            #s_fix = s_fix.reshape(-len(s_fix),1)
            for st_, LN_, s_ in zip(s_fix,LN,s_array):
                ffix.append(mass_frac_above_crit(st_,LN_,s_))
            ffix = np.array(ffix)
        else:
            ffix = mass_frac_above_crit(s_fix,LN,s_array)
        if s_const == True:
            s_crit = s_fix
            fcrit = ffix
                
            
    elif lognorm == False:
        #s_choice = s_crit
        s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
        s_fix = np.log(n_ref/n_src)
        fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)
        ffix = mass_frac_above_crit(s_fix,PDF,s_array)
        f_t = mass_frac_above_crit(s_t,PDF,s_array)
        if isinstance(s_crit,(np.ndarray,list)):
            fcrit = calc_fdense_Burk2019(sigma_s.reshape(-len(sigma_s),1),alpha_s)
            fcrit = np.array(fcrit)
        if isinstance(s_fix,(np.ndarray,list)):
            ffix = []
            s_fix = s_fix.reshape(-len(s_fix),1)
            for sf_,PDF_, s_ in zip(s_fix,PDF,s_array):
                ffix.append(mass_frac_above_crit(sf_,PDF_,s_))
            ffix = np.array(ffix)
        if isinstance(s_t,(np.ndarray,list)):
            f_t = []
            s_t = s_t.reshape(-len(s_t),1)
            for sf_,PDF_, s_ in zip(s_t,PDF,s_array):
                f_t.append(mass_frac_above_crit(sf_,PDF_,s_))
            f_t = np.array(f_t)
        else:
            fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)
            ffix = mass_frac_above_crit(s_fix,PDF,s_array)
            f_t = mass_frac_above_crit(s_t,PDF,s_array)
        if (s_const == True):
            fcrit = ffix

    ##############################
    # SAVING PDF(S) TO DATAFRAME #
    ##############################

    #plt.plot((s_array[100].T),np.log10(PDF[100].T))
    #plt.show()
        
    df_pdf = pd.DataFrame()
    if isinstance(n_src,(list,np.ndarray)):
        nmodel = np.arange(1,len(n_src.flatten())+1)
    elif isinstance(n_src,(list,np.ndarray)) == False:
        nmodel = 0.
    else:
        print('CANNOT PRODUCE MODEL INDEX')
        return


    # DENSITY ARRAYS
    df_pdf['s_array'] = s_array.flatten()
    df_pdf['eta'] = eta.flatten()
    df_pdf['log n_arr'] = np.log10(n_array.flatten())
    df_pdf['log N_arr'] = np.log10(N_array.flatten())
    # RADII
    df_pdf['r_arr_pc'] = r.flatten()
    

    # 3D PDF MODEL ARRAYS
    df_pdf['LN'] = LN.flatten()
    df_pdf['PL'] = PL.flatten()
    df_pdf['PDF'] = PDF.flatten()
    # 2D PDF MODEL ARRAYS
    df_pdf['LN_eta'] = LN_N.flatten()
    df_pdf['PL_eta'] = PL_N.flatten()
    df_pdf['PDF_eta'] = PDF_N.flatten()

    # SLOPES
    if isinstance(alpha_s,(list,np.ndarray)) == False:
        df_pdf['kappa'] = kappa
        df_pdf['alpha_s'] = alpha_s
        df_pdf['alpha_N'] = alpha_N
        df_pdf['alpha_n'] = alpha_n
        df_pdf['alpha_eta'] = alpha_eta
    elif isinstance(alpha_s,(list,np.ndarray)):
        df_pdf['kappa'] = np.repeat(kappa,size)
        df_pdf['alpha_s'] = np.repeat(alpha_s.flatten(),size)
        df_pdf['alpha_N'] = np.repeat(alpha_N.flatten(),size)
        df_pdf['alpha_n'] = np.repeat(alpha_n.flatten(),size)
        df_pdf['alpha_eta'] = np.repeat(alpha_eta.flatten(),size)
        df_pdf['model'] = np.repeat(nmodel,size)
    else:
        print('ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return
        
    # PDF INPUTS AND NORMALIZATION CONSTANTS
    if isinstance(sigma_s,(list,np.ndarray)) == False:
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = sigma_v
        df_pdf['mach'] = mach
        df_pdf['N_surface'] = N_surface
        df_pdf['N_src'] = N_src
        df_pdf['n_src'] = n_src
        df_pdf['Mc_Msun'] = Mc
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
        if lognorm == False:
            df_pdf['f_t'] = f_t
            # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
            # EVEN CALCULATED FOR LN-ONLY MODELS
            df_pdf['eta_t'] = eta_t
            df_pdf['s_t'] = s_t
            df_pdf['N_t cm-2'] = N_t
            df_pdf['log n_t'] = np.log10(n_t)
            df_pdf['r_t pc'] = r_t
        # SPATIAL SLOPE OF OUTER PL
        df_pdf['ks'] = ks
        # DENSITY CORRECTION
        df_pdf['fcorr'] = fcorr
        # CRITICAL DENSITY
        df_pdf['s_crit'] = s_crit
        # NORMALIZATION DENSITY & CORRESPONDING VALUES
        df_pdf['r_norm_pc'] = rnorm
        df_pdf['N_norm cm-2'] = Nnorm
        df_pdf['n_norm cm-3'] = nnorm
    elif isinstance(sigma_s,(list,np.ndarray)):
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = np.repeat(sigma_v.flatten(),size)
        df_pdf['mach'] = np.repeat(mach.flatten(),size)
        df_pdf['N_src'] = np.repeat(N_src.flatten(),size)
        df_pdf['n_src'] = np.repeat(n_src.flatten(),size)
        df_pdf['Mc_Msun'] = np.repeat(Mc.flatten(),size)
        # 3D PDF
        df_pdf['sigma_s'] = np.repeat(sigma_s.flatten(),size)
        df_pdf['sigma_n'] = np.repeat(sigma_n.flatten(),size)
        df_pdf['s_0'] = np.repeat(s_0.flatten(),size)
        df_pdf['C norm'] = np.repeat(C.flatten(),size)
        df_pdf['N norm'] = np.repeat(N.flatten(),size)
        # 2D PDF
        df_pdf['sigma_eta'] = np.repeat(sigma_eta.flatten(),size)
        df_pdf['sigma_N'] = np.repeat(sigma_N.flatten(),size)
        df_pdf['eta_0'] = np.repeat(eta_0.flatten(),size)
        df_pdf['N_N norm'] = np.repeat(C_N.flatten(),size)
        df_pdf['N_N norm'] = np.repeat(N_N.flatten(),size)
        # FRACTIONS
        df_pdf['fcrit'] = np.repeat(fcrit.flatten(),size)
        df_pdf['f_above_fix'] = np.repeat(ffix.flatten(),size)
        # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
        # EVEN CALCULATED FOR LN-ONLY MODELS
        df_pdf['eta_t'] = np.repeat(eta_t.flatten(),size)
        df_pdf['s_t'] = np.repeat(s_t.flatten(),size)
        df_pdf['N_t cm-2'] = np.repeat(N_t.flatten(),size)
        df_pdf['log n_t'] = np.log10(np.repeat(n_t.flatten(),size))
        df_pdf['r_t pc'] = np.repeat(r_t.flatten(),size)
        # SPATIAL SLOPE OF OUTER PL
        df_pdf['ks'] = np.repeat(ks.flatten(),size)
        # DENSITY CORRECTION
        df_pdf['fcorr'] = np.repeat(fcorr.flatten(),size)
        # CRITICAL DENSITY
        df_pdf['s_crit'] = np.repeat(s_crit.flatten(),size)
        # NORMALIZATION DENSITY & CORRESPONDING VALUES
        df_pdf['r_norm_pc'] = np.repeat(rnorm.flatten(),size)
        df_pdf['N_norm cm-2'] = np.repeat(Nnorm.flatten(),size)
        df_pdf['n_norm cm-3'] = np.repeat(nnorm.flatten(),size)
    else:
        print('ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # WE CAN SET VIRIAL PARAMETER MANUALLY
    # VIRIAL PARAMETER
    if isinstance(alpha_vir,(list,np.ndarray)) == False:
        df_pdf['alpha_vir'] = alpha_vir
    elif isinstance(alpha_vir,(list,np.ndarray)):
        df_pdf['alpha_vir'] = np.repeat(alpha_vir.flatten(),size)
    else:
        print('VIRIAL PARAMETER ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # SOURCE SIZE CAN BE A SINGLE VALUE EVEN IF THERE ARE MUTIPLE MODELS
    if isinstance(rs,(list,np.ndarray)) == False:
        df_pdf['rs_pc'] = rs
    elif isinstance(rs,(list,np.ndarray)):
        df_pdf['rs_pc'] = np.repeat(rs.flatten(),size)
    else:
        print('SOURCE SIZE ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # TEMP CAN BE A SINGLE VALUE EVEN IF THERE ARE MUTIPLE MODELS
    if isinstance(temp,(list,np.ndarray)) == False:
        df_pdf['temp'] = temp
        df_pdf['c_s'] = c_s
    elif isinstance(temp,(list,np.ndarray)):
        df_pdf['temp'] = np.repeat(temp.flatten(),size)
        df_pdf['c_s'] = np.repeat(c_s.flatten(),size)
    else:
        print('TEMP ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return
                
    return df_pdfs
    

def get_PDFs3(sigma_v,
             temp,
             Sigma,
             rs,
             kappa=1.5,
             size=1000,
             b=0.4,
             beta=np.inf,
             n_ref = 10**4.5,
             mass_conserving=False,
             lognorm=False,
             s_crit=False,
             s_const=False,
             alpha_vir = 1.,
             correct_dens=False,
             user_nsrc = -1,
             r_ref = 1.):


    N_surface = Sigma*u.M_sun/u.pc**2/(2.33*(const.m_p+const.m_e))
    N_surface = N_surface.to(u.cm**-2).value
    
    sigma_v = np.sqrt(3)*sigma_v
    c_s = sound_speed(temp)
    mach = sigma_v/c_s
    alpha_s, alpha_N, alpha_n, alpha_eta = calc_slopes(kappa)

    if isinstance(beta,np.ndarray):
        sigma_s = LN_width_alfven(b, mach, beta)
    else:
        if np.isfinite(beta):
            sigma_s = LN_width_alfven(b, mach, beta)
        else:
            sigma_s = LN_width(b, mach)

    # 3D + 2D - LOG + LINEAR STD
    sigma_n = np.sqrt(np.exp(sigma_s**2)-1)
    A = 0.11
    sigma_eta = np.sqrt(A)*sigma_s
    sigma_N = np.sqrt(np.exp(sigma_eta**2)-1)

    # MASS OF SPHERE WITH CONSTANT DENSITY FROM SURFACE DENSITY, SIZE
    Mc_const = 4*np.pi*Sigma*u.M_sun/u.pc**2*(rs*u.pc)**2
    Mc_const = Mc_const.to(u.M_sun)
    rho_surface = Mc_const/(4/3*np.pi)/(rs*u.pc)**3
    rho_surface = rho_surface.to(u.kg/u.cm**3).value
    n_surface = rho_surface/(2.33*(const.m_p+const.m_e)).value
    Mc_const = Mc_const.value

    #n_surface = (3 - kappa) * N_surface / (rs * u.pc.to(u.cm))
    #rho_surface = n_surface * (2.33*(const.m_p+const.m_e)).value
    #M_const = 4 * np.pi / (3 - kappa) * 

    if correct_dens == True:
        # MEAN DENSITY FROM SURFACE DENSITY AND WIDTH OF PDFS
        n_src = n_surface*sigma_n**2/2
        N_src = N_surface*sigma_N**2/2
        #n_src = n_surface*sigma_n/2
        #N_src = N_surface*sigma_N/2
        #n_src = n_surface*np.exp(sigma_s)**2/2
        #N_src = N_surface*np.exp(sigma_eta)**2/2
        
        # MASS OF SPHERE WITH CONSTANT DENSITY FROM MEAN DENSITY
        Mc = 4/3*np.pi*(n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
        Mc = Mc.to(u.M_sun).value
    else:
        n_src = n_surface
        N_src = N_surface
        Mc = Mc_const

    if isinstance(user_nsrc,(list,np.ndarray)):
        n_src = user_nsrc
        # MASS OF SPHERE WITH CONSTANT DENSITY FROM MEAN DENSITY
        Mc = 4/3*np.pi*(n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
        Mc = Mc.to(u.M_sun).value
    elif isinstance(user_nsrc,int):
        if (user_nsrc != -1):
            n_src = user_nsrc
            # MASS OF SPHERE WITH CONSTANT DENSITY FROM MEAN DENSITY
            Mc = 4/3*np.pi*(n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
            Mc = Mc.to(u.M_sun).value
        
    fcorr = Mc_const/Mc
    n_src = n_surface/fcorr
    N_src = N_surface/fcorr
    rho_src = n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3

    # TRANSITION DENSITY
    eta_t = 0.5*(2*abs(alpha_eta)-1)*sigma_eta**2
    N_t = N_src*np.exp(eta_t)
    s_t = transition_density(alpha_s,sigma_s)
    n_t = n_src*np.exp(s_t)
    r_t = (N_t/n_t)*u.cm.to(u.pc)

    rnorm = rs
    nnorm = n_surface
    Nnorm = N_surface

    # 3D
    if isinstance(n_src,(list,np.ndarray)):
        sigma_s = sigma_s.reshape(-1,1)
        s_t = s_t.reshape(-1,1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)

    # 2D
    if isinstance(sigma_eta,(list,np.ndarray)):
        sigma_eta = sigma_eta.reshape(-len(sigma_eta),1)
        eta_t = eta_t.reshape(-len(eta_t),1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)
        alpha_eta = alpha_eta.reshape(-1,1)

    if lognorm:
        nmin = 1e-2
        nmax = 1e10
        
        if isinstance(n_src,(list,np.ndarray)):
            # assumes r0 cons
            n_src = n_src.reshape(-1,1)
            N_src = N_src.reshape(-1,1)
            N_surface = N_surface.reshape(-1,1)
            ks = ks.reshape(-1,1)
            if isinstance(rs,(list,np.ndarray)):
                rs = rs.reshape(-1,1) #.flatten()
            n_array = np.logspace(np.log10(nmin),np.log10(nmax),size)
            r = rs*(n_array/n_src)**(1/-ks)
            n_array = (n_src)*(r/rs)**(-ks)
            N_array = (N_src)*(r/rs)**(-ks+1)
            s_array = np.log(n_array/(n_src))
            eta = np.log(N_array/N_src)
            n_src = n_src.flatten()
            N_src = N_src.flatten()
            N_surface = N_surface.flatten()
            ks = ks.flatten()
        else:
            n_array = np.logspace(np.log10(nmin),np.log10(nmax),size)
            r = rs*(n_array/n_src)**(-1/ks)
            n_array = n_src*(r/rs)**(-ks)
            N_array = N_src*(r/rs)**(-ks+1)
            s_array = np.log(n_array/n_src)
            eta = np.log(N_array/N_src)

    
    if lognorm == False:
        nmin = 1e-1
        nmax = 1e10
        
        if isinstance(n_src,(list,np.ndarray)):
            print('TEST')
            n_surface = n_surface.flatten()
            #ks = ks.flatten()
            if isinstance(kappa,(np.ndarray,list)):
                kappa = kappa.flatten()
            r = np.logspace(6,-6,1000).reshape(-1,1)
            #n0_ = n_t*(r/r_t)**-kappa
            #n1_ = n_surface*(r/rs)**-ks
            #n_arr = 1/(1/n0_ + 1/n1_)
            n_arr = n_surface*(r/rs)**-kappa
            i0 = np.argmin(abs(1-n_arr/nmin),axis=0)
            i1 = np.argmin(abs(1-n_arr/nmax),axis=0)
            rmax = r[i0].flatten()
            rmin = r[i1].flatten()
            r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
            nnorm = nnorm.reshape(-1,1)
            rnorm = rnorm.reshape(-1,1)
            if isinstance(kappa,(np.ndarray,list)):
                kappa = kappa.reshape(-1,1)
            #ks = ks.reshape(-1,1)
            #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
            n_array = n_surface*(r/rs)**-kappa
            imax = np.argmax(n_array,axis=1)
            imin = np.argmin(n_array,axis=1)
            rmax = r[np.arange(len(imin)),imin]
            rmin = r[np.arange(len(imax)),imax]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
            while np.any(abs(np.log10(np.nanmax(n_array,axis=1)/nmax)) > 0.1) or np.any(abs(np.log10(np.nanmin(n_array,axis=1)/nmin)) > 0.1):
                #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
                n_array = n_surface*(r/rs)**-kappa
                imax = np.argmax(n_array,axis=1)
                imin = np.argmin(n_array,axis=1)
                cmax = np.nanmax(n_array,axis=1)/nmax
                cmin = np.nanmin(n_array,axis=1)/nmin
                rmax = r[np.arange(len(imin)),imin]
                rmin = r[np.arange(len(imax)),imax]
                i0 = abs(np.log10(cmax)) > 0.1
                i1 = abs(np.log10(cmin)) > 0.1
                rmax[i1] = (rmax*(cmin**0.33))[i1]
                rmin[i0] = (rmin*(cmax**0.33))[i0]
                r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
        
        else:
            r = np.logspace(6,-6,1000)
            #n0_ = n_t*(r/r_t)**-kappa
            #n1_ = n_src*(r/rs)**-ks
            #n_arr = 1/(1/n0_ + 1/n1_)
            n_arr = n_surface*(r/rs)**-kappa
            i0 = np.argmin(abs(1-n_arr/nmin))
            i1 = np.argmin(abs(1-n_arr/nmax))
            rmax = r[i0]
            rmin = r[i1]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size)
            #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(ks-kappa)
            #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
            n_array = n_surface*(r/rs)**-kappa
            imax = np.argmax(n_array) # removed, axis=1
            imin = np.argmin(n_array)
            rmax = r[imin] #np.arange(len(imin)),imin]
            rmin = r[imax] #np.arange(len(imax)),imax]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size)
            while np.any(abs(np.log10(np.nanmax(n_array)/nmax)) > 1.) or np.any(abs(np.log10(np.nanmin(n_array)/nmin)) > 1.):
                # removed axis=1
                #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(ks-kappa)
                #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
                n_array = n_surface*(r/rs)**-kappa
                imax = np.argmax(n_array) #,axis=1)
                imin = np.argmin(n_array) #,axis=1)
                cmax = np.nanmax(n_array)/nmax # removed axis=1
                cmin = np.nanmin(n_array)/nmin # removed axis=1
                #print(cmin,cmax)
                rmax = r[imin] #[np.arange(len(imin)),imin]
                rmin = r[imax] #[np.arange(len(imax)),imax]
                i0 = np.argmin(abs(1-n_array/nmin)) # 
                i1 =  np.argmin(abs(1-n_array/nmax)) # abs(np.log10(cmin)) > 1.
                rmax = (rmax*(cmin**0.33)) # [i1]
                rmin = (rmin*(cmax**0.33)) # [i0]
                r = np.logspace(np.log10(rmax),np.log10(rmin),size)

        if isinstance(n_src,(list,np.ndarray)):
            Nnorm = Nnorm.reshape(-1,1)
            n_src = n_src.reshape(-1,1)
            N_src = N_src.reshape(-1,1)
            N_surface = N_surface.reshape(-1,1)
        
        if isinstance(n_t,(np.ndarray,list)):
            n_t = np.vstack(n_t)
            N_t = np.vstack(N_t)
            r_t = np.vstack(r_t)
            n_surface = np.vstack(n_surface)
            N_surface = np.vstack(N_surface)
            n_src = np.vstack(n_src)
            N_src = np.vstack(N_src)
            #ks = np.vstack(ks)
           
        if isinstance(rs,(np.ndarray,list)):
            rs = np.vstack(rs)

        if isinstance(kappa,(np.ndarray,list)):
            kappa = np.vstack(kappa)

        if isinstance(n_src,(np.ndarray,list)):
            r = np.tile(np.logspace(3,-3,size),len(n_src)).reshape(len(n_src),size)
        else:
            r = np.logspace(3,-3,size)

        n_array = n_surface*(r/rs)**-kappa
        s_array = np.log(n_array/n_src)

        N_array = N_surface*(r/rs)**-(kappa+1)
        eta = np.log(N_array/N_src)
           
    if isinstance(n_src,(list,np.ndarray)):
        sigma_s = sigma_s.reshape(-1,1)
        s_t = s_t.reshape(-1,1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)

    # 3D
    s_0 = mean_log_density(sigma_s)
    C = PL_amplitude(alpha_s,sigma_s)
    N = normalization(C,s_t,alpha_s,sigma_s)
    LN = lognormal(N,sigma_s,s_array,s_0)
    PL = power_law(N,C,alpha_s,s_array)
    if lognorm == False:
        PDF = combine_PL_LN(s_t,PL,LN,s_array)
    elif lognorm == True:
        PDF = LN
    
    if mass_conserving:
        s_s = conservation_shift(N,C,alpha_s,s_t,sigma_s)
        s_array = s_array+np.nan_to_num(s_s)
        
    # 2D
    if isinstance(sigma_eta,(list,np.ndarray)):
        sigma_eta = sigma_eta.reshape(-len(sigma_eta),1)
        eta_t = eta_t.reshape(-len(eta_t),1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_eta = alpha_eta.reshape(-1,1)

    # 2D
    eta_0 = -0.5*sigma_eta**2
    C_N = PL_amplitude(alpha_eta,sigma_eta)
    N_N = normalization(C_N,eta_t,alpha_eta,sigma_eta)
    LN_N = lognormal(N_N,sigma_eta,eta,eta_0)
    PL_N = power_law(N_N,C_N,alpha_eta,eta)
    PDF_N = combine_PL_LN(eta_t,PL_N,LN_N,eta)

    if mass_conserving:
        eta_s = conservation_shift(N_N,C_N,alpha_eta,eta_t,sigma_eta)
        eta = eta-np.nan_to_num(eta_s)

    if lognorm:
        s_fix = np.log(n_ref/n_src)
        if s_crit == 'KM05':
            s_crit = calc_scrit_KM05(mach, alpha_vir)
            if isinstance(s_crit,(np.ndarray,list)):
                fcrit = []
                #s_crit = s_crit.reshape(-len(s_crit),1)
                for st_, LN_, s_ in zip(s_crit,LN,s_array):
                    fcrit.append(mass_frac_above_crit(st_,LN_,s_))
                    #plt.plot(s_,LN_)
                fcrit = np.array(fcrit)
            else:
                fcrit = mass_frac_above_crit(s_crit,LN,s_array)
        elif s_crit == 'PN11':
            #theta = np.sqrt(1/(np.pi**2/15/0.067))
            s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
            if isinstance(s_crit,(np.ndarray,list)):
                fcrit = []
                #s_crit = s_crit.reshape(-len(s_crit),1)
                for st_, LN_, s_ in zip(s_crit,LN,s_array):
                    fcrit.append(mass_frac_above_crit(st_,LN_,s_))
                    #plt.plot(s_,LN_)
                fcrit = np.array(fcrit)
            else:
                fcrit = mass_frac_above_crit(s_crit,LN,s_array)
        if isinstance(s_fix,(np.ndarray,list)):
            ffix = []
            #s_fix = s_fix.reshape(-len(s_fix),1)
            for st_, LN_, s_ in zip(s_fix,LN,s_array):
                ffix.append(mass_frac_above_crit(st_,LN_,s_))
            ffix = np.array(ffix)
        else:
            ffix = mass_frac_above_crit(s_fix,LN,s_array)
        if s_const == True:
            s_crit = s_fix
            fcrit = ffix
                
            
    elif lognorm == False:
        #s_choice = s_crit
        s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
        s_fix = np.log(n_ref/n_src)
        fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)
        ffix = mass_frac_above_crit(s_fix,PDF,s_array)
        f_t = mass_frac_above_crit(s_t,PDF,s_array)
        if isinstance(s_crit,(np.ndarray,list)):
            fcrit = calc_fdense_Burk2019(sigma_s.reshape(-len(sigma_s),1),alpha_s)
            fcrit = np.array(fcrit)
        if isinstance(s_fix,(np.ndarray,list)):
            ffix = []
            s_fix = s_fix.reshape(-len(s_fix),1)
            for sf_,PDF_, s_ in zip(s_fix,PDF,s_array):
                ffix.append(mass_frac_above_crit(sf_,PDF_,s_))
            ffix = np.array(ffix)
        if isinstance(s_t,(np.ndarray,list)):
            f_t = []
            s_t = s_t.reshape(-len(s_t),1)
            for sf_,PDF_, s_ in zip(s_t,PDF,s_array):
                f_t.append(mass_frac_above_crit(sf_,PDF_,s_))
            f_t = np.array(f_t)
        else:
            fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)
            ffix = mass_frac_above_crit(s_fix,PDF,s_array)
            f_t = mass_frac_above_crit(s_t,PDF,s_array)
        if (s_const == True):
            fcrit = ffix

    ##############################
    # SAVING PDF(S) TO DATAFRAME #
    ##############################

        
    df_pdf = pd.DataFrame()
    if isinstance(n_src,(list,np.ndarray)):
        nmodel = np.arange(1,len(n_src.flatten())+1)
    elif isinstance(n_src,(list,np.ndarray)) == False:
        nmodel = 0.
    else:
        print('CANNOT PRODUCE MODEL INDEX')
        return


    # DENSITY ARRAYS
    df_pdf['s_array'] = s_array.flatten()
    df_pdf['eta'] = eta.flatten()
    df_pdf['log n_arr'] = np.log10(n_array.flatten())
    df_pdf['log N_arr'] = np.log10(N_array.flatten())
    # RADII
    df_pdf['r_arr_pc'] = r.flatten()
    

    # 3D PDF MODEL ARRAYS
    df_pdf['LN'] = LN.flatten()
    df_pdf['PL'] = PL.flatten()
    df_pdf['PDF'] = PDF.flatten()
    # 2D PDF MODEL ARRAYS
    df_pdf['LN_eta'] = LN_N.flatten()
    df_pdf['PL_eta'] = PL_N.flatten()
    df_pdf['PDF_eta'] = PDF_N.flatten()

    # SLOPES
    if isinstance(alpha_s,(list,np.ndarray)) == False:
        df_pdf['kappa'] = kappa
        df_pdf['alpha_s'] = alpha_s
        df_pdf['alpha_N'] = alpha_N
        df_pdf['alpha_n'] = alpha_n
        df_pdf['alpha_eta'] = alpha_eta
    elif isinstance(alpha_s,(list,np.ndarray)):
        df_pdf['kappa'] = np.repeat(kappa,size)
        df_pdf['alpha_s'] = np.repeat(alpha_s.flatten(),size)
        df_pdf['alpha_N'] = np.repeat(alpha_N.flatten(),size)
        df_pdf['alpha_n'] = np.repeat(alpha_n.flatten(),size)
        df_pdf['alpha_eta'] = np.repeat(alpha_eta.flatten(),size)
        df_pdf['model'] = np.repeat(nmodel,size)
    else:
        print('ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return
        
    # PDF INPUTS AND NORMALIZATION CONSTANTS
    if isinstance(sigma_s,(list,np.ndarray)) == False:
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = sigma_v
        df_pdf['mach'] = mach
        df_pdf['N_surface'] = N_surface
        df_pdf['N_src'] = N_src
        df_pdf['n_src'] = n_src
        df_pdf['Mc_Msun'] = Mc
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
        if lognorm == False:
            df_pdf['f_t'] = f_t
            # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
            # EVEN CALCULATED FOR LN-ONLY MODELS
            df_pdf['eta_t'] = eta_t
            df_pdf['s_t'] = s_t
            df_pdf['N_t cm-2'] = N_t
            df_pdf['log n_t'] = np.log10(n_t)
            df_pdf['r_t pc'] = r_t
        # SPATIAL SLOPE OF OUTER PL
        #df_pdf['ks'] = ks
        # DENSITY CORRECTION
        df_pdf['fcorr'] = fcorr
        # CRITICAL DENSITY
        df_pdf['s_crit'] = s_crit
        # NORMALIZATION DENSITY & CORRESPONDING VALUES
        df_pdf['r_norm_pc'] = rnorm
        df_pdf['N_norm cm-2'] = Nnorm
        df_pdf['n_norm cm-3'] = nnorm
    elif isinstance(sigma_s,(list,np.ndarray)):
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = np.repeat(sigma_v.flatten(),size)
        df_pdf['mach'] = np.repeat(mach.flatten(),size)
        df_pdf['N_src'] = np.repeat(N_src.flatten(),size)
        df_pdf['n_src'] = np.repeat(n_src.flatten(),size)
        df_pdf['Mc_Msun'] = np.repeat(Mc.flatten(),size)
        # 3D PDF
        df_pdf['sigma_s'] = np.repeat(sigma_s.flatten(),size)
        df_pdf['sigma_n'] = np.repeat(sigma_n.flatten(),size)
        df_pdf['s_0'] = np.repeat(s_0.flatten(),size)
        df_pdf['C norm'] = np.repeat(C.flatten(),size)
        df_pdf['N norm'] = np.repeat(N.flatten(),size)
        # 2D PDF
        df_pdf['sigma_eta'] = np.repeat(sigma_eta.flatten(),size)
        df_pdf['sigma_N'] = np.repeat(sigma_N.flatten(),size)
        df_pdf['eta_0'] = np.repeat(eta_0.flatten(),size)
        df_pdf['N_N norm'] = np.repeat(C_N.flatten(),size)
        df_pdf['N_N norm'] = np.repeat(N_N.flatten(),size)
        # FRACTIONS
        df_pdf['fcrit'] = np.repeat(fcrit.flatten(),size)
        df_pdf['f_above_fix'] = np.repeat(ffix.flatten(),size)
        # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
        # EVEN CALCULATED FOR LN-ONLY MODELS
        df_pdf['eta_t'] = np.repeat(eta_t.flatten(),size)
        df_pdf['s_t'] = np.repeat(s_t.flatten(),size)
        df_pdf['N_t cm-2'] = np.repeat(N_t.flatten(),size)
        df_pdf['log n_t'] = np.log10(np.repeat(n_t.flatten(),size))
        df_pdf['r_t pc'] = np.repeat(r_t.flatten(),size)
        # SPATIAL SLOPE OF OUTER PL
        #df_pdf['ks'] = np.repeat(ks.flatten(),size)
        # DENSITY CORRECTION
        df_pdf['fcorr'] = np.repeat(fcorr.flatten(),size)
        # CRITICAL DENSITY
        df_pdf['s_crit'] = np.repeat(s_crit.flatten(),size)
        # NORMALIZATION DENSITY & CORRESPONDING VALUES
        df_pdf['r_norm_pc'] = np.repeat(rnorm.flatten(),size)
        df_pdf['N_norm cm-2'] = np.repeat(Nnorm.flatten(),size)
        df_pdf['n_norm cm-3'] = np.repeat(nnorm.flatten(),size)
    else:
        print('ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # WE CAN SET VIRIAL PARAMETER MANUALLY
    # VIRIAL PARAMETER
    if isinstance(alpha_vir,(list,np.ndarray)) == False:
        df_pdf['alpha_vir'] = alpha_vir
    elif isinstance(alpha_vir,(list,np.ndarray)):
        df_pdf['alpha_vir'] = np.repeat(alpha_vir.flatten(),size)
    else:
        print('VIRIAL PARAMETER ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # SOURCE SIZE CAN BE A SINGLE VALUE EVEN IF THERE ARE MUTIPLE MODELS
    if isinstance(rs,(list,np.ndarray)) == False:
        df_pdf['rs_pc'] = rs
    elif isinstance(rs,(list,np.ndarray)):
        df_pdf['rs_pc'] = np.repeat(rs.flatten(),size)
    else:
        print('SOURCE SIZE ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # TEMP CAN BE A SINGLE VALUE EVEN IF THERE ARE MUTIPLE MODELS
    if isinstance(temp,(list,np.ndarray)) == False:
        df_pdf['temp'] = temp
        df_pdf['c_s'] = c_s
    elif isinstance(temp,(list,np.ndarray)):
        df_pdf['temp'] = np.repeat(temp.flatten(),size)
        df_pdf['c_s'] = np.repeat(c_s.flatten(),size)
    else:
        print('TEMP ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return
                
    return df_pdf

def get_PDFs5(sigma_v,
             temp,
             Sigma,
             rs,
             kappa=1.5,
             size=1000,
             b=0.4,
             beta=np.inf,
             n_ref = 10**4.5,
             mass_conserving=False,
             lognorm=False,
             s_crit=False,
             s_const=False,
             alpha_vir = 1.,
             correct_dens=False,
             user_nsrc = -1,
             r_ref = 1.):


    N_surface = Sigma*u.M_sun/u.pc**2/(2.33*(const.m_p+const.m_e))
    N_surface = N_surface.to(u.cm**-2).value
    
    sigma_v = np.sqrt(3)*sigma_v
    c_s = sound_speed(temp)
    mach = sigma_v/c_s
    alpha_s, alpha_N, alpha_n, alpha_eta = calc_slopes(kappa)

    if isinstance(beta,np.ndarray):
        sigma_s = LN_width_alfven(b, mach, beta)
    else:
        if np.isfinite(beta):
            sigma_s = LN_width_alfven(b, mach, beta)
        else:
            sigma_s = LN_width(b, mach)

    # 3D + 2D - LOG + LINEAR STD
    sigma_n = np.sqrt(np.exp(sigma_s**2)-1)
    A = 0.11
    sigma_eta = np.sqrt(A)*sigma_s
    sigma_N = np.sqrt(np.exp(sigma_eta**2)-1)

    # MASS OF SPHERE WITH CONSTANT DENSITY FROM SURFACE DENSITY, SIZE
    #Mc_const = 4*np.pi*Sigma*u.M_sun/u.pc**2*(rs*u.pc)**2
    #Mc_const = Mc_const.to(u.M_sun)
    #rho_surface = Mc_const/(4/3*np.pi)/(rs*u.pc)**3
    #rho_surface = rho_surface.to(u.kg/u.cm**3).value
    #n_surface = rho_surface/(2.33*(const.m_p+const.m_e)).value
    #Mc_const = Mc_const.value

    kmpers = 1e5 # convert km/s to cm/s
    gcm2 = 0.000208836 # convert Msun / pc^2 to g / cm^2
    pc = 3.086*1e18 # convert pc to cm
    G = 6.67384 * 1e-8
    userSigma = Sigma * gcm2
    userveldisp = sigma_v * kmpers
    useravir = alpha_vir
    userR = rs * pc

    P = useravir * userSigma**2 *3 *np.pi * G / 20
    rho_surface = P / userveldisp**2
    n_surface = rho_surface/(2.33*(const.m_p+const.m_e)).value

    Mc_const = 4/3*np.pi*(n_surface*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
    Mc_const = Mc_const.to(u.M_sun).value

    if correct_dens == True:
        # MEAN DENSITY FROM SURFACE DENSITY AND WIDTH OF PDFS
        #if sigma_n**2 > 1:
        n_src = n_surface*sigma_n**2/2
        N_src = N_surface*sigma_N**2/2
        # TOO SHALLOW ALPHA_CO SLOPE
        #if sigma_n > 1:
        #n_src = n_surface*sigma_n/2
        #N_src = N_surface*sigma_N/2
        #n_src = n_surface*np.exp(sigma_s)**2/2
        #N_src = N_surface*np.exp(sigma_eta)**2/2
        
        # MASS OF SPHERE WITH CONSTANT DENSITY FROM MEAN DENSITY
        Mc = 4/3*np.pi*(n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
        Mc = Mc.to(u.M_sun).value
    else:
        n_src = n_surface
        N_src = N_surface
        Mc = Mc_const

    '''
    # TOO SHALLOW ALPHA_CO SLOPE
    if correct_dens == True:
        # MASS OF SPHERE WITH SLOPE
        Mc = 4*np.pi/(3-kappa)*(n_surface*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
        Mc = Mc.to(u.M_sun).value
    else:
        n_src = n_surface
        N_src = N_surface
        Mc = Mc_const
    '''
    

    if isinstance(user_nsrc,(list,np.ndarray)):
        n_src = user_nsrc
        # MASS OF SPHERE WITH CONSTANT DENSITY FROM MEAN DENSITY
        Mc = 4/3*np.pi*(n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
        Mc = Mc.to(u.M_sun).value
    elif isinstance(user_nsrc,int):
        if (user_nsrc != -1):
            n_src = user_nsrc
            # MASS OF SPHERE WITH CONSTANT DENSITY FROM MEAN DENSITY
            Mc = 4/3*np.pi*(n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
            Mc = Mc.to(u.M_sun).value
        
    fcorr = Mc_const/Mc
    n_src = n_surface/fcorr
    N_src = N_surface/fcorr
    rho_src = n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3

    # TRANSITION DENSITY
    eta_t = 0.5*(2*abs(alpha_eta)-1)*sigma_eta**2
    N_t = N_src*np.exp(eta_t)
    s_t = transition_density(alpha_s,sigma_s)
    n_t = n_src*np.exp(s_t)
    r_t = (N_t/n_t)*u.cm.to(u.pc)

    rnorm = rs
    nnorm = n_surface
    Nnorm = N_surface

    # 3D
    if isinstance(n_src,(list,np.ndarray)):
        sigma_s = sigma_s.reshape(-1,1)
        s_t = s_t.reshape(-1,1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)

    # 2D
    if isinstance(sigma_eta,(list,np.ndarray)):
        sigma_eta = sigma_eta.reshape(-len(sigma_eta),1)
        eta_t = eta_t.reshape(-len(eta_t),1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)
        alpha_eta = alpha_eta.reshape(-1,1)

    if lognorm:
        nmin = 1e-2
        nmax = 1e10
        
        if isinstance(n_src,(list,np.ndarray)):
            # assumes r0 cons
            n_src = n_src.reshape(-1,1)
            N_src = N_src.reshape(-1,1)
            N_surface = N_surface.reshape(-1,1)
            ks = ks.reshape(-1,1)
            if isinstance(rs,(list,np.ndarray)):
                rs = rs.reshape(-1,1) #.flatten()
            n_array = np.logspace(np.log10(nmin),np.log10(nmax),size)
            r = rs*(n_array/n_src)**(1/-ks)
            n_array = (n_src)*(r/rs)**(-ks)
            N_array = (N_src)*(r/rs)**(-ks+1)
            s_array = np.log(n_array/(n_src))
            eta = np.log(N_array/N_src)
            n_src = n_src.flatten()
            N_src = N_src.flatten()
            N_surface = N_surface.flatten()
            ks = ks.flatten()
        else:
            n_array = np.logspace(np.log10(nmin),np.log10(nmax),size)
            r = rs*(n_array/n_src)**(-1/ks)
            n_array = n_src*(r/rs)**(-ks)
            N_array = N_src*(r/rs)**(-ks+1)
            s_array = np.log(n_array/n_src)
            eta = np.log(N_array/N_src)

    
    if lognorm == False:
        nmin = 1e-1
        nmax = 1e10
        
        if isinstance(n_src,(list,np.ndarray)):
            print('TEST')
            n_surface = n_surface.flatten()
            #ks = ks.flatten()
            if isinstance(kappa,(np.ndarray,list)):
                kappa = kappa.flatten()
            r = np.logspace(6,-6,1000).reshape(-1,1)
            #n0_ = n_t*(r/r_t)**-kappa
            #n1_ = n_surface*(r/rs)**-ks
            #n_arr = 1/(1/n0_ + 1/n1_)
            n_arr = n_surface*(r/rs)**-kappa
            i0 = np.argmin(abs(1-n_arr/nmin),axis=0)
            i1 = np.argmin(abs(1-n_arr/nmax),axis=0)
            rmax = r[i0].flatten()
            rmin = r[i1].flatten()
            r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
            nnorm = nnorm.reshape(-1,1)
            rnorm = rnorm.reshape(-1,1)
            if isinstance(kappa,(np.ndarray,list)):
                kappa = kappa.reshape(-1,1)
            #ks = ks.reshape(-1,1)
            #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
            n_array = n_surface*(r/rs)**-kappa
            imax = np.argmax(n_array,axis=1)
            imin = np.argmin(n_array,axis=1)
            rmax = r[np.arange(len(imin)),imin]
            rmin = r[np.arange(len(imax)),imax]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
            while np.any(abs(np.log10(np.nanmax(n_array,axis=1)/nmax)) > 0.1) or np.any(abs(np.log10(np.nanmin(n_array,axis=1)/nmin)) > 0.1):
                #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
                n_array = n_surface*(r/rs)**-kappa
                imax = np.argmax(n_array,axis=1)
                imin = np.argmin(n_array,axis=1)
                cmax = np.nanmax(n_array,axis=1)/nmax
                cmin = np.nanmin(n_array,axis=1)/nmin
                rmax = r[np.arange(len(imin)),imin]
                rmin = r[np.arange(len(imax)),imax]
                i0 = abs(np.log10(cmax)) > 0.1
                i1 = abs(np.log10(cmin)) > 0.1
                rmax[i1] = (rmax*(cmin**0.33))[i1]
                rmin[i0] = (rmin*(cmax**0.33))[i0]
                r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
        
        else:
            r = np.logspace(6,-6,1000)
            #n0_ = n_t*(r/r_t)**-kappa
            #n1_ = n_src*(r/rs)**-ks
            #n_arr = 1/(1/n0_ + 1/n1_)
            n_arr = n_surface*(r/rs)**-kappa
            i0 = np.argmin(abs(1-n_arr/nmin))
            i1 = np.argmin(abs(1-n_arr/nmax))
            rmax = r[i0]
            rmin = r[i1]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size)
            #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(ks-kappa)
            #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
            n_array = n_surface*(r/rs)**-kappa
            imax = np.argmax(n_array) # removed, axis=1
            imin = np.argmin(n_array)
            rmax = r[imin] #np.arange(len(imin)),imin]
            rmin = r[imax] #np.arange(len(imax)),imax]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size)
            while np.any(abs(np.log10(np.nanmax(n_array)/nmax)) > 1.) or np.any(abs(np.log10(np.nanmin(n_array)/nmin)) > 1.):
                # removed axis=1
                #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(ks-kappa)
                #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
                n_array = n_surface*(r/rs)**-kappa
                imax = np.argmax(n_array) #,axis=1)
                imin = np.argmin(n_array) #,axis=1)
                cmax = np.nanmax(n_array)/nmax # removed axis=1
                cmin = np.nanmin(n_array)/nmin # removed axis=1
                #print(cmin,cmax)
                rmax = r[imin] #[np.arange(len(imin)),imin]
                rmin = r[imax] #[np.arange(len(imax)),imax]
                i0 = np.argmin(abs(1-n_array/nmin)) # 
                i1 =  np.argmin(abs(1-n_array/nmax)) # abs(np.log10(cmin)) > 1.
                rmax = (rmax*(cmin**0.33)) # [i1]
                rmin = (rmin*(cmax**0.33)) # [i0]
                r = np.logspace(np.log10(rmax),np.log10(rmin),size)

        if isinstance(n_src,(list,np.ndarray)):
            Nnorm = Nnorm.reshape(-1,1)
            n_src = n_src.reshape(-1,1)
            N_src = N_src.reshape(-1,1)
            N_surface = N_surface.reshape(-1,1)
        
        if isinstance(n_t,(np.ndarray,list)):
            n_t = np.vstack(n_t)
            N_t = np.vstack(N_t)
            r_t = np.vstack(r_t)
            n_surface = np.vstack(n_surface)
            N_surface = np.vstack(N_surface)
            n_src = np.vstack(n_src)
            N_src = np.vstack(N_src)
            #ks = np.vstack(ks)
           
        if isinstance(rs,(np.ndarray,list)):
            rs = np.vstack(rs)

        if isinstance(kappa,(np.ndarray,list)):
            kappa = np.vstack(kappa)

        if isinstance(n_src,(np.ndarray,list)):
            r = np.tile(np.logspace(3,-3,size),len(n_src)).reshape(len(n_src),size)
        else:
            r = np.logspace(3,-3,size)

        n_array = n_surface*(r/rs)**-kappa
        s_array = np.log(n_array/n_src)

        N_array = N_surface*(r/rs)**-(kappa+1)
        eta = np.log(N_array/N_src)
           
    if isinstance(n_src,(list,np.ndarray)):
        sigma_s = sigma_s.reshape(-1,1)
        s_t = s_t.reshape(-1,1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)

    # 3D
    s_0 = mean_log_density(sigma_s)
    C = PL_amplitude(alpha_s,sigma_s)
    N = normalization(C,s_t,alpha_s,sigma_s)
    LN = lognormal(N,sigma_s,s_array,s_0)
    PL = power_law(N,C,alpha_s,s_array)
    if lognorm == False:
        PDF = combine_PL_LN(s_t,PL,LN,s_array)
    elif lognorm == True:
        PDF = LN
    
    if mass_conserving:
        s_s = conservation_shift(N,C,alpha_s,s_t,sigma_s)
        s_array = s_array+np.nan_to_num(s_s)
        
    # 2D
    if isinstance(sigma_eta,(list,np.ndarray)):
        sigma_eta = sigma_eta.reshape(-len(sigma_eta),1)
        eta_t = eta_t.reshape(-len(eta_t),1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_eta = alpha_eta.reshape(-1,1)

    # 2D
    eta_0 = -0.5*sigma_eta**2
    C_N = PL_amplitude(alpha_eta,sigma_eta)
    N_N = normalization(C_N,eta_t,alpha_eta,sigma_eta)
    LN_N = lognormal(N_N,sigma_eta,eta,eta_0)
    PL_N = power_law(N_N,C_N,alpha_eta,eta)
    PDF_N = combine_PL_LN(eta_t,PL_N,LN_N,eta)

    if mass_conserving:
        eta_s = conservation_shift(N_N,C_N,alpha_eta,eta_t,sigma_eta)
        eta = eta-np.nan_to_num(eta_s)

    if lognorm:
        s_fix = np.log(n_ref/n_src)
        if s_crit == 'KM05':
            s_crit = calc_scrit_KM05(mach, alpha_vir)
            if isinstance(s_crit,(np.ndarray,list)):
                fcrit = []
                #s_crit = s_crit.reshape(-len(s_crit),1)
                for st_, LN_, s_ in zip(s_crit,LN,s_array):
                    fcrit.append(mass_frac_above_crit(st_,LN_,s_))
                    #plt.plot(s_,LN_)
                fcrit = np.array(fcrit)
            else:
                fcrit = mass_frac_above_crit(s_crit,LN,s_array)
        elif s_crit == 'PN11':
            #theta = np.sqrt(1/(np.pi**2/15/0.067))
            s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
            if isinstance(s_crit,(np.ndarray,list)):
                fcrit = []
                #s_crit = s_crit.reshape(-len(s_crit),1)
                for st_, LN_, s_ in zip(s_crit,LN,s_array):
                    fcrit.append(mass_frac_above_crit(st_,LN_,s_))
                    #plt.plot(s_,LN_)
                fcrit = np.array(fcrit)
            else:
                fcrit = mass_frac_above_crit(s_crit,LN,s_array)
        if isinstance(s_fix,(np.ndarray,list)):
            ffix = []
            #s_fix = s_fix.reshape(-len(s_fix),1)
            for st_, LN_, s_ in zip(s_fix,LN,s_array):
                ffix.append(mass_frac_above_crit(st_,LN_,s_))
            ffix = np.array(ffix)
        else:
            ffix = mass_frac_above_crit(s_fix,LN,s_array)
        if s_const == True:
            s_crit = s_fix
            fcrit = ffix
                
            
    elif lognorm == False:
        #s_choice = s_crit
        s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
        s_fix = np.log(n_ref/n_src)
        fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)
        ffix = mass_frac_above_crit(s_fix,PDF,s_array)
        f_t = mass_frac_above_crit(s_t,PDF,s_array)
        if isinstance(s_crit,(np.ndarray,list)):
            fcrit = calc_fdense_Burk2019(sigma_s.reshape(-len(sigma_s),1),alpha_s)
            fcrit = np.array(fcrit)
        if isinstance(s_fix,(np.ndarray,list)):
            ffix = []
            s_fix = s_fix.reshape(-len(s_fix),1)
            for sf_,PDF_, s_ in zip(s_fix,PDF,s_array):
                ffix.append(mass_frac_above_crit(sf_,PDF_,s_))
            ffix = np.array(ffix)
        if isinstance(s_t,(np.ndarray,list)):
            f_t = []
            s_t = s_t.reshape(-len(s_t),1)
            for sf_,PDF_, s_ in zip(s_t,PDF,s_array):
                f_t.append(mass_frac_above_crit(sf_,PDF_,s_))
            f_t = np.array(f_t)
        else:
            fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)
            ffix = mass_frac_above_crit(s_fix,PDF,s_array)
            f_t = mass_frac_above_crit(s_t,PDF,s_array)
        if (s_const == True):
            fcrit = ffix

    ##############################
    # SAVING PDF(S) TO DATAFRAME #
    ##############################

        
    df_pdf = pd.DataFrame()
    if isinstance(n_src,(list,np.ndarray)):
        nmodel = np.arange(1,len(n_src.flatten())+1)
    elif isinstance(n_src,(list,np.ndarray)) == False:
        nmodel = 0.
    else:
        print('CANNOT PRODUCE MODEL INDEX')
        return


    # DENSITY ARRAYS
    df_pdf['s_array'] = s_array.flatten()
    df_pdf['eta'] = eta.flatten()
    df_pdf['log n_arr'] = np.log10(n_array.flatten())
    df_pdf['log N_arr'] = np.log10(N_array.flatten())
    # RADII
    df_pdf['r_arr_pc'] = r.flatten()
    

    # 3D PDF MODEL ARRAYS
    df_pdf['LN'] = LN.flatten()
    df_pdf['PL'] = PL.flatten()
    df_pdf['PDF'] = PDF.flatten()
    # 2D PDF MODEL ARRAYS
    df_pdf['LN_eta'] = LN_N.flatten()
    df_pdf['PL_eta'] = PL_N.flatten()
    df_pdf['PDF_eta'] = PDF_N.flatten()

    # SLOPES
    if isinstance(alpha_s,(list,np.ndarray)) == False:
        df_pdf['kappa'] = kappa
        df_pdf['alpha_s'] = alpha_s
        df_pdf['alpha_N'] = alpha_N
        df_pdf['alpha_n'] = alpha_n
        df_pdf['alpha_eta'] = alpha_eta
    elif isinstance(alpha_s,(list,np.ndarray)):
        df_pdf['kappa'] = np.repeat(kappa,size)
        df_pdf['alpha_s'] = np.repeat(alpha_s.flatten(),size)
        df_pdf['alpha_N'] = np.repeat(alpha_N.flatten(),size)
        df_pdf['alpha_n'] = np.repeat(alpha_n.flatten(),size)
        df_pdf['alpha_eta'] = np.repeat(alpha_eta.flatten(),size)
        df_pdf['model'] = np.repeat(nmodel,size)
    else:
        print('ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return
        
    # PDF INPUTS AND NORMALIZATION CONSTANTS
    if isinstance(sigma_s,(list,np.ndarray)) == False:
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = sigma_v
        df_pdf['mach'] = mach
        df_pdf['N_surface'] = N_surface
        df_pdf['N_src'] = N_src
        df_pdf['n_src'] = n_src
        df_pdf['Mc_Msun'] = Mc
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
        if lognorm == False:
            df_pdf['f_t'] = f_t
            # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
            # EVEN CALCULATED FOR LN-ONLY MODELS
            df_pdf['eta_t'] = eta_t
            df_pdf['s_t'] = s_t
            df_pdf['N_t cm-2'] = N_t
            df_pdf['log n_t'] = np.log10(n_t)
            df_pdf['r_t pc'] = r_t
        # SPATIAL SLOPE OF OUTER PL
        #df_pdf['ks'] = ks
        # DENSITY CORRECTION
        df_pdf['fcorr'] = fcorr
        # CRITICAL DENSITY
        df_pdf['s_crit'] = s_crit
        # NORMALIZATION DENSITY & CORRESPONDING VALUES
        df_pdf['r_norm_pc'] = rnorm
        df_pdf['N_norm cm-2'] = Nnorm
        df_pdf['n_norm cm-3'] = nnorm
    elif isinstance(sigma_s,(list,np.ndarray)):
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = np.repeat(sigma_v.flatten(),size)
        df_pdf['mach'] = np.repeat(mach.flatten(),size)
        df_pdf['N_src'] = np.repeat(N_src.flatten(),size)
        df_pdf['n_src'] = np.repeat(n_src.flatten(),size)
        df_pdf['Mc_Msun'] = np.repeat(Mc.flatten(),size)
        # 3D PDF
        df_pdf['sigma_s'] = np.repeat(sigma_s.flatten(),size)
        df_pdf['sigma_n'] = np.repeat(sigma_n.flatten(),size)
        df_pdf['s_0'] = np.repeat(s_0.flatten(),size)
        df_pdf['C norm'] = np.repeat(C.flatten(),size)
        df_pdf['N norm'] = np.repeat(N.flatten(),size)
        # 2D PDF
        df_pdf['sigma_eta'] = np.repeat(sigma_eta.flatten(),size)
        df_pdf['sigma_N'] = np.repeat(sigma_N.flatten(),size)
        df_pdf['eta_0'] = np.repeat(eta_0.flatten(),size)
        df_pdf['N_N norm'] = np.repeat(C_N.flatten(),size)
        df_pdf['N_N norm'] = np.repeat(N_N.flatten(),size)
        # FRACTIONS
        df_pdf['fcrit'] = np.repeat(fcrit.flatten(),size)
        df_pdf['f_above_fix'] = np.repeat(ffix.flatten(),size)
        # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
        # EVEN CALCULATED FOR LN-ONLY MODELS
        df_pdf['eta_t'] = np.repeat(eta_t.flatten(),size)
        df_pdf['s_t'] = np.repeat(s_t.flatten(),size)
        df_pdf['N_t cm-2'] = np.repeat(N_t.flatten(),size)
        df_pdf['log n_t'] = np.log10(np.repeat(n_t.flatten(),size))
        df_pdf['r_t pc'] = np.repeat(r_t.flatten(),size)
        # SPATIAL SLOPE OF OUTER PL
        #df_pdf['ks'] = np.repeat(ks.flatten(),size)
        # DENSITY CORRECTION
        df_pdf['fcorr'] = np.repeat(fcorr.flatten(),size)
        # CRITICAL DENSITY
        df_pdf['s_crit'] = np.repeat(s_crit.flatten(),size)
        # NORMALIZATION DENSITY & CORRESPONDING VALUES
        df_pdf['r_norm_pc'] = np.repeat(rnorm.flatten(),size)
        df_pdf['N_norm cm-2'] = np.repeat(Nnorm.flatten(),size)
        df_pdf['n_norm cm-3'] = np.repeat(nnorm.flatten(),size)
    else:
        print('ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # WE CAN SET VIRIAL PARAMETER MANUALLY
    # VIRIAL PARAMETER
    if isinstance(alpha_vir,(list,np.ndarray)) == False:
        df_pdf['alpha_vir'] = alpha_vir
    elif isinstance(alpha_vir,(list,np.ndarray)):
        df_pdf['alpha_vir'] = np.repeat(alpha_vir.flatten(),size)
    else:
        print('VIRIAL PARAMETER ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # SOURCE SIZE CAN BE A SINGLE VALUE EVEN IF THERE ARE MUTIPLE MODELS
    if isinstance(rs,(list,np.ndarray)) == False:
        df_pdf['rs_pc'] = rs
    elif isinstance(rs,(list,np.ndarray)):
        df_pdf['rs_pc'] = np.repeat(rs.flatten(),size)
    else:
        print('SOURCE SIZE ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # TEMP CAN BE A SINGLE VALUE EVEN IF THERE ARE MUTIPLE MODELS
    if isinstance(temp,(list,np.ndarray)) == False:
        df_pdf['temp'] = temp
        df_pdf['c_s'] = c_s
    elif isinstance(temp,(list,np.ndarray)):
        df_pdf['temp'] = np.repeat(temp.flatten(),size)
        df_pdf['c_s'] = np.repeat(c_s.flatten(),size)
    else:
        print('TEMP ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return
                
    return df_pdf


# Added by Ashley to match cloud model in Sharda+2022
def cloud_model(userSigma, userveldisp, userR, useravir):
    # userSigma is in g / cm^2
    # userR is in pc
    # userveldisp is in cm / sec
    # const G is in cgs
    # userR is in cm
    G = 6.67384 * 1e-8
    P = (3 / 20) * np.pi * useravir * G * userSigma**2
    rho = P / (userveldisp)**2 # edge density
    kappa = 3 - 4 * userR * rho / userSigma
    #M = np.pi * userSigma * userR**2
    return P, rho, kappa

def get_PDFs4(sigma_v,
              temp,
              Sigma,
              rs,
              kappa=1.5,
              size=1000,
              b=0.4,
              beta=np.inf,
              n_ref = 10**4.5,
              mass_conserving=False,
              lognorm=False,
              s_crit=False,
              s_const=False,
              alpha_vir = 1.,
              correct_dens=False,
              user_nsrc = -1,
              r_ref = 1.):


    N_surface = Sigma*u.M_sun/u.pc**2/(2.33*(const.m_p+const.m_e))
    N_surface = N_surface.to(u.cm**-2).value
    
    sigma_v = np.sqrt(3)*sigma_v
    c_s = sound_speed(temp)
    mach = sigma_v/c_s
    alpha_s, alpha_N, alpha_n, alpha_eta = calc_slopes(kappa)

    if isinstance(beta,np.ndarray):
        sigma_s = LN_width_alfven(b, mach, beta)
    else:
        if np.isfinite(beta):
            sigma_s = LN_width_alfven(b, mach, beta)
        else:
            sigma_s = LN_width(b, mach)

    # 3D + 2D - LOG + LINEAR STD
    sigma_n = np.sqrt(np.exp(sigma_s**2)-1)
    A = 0.11
    sigma_eta = np.sqrt(A)*sigma_s
    sigma_N = np.sqrt(np.exp(sigma_eta**2)-1)

    # MASS OF SPHERE WITH CONSTANT DENSITY FROM SURFACE DENSITY, SIZE
    Mc_const = 4*np.pi*Sigma*u.M_sun/u.pc**2*(rs*u.pc)**2
    Mc_const = Mc_const.to(u.M_sun)
    Mc_const = Mc_const.value
    #rho_surface = Mc_const/(4/3*np.pi)/(rs*u.pc)**3
    #rho_surface = rho_surface.to(u.kg/u.cm**3).value
    #n_surface = rho_surface/(2.33*(const.m_p+const.m_e)).value
    #

    kmpers = 1e5 # convert km/s to cm/s
    gcm2 = 0.000208836 # convert Msun / pc^2 to g / cm^2
    pc = 3.086*1e18 # convert pc to cm
    P_surface, rho_surface, kappa = cloud_model(Sigma * gcm2, sigma_v * kmpers, rs * pc, alpha_vir)
    n_surface = rho_surface/(2.33*(const.m_p+const.m_e)).value

    #if correct_dens == True:
    #    # MASS OF SPHERE WITH SLOPE
    #    Mc = 4*np.pi/(3-kappa)*(n_surface*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
    #    Mc = Mc.to(u.M_sun).value
    #else:
    #    n_src = n_surface
    #    N_src = N_surface
    #    Mc = Mc_const

    if correct_dens == True:
        # MEAN DENSITY FROM SURFACE DENSITY AND WIDTH OF PDFS
        n_src = n_surface*sigma_n**2/2
        N_src = N_surface*sigma_N**2/2
        #n_src = n_surface*np.exp(sigma_s)**2/2
        #N_src = N_surface*np.exp(sigma_eta)**2/2
        
        # MASS OF SPHERE WITH CONSTANT DENSITY FROM MEAN DENSITY
        Mc = 4/3*np.pi*(n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
        Mc = Mc.to(u.M_sun).value
    else:
        n_src = n_surface
        N_src = N_surface
        Mc = Mc_const

    if isinstance(user_nsrc,(list,np.ndarray)):
        n_src = user_nsrc
        # MASS OF SPHERE WITH CONSTANT DENSITY FROM MEAN DENSITY
        Mc = 4/3*np.pi*(n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
        Mc = Mc.to(u.M_sun).value
    elif isinstance(user_nsrc,int):
        if (user_nsrc != -1):
            n_src = user_nsrc
            # MASS OF SPHERE WITH CONSTANT DENSITY FROM MEAN DENSITY
            Mc = 4/3*np.pi*(n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
            Mc = Mc.to(u.M_sun).value
        
    fcorr = Mc_const/Mc
    #print(fcorr)
    n_src = n_surface/fcorr
    N_src = N_surface/fcorr
    rho_src = n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3
    r0 = rs*(n_src/n_surface)
    

    # TRANSITION DENSITY
    eta_t = 0.5*(2*abs(alpha_eta)-1)*sigma_eta**2
    N_t = N_src*np.exp(eta_t)
    s_t = transition_density(alpha_s,sigma_s)
    n_t = n_src*np.exp(s_t)
    r_t = (N_t/n_t)*u.cm.to(u.pc)

    rnorm = rs
    nnorm = n_surface
    Nnorm = N_surface

    # 3D
    if isinstance(n_src,(list,np.ndarray)):
        sigma_s = sigma_s.reshape(-1,1)
        s_t = s_t.reshape(-1,1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)

    # 2D
    if isinstance(sigma_eta,(list,np.ndarray)):
        sigma_eta = sigma_eta.reshape(-len(sigma_eta),1)
        eta_t = eta_t.reshape(-len(eta_t),1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)
        alpha_eta = alpha_eta.reshape(-1,1)

    if lognorm:
        nmin = 1e-2
        nmax = 1e10
        
        if isinstance(n_src,(list,np.ndarray)):
            # assumes r0 cons
            n_src = n_src.reshape(-1,1)
            N_src = N_src.reshape(-1,1)
            N_surface = N_surface.reshape(-1,1)
            ks = ks.reshape(-1,1)
            if isinstance(rs,(list,np.ndarray)):
                rs = rs.reshape(-1,1) #.flatten()
            n_array = np.logspace(np.log10(nmin),np.log10(nmax),size)
            r = rs*(n_array/n_src)**(1/-ks)
            n_array = (n_src)*(r/rs)**(-ks)
            N_array = (N_src)*(r/rs)**(-ks+1)
            s_array = np.log(n_array/(n_src))
            eta = np.log(N_array/N_src)
            n_src = n_src.flatten()
            N_src = N_src.flatten()
            N_surface = N_surface.flatten()
            ks = ks.flatten()
        else:
            n_array = np.logspace(np.log10(nmin),np.log10(nmax),size)
            r = rs*(n_array/n_src)**(-1/ks)
            n_array = n_src*(r/rs)**(-ks)
            N_array = N_src*(r/rs)**(-ks+1)
            s_array = np.log(n_array/n_src)
            eta = np.log(N_array/N_src)

    
    if lognorm == False:
        nmin = 1e-1
        nmax = 1e10
        
        if isinstance(n_src,(list,np.ndarray)):
            print('TEST')
            n_surface = n_surface.flatten()
            #ks = ks.flatten()
            if isinstance(kappa,(np.ndarray,list)):
                kappa = kappa.flatten()
            r = np.logspace(6,-6,1000).reshape(-1,1)
            #n0_ = n_t*(r/r_t)**-kappa
            #n1_ = n_surface*(r/rs)**-ks
            #n_arr = 1/(1/n0_ + 1/n1_)
            n_arr = n_surface*(r/rs)**-kappa
            i0 = np.argmin(abs(1-n_arr/nmin),axis=0)
            i1 = np.argmin(abs(1-n_arr/nmax),axis=0)
            rmax = r[i0].flatten()
            rmin = r[i1].flatten()
            r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
            nnorm = nnorm.reshape(-1,1)
            rnorm = rnorm.reshape(-1,1)
            if isinstance(kappa,(np.ndarray,list)):
                kappa = kappa.reshape(-1,1)
            #ks = ks.reshape(-1,1)
            #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
            n_array = n_surface*(r/rs)**-kappa
            imax = np.argmax(n_array,axis=1)
            imin = np.argmin(n_array,axis=1)
            rmax = r[np.arange(len(imin)),imin]
            rmin = r[np.arange(len(imax)),imax]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
            while np.any(abs(np.log10(np.nanmax(n_array,axis=1)/nmax)) > 0.1) or np.any(abs(np.log10(np.nanmin(n_array,axis=1)/nmin)) > 0.1):
                #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
                n_array = n_surface*(r/rs)**-kappa
                imax = np.argmax(n_array,axis=1)
                imin = np.argmin(n_array,axis=1)
                cmax = np.nanmax(n_array,axis=1)/nmax
                cmin = np.nanmin(n_array,axis=1)/nmin
                rmax = r[np.arange(len(imin)),imin]
                rmin = r[np.arange(len(imax)),imax]
                i0 = abs(np.log10(cmax)) > 0.1
                i1 = abs(np.log10(cmin)) > 0.1
                rmax[i1] = (rmax*(cmin**0.33))[i1]
                rmin[i0] = (rmin*(cmax**0.33))[i0]
                r = np.logspace(np.log10(rmax),np.log10(rmin),size).T
        
        else:
            r = np.logspace(6,-6,1000)
            #n0_ = n_t*(r/r_t)**-kappa
            #n1_ = n_src*(r/rs)**-ks
            #n_arr = 1/(1/n0_ + 1/n1_)
            n_arr = n_surface*(r/rs)**-kappa
            i0 = np.argmin(abs(1-n_arr/nmin))
            i1 = np.argmin(abs(1-n_arr/nmax))
            rmax = r[i0]
            rmin = r[i1]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size)
            #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(ks-kappa)
            #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
            n_array = n_surface*(r/rs)**-kappa
            imax = np.argmax(n_array) # removed, axis=1
            imin = np.argmin(n_array)
            rmax = r[imin] #np.arange(len(imin)),imin]
            rmin = r[imax] #np.arange(len(imax)),imax]
            r = np.logspace(np.log10(rmax),np.log10(rmin),size)
            while np.any(abs(np.log10(np.nanmax(n_array)/nmax)) > 1.) or np.any(abs(np.log10(np.nanmin(n_array)/nmin)) > 1.):
                # removed axis=1
                #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(ks-kappa)
                #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
                n_array = n_surface*(r/rs)**-kappa
                imax = np.argmax(n_array) #,axis=1)
                imin = np.argmin(n_array) #,axis=1)
                cmax = np.nanmax(n_array)/nmax # removed axis=1
                cmin = np.nanmin(n_array)/nmin # removed axis=1
                #print(cmin,cmax)
                rmax = r[imin] #[np.arange(len(imin)),imin]
                rmin = r[imax] #[np.arange(len(imax)),imax]
                i0 = np.argmin(abs(1-n_array/nmin)) # 
                i1 =  np.argmin(abs(1-n_array/nmax)) # abs(np.log10(cmin)) > 1.
                rmax = (rmax*(cmin**0.33)) # [i1]
                rmin = (rmin*(cmax**0.33)) # [i0]
                r = np.logspace(np.log10(rmax),np.log10(rmin),size)

        if isinstance(n_src,(list,np.ndarray)):
            Nnorm = Nnorm.reshape(-1,1)
            n_src = n_src.reshape(-1,1)
            N_src = N_src.reshape(-1,1)
            N_surface = N_surface.reshape(-1,1)
        
        if isinstance(n_t,(np.ndarray,list)):
            n_t = np.vstack(n_t)
            N_t = np.vstack(N_t)
            r_t = np.vstack(r_t)
            n_surface = np.vstack(n_surface)
            N_surface = np.vstack(N_surface)
            n_src = np.vstack(n_src)
            N_src = np.vstack(N_src)
            #ks = np.vstack(ks)
           
        if isinstance(rs,(np.ndarray,list)):
            rs = np.vstack(rs)

        if isinstance(kappa,(np.ndarray,list)):
            kappa = np.vstack(kappa)

        if isinstance(n_src,(np.ndarray,list)):
            r = np.tile(np.logspace(3,-3,size),len(n_src)).reshape(len(n_src),size)
        else:
            r = np.logspace(3,-3,size)

        n_array = n_surface*(r/rs)**-kappa
        s_array = np.log(n_array/n_src)

        N_array = N_surface*(r/rs)**-(kappa+1)
        eta = np.log(N_array/N_src)

        #n_array = n_src*(r/r0)**-kappa
        #s_array = np.log(n_array/n_src)

        #N_array = N_src*(r/r0)**-(kappa+1)
        #eta = np.log(N_array/N_src)
           
    if isinstance(n_src,(list,np.ndarray)):
        sigma_s = sigma_s.reshape(-1,1)
        s_t = s_t.reshape(-1,1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)

    # 3D
    s_0 = mean_log_density(sigma_s)
    C = PL_amplitude(alpha_s,sigma_s)
    N = normalization(C,s_t,alpha_s,sigma_s)
    LN = lognormal(N,sigma_s,s_array,s_0)
    PL = power_law(N,C,alpha_s,s_array)
    if lognorm == False:
        PDF = combine_PL_LN(s_t,PL,LN,s_array)
    elif lognorm == True:
        PDF = LN
    
    if mass_conserving:
        s_s = conservation_shift(N,C,alpha_s,s_t,sigma_s)
        s_array = s_array+np.nan_to_num(s_s)
        
    # 2D
    if isinstance(sigma_eta,(list,np.ndarray)):
        sigma_eta = sigma_eta.reshape(-len(sigma_eta),1)
        eta_t = eta_t.reshape(-len(eta_t),1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_eta = alpha_eta.reshape(-1,1)

    # 2D
    eta_0 = -0.5*sigma_eta**2
    C_N = PL_amplitude(alpha_eta,sigma_eta)
    N_N = normalization(C_N,eta_t,alpha_eta,sigma_eta)
    LN_N = lognormal(N_N,sigma_eta,eta,eta_0)
    PL_N = power_law(N_N,C_N,alpha_eta,eta)
    PDF_N = combine_PL_LN(eta_t,PL_N,LN_N,eta)

    if mass_conserving:
        eta_s = conservation_shift(N_N,C_N,alpha_eta,eta_t,sigma_eta)
        eta = eta-np.nan_to_num(eta_s)

    if lognorm:
        s_fix = np.log(n_ref/n_src)
        if s_crit == 'KM05':
            s_crit = calc_scrit_KM05(mach, alpha_vir)
            if isinstance(s_crit,(np.ndarray,list)):
                fcrit = []
                #s_crit = s_crit.reshape(-len(s_crit),1)
                for st_, LN_, s_ in zip(s_crit,LN,s_array):
                    fcrit.append(mass_frac_above_crit(st_,LN_,s_))
                    #plt.plot(s_,LN_)
                fcrit = np.array(fcrit)
            else:
                fcrit = mass_frac_above_crit(s_crit,LN,s_array)
        elif s_crit == 'PN11':
            #theta = np.sqrt(1/(np.pi**2/15/0.067))
            s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
            if isinstance(s_crit,(np.ndarray,list)):
                fcrit = []
                #s_crit = s_crit.reshape(-len(s_crit),1)
                for st_, LN_, s_ in zip(s_crit,LN,s_array):
                    fcrit.append(mass_frac_above_crit(st_,LN_,s_))
                    #plt.plot(s_,LN_)
                fcrit = np.array(fcrit)
            else:
                fcrit = mass_frac_above_crit(s_crit,LN,s_array)
        if isinstance(s_fix,(np.ndarray,list)):
            ffix = []
            #s_fix = s_fix.reshape(-len(s_fix),1)
            for st_, LN_, s_ in zip(s_fix,LN,s_array):
                ffix.append(mass_frac_above_crit(st_,LN_,s_))
            ffix = np.array(ffix)
        else:
            ffix = mass_frac_above_crit(s_fix,LN,s_array)
        if s_const == True:
            s_crit = s_fix
            fcrit = ffix
                
            
    elif lognorm == False:
        #s_choice = s_crit
        s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
        s_fix = np.log(n_ref/n_src)
        fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)
        ffix = mass_frac_above_crit(s_fix,PDF,s_array)
        f_t = mass_frac_above_crit(s_t,PDF,s_array)
        if isinstance(s_crit,(np.ndarray,list)):
            fcrit = calc_fdense_Burk2019(sigma_s.reshape(-len(sigma_s),1),alpha_s)
            fcrit = np.array(fcrit)
        if isinstance(s_fix,(np.ndarray,list)):
            ffix = []
            s_fix = s_fix.reshape(-len(s_fix),1)
            for sf_,PDF_, s_ in zip(s_fix,PDF,s_array):
                ffix.append(mass_frac_above_crit(sf_,PDF_,s_))
            ffix = np.array(ffix)
        if isinstance(s_t,(np.ndarray,list)):
            f_t = []
            s_t = s_t.reshape(-len(s_t),1)
            for sf_,PDF_, s_ in zip(s_t,PDF,s_array):
                f_t.append(mass_frac_above_crit(sf_,PDF_,s_))
            f_t = np.array(f_t)
        else:
            fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)
            ffix = mass_frac_above_crit(s_fix,PDF,s_array)
            f_t = mass_frac_above_crit(s_t,PDF,s_array)
        if (s_const == True):
            fcrit = ffix

    ##############################
    # SAVING PDF(S) TO DATAFRAME #
    ##############################

        
    df_pdf = pd.DataFrame()
    if isinstance(n_src,(list,np.ndarray)):
        nmodel = np.arange(1,len(n_src.flatten())+1)
    elif isinstance(n_src,(list,np.ndarray)) == False:
        nmodel = 0.
    else:
        print('CANNOT PRODUCE MODEL INDEX')
        return


    # DENSITY ARRAYS
    df_pdf['s_array'] = s_array.flatten()
    df_pdf['eta'] = eta.flatten()
    df_pdf['log n_arr'] = np.log10(n_array.flatten())
    df_pdf['log N_arr'] = np.log10(N_array.flatten())
    # RADII
    df_pdf['r_arr_pc'] = r.flatten()
    

    # 3D PDF MODEL ARRAYS
    df_pdf['LN'] = LN.flatten()
    df_pdf['PL'] = PL.flatten()
    df_pdf['PDF'] = PDF.flatten()
    # 2D PDF MODEL ARRAYS
    df_pdf['LN_eta'] = LN_N.flatten()
    df_pdf['PL_eta'] = PL_N.flatten()
    df_pdf['PDF_eta'] = PDF_N.flatten()

    # SLOPES
    if isinstance(alpha_s,(list,np.ndarray)) == False:
        df_pdf['kappa'] = kappa
        df_pdf['alpha_s'] = alpha_s
        df_pdf['alpha_N'] = alpha_N
        df_pdf['alpha_n'] = alpha_n
        df_pdf['alpha_eta'] = alpha_eta
    elif isinstance(alpha_s,(list,np.ndarray)):
        df_pdf['kappa'] = np.repeat(kappa,size)
        df_pdf['alpha_s'] = np.repeat(alpha_s.flatten(),size)
        df_pdf['alpha_N'] = np.repeat(alpha_N.flatten(),size)
        df_pdf['alpha_n'] = np.repeat(alpha_n.flatten(),size)
        df_pdf['alpha_eta'] = np.repeat(alpha_eta.flatten(),size)
        df_pdf['model'] = np.repeat(nmodel,size)
    else:
        print('ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return
        
    # PDF INPUTS AND NORMALIZATION CONSTANTS
    if isinstance(sigma_s,(list,np.ndarray)) == False:
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = sigma_v
        df_pdf['mach'] = mach
        df_pdf['N_surface'] = N_surface
        df_pdf['N_src'] = N_src
        df_pdf['n_src'] = n_src
        df_pdf['Mc_Msun'] = Mc
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
        if lognorm == False:
            df_pdf['f_t'] = f_t
            # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
            # EVEN CALCULATED FOR LN-ONLY MODELS
            df_pdf['eta_t'] = eta_t
            df_pdf['s_t'] = s_t
            df_pdf['N_t cm-2'] = N_t
            df_pdf['log n_t'] = np.log10(n_t)
            df_pdf['r_t pc'] = r_t
        # SPATIAL SLOPE OF OUTER PL
        #df_pdf['ks'] = ks
        # DENSITY CORRECTION
        df_pdf['fcorr'] = fcorr
        # CRITICAL DENSITY
        df_pdf['s_crit'] = s_crit
        # NORMALIZATION DENSITY & CORRESPONDING VALUES
        df_pdf['r_norm_pc'] = rnorm
        df_pdf['N_norm cm-2'] = Nnorm
        df_pdf['n_norm cm-3'] = nnorm
    elif isinstance(sigma_s,(list,np.ndarray)):
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = np.repeat(sigma_v.flatten(),size)
        df_pdf['mach'] = np.repeat(mach.flatten(),size)
        df_pdf['N_src'] = np.repeat(N_src.flatten(),size)
        df_pdf['n_src'] = np.repeat(n_src.flatten(),size)
        df_pdf['Mc_Msun'] = np.repeat(Mc.flatten(),size)
        # 3D PDF
        df_pdf['sigma_s'] = np.repeat(sigma_s.flatten(),size)
        df_pdf['sigma_n'] = np.repeat(sigma_n.flatten(),size)
        df_pdf['s_0'] = np.repeat(s_0.flatten(),size)
        df_pdf['C norm'] = np.repeat(C.flatten(),size)
        df_pdf['N norm'] = np.repeat(N.flatten(),size)
        # 2D PDF
        df_pdf['sigma_eta'] = np.repeat(sigma_eta.flatten(),size)
        df_pdf['sigma_N'] = np.repeat(sigma_N.flatten(),size)
        df_pdf['eta_0'] = np.repeat(eta_0.flatten(),size)
        df_pdf['N_N norm'] = np.repeat(C_N.flatten(),size)
        df_pdf['N_N norm'] = np.repeat(N_N.flatten(),size)
        # FRACTIONS
        df_pdf['fcrit'] = np.repeat(fcrit.flatten(),size)
        df_pdf['f_above_fix'] = np.repeat(ffix.flatten(),size)
        # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
        # EVEN CALCULATED FOR LN-ONLY MODELS
        df_pdf['eta_t'] = np.repeat(eta_t.flatten(),size)
        df_pdf['s_t'] = np.repeat(s_t.flatten(),size)
        df_pdf['N_t cm-2'] = np.repeat(N_t.flatten(),size)
        df_pdf['log n_t'] = np.log10(np.repeat(n_t.flatten(),size))
        df_pdf['r_t pc'] = np.repeat(r_t.flatten(),size)
        # SPATIAL SLOPE OF OUTER PL
        #df_pdf['ks'] = np.repeat(ks.flatten(),size)
        # DENSITY CORRECTION
        df_pdf['fcorr'] = np.repeat(fcorr.flatten(),size)
        # CRITICAL DENSITY
        df_pdf['s_crit'] = np.repeat(s_crit.flatten(),size)
        # NORMALIZATION DENSITY & CORRESPONDING VALUES
        df_pdf['r_norm_pc'] = np.repeat(rnorm.flatten(),size)
        df_pdf['N_norm cm-2'] = np.repeat(Nnorm.flatten(),size)
        df_pdf['n_norm cm-3'] = np.repeat(nnorm.flatten(),size)
    else:
        print('ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # WE CAN SET VIRIAL PARAMETER MANUALLY
    # VIRIAL PARAMETER
    if isinstance(alpha_vir,(list,np.ndarray)) == False:
        df_pdf['alpha_vir'] = alpha_vir
    elif isinstance(alpha_vir,(list,np.ndarray)):
        df_pdf['alpha_vir'] = np.repeat(alpha_vir.flatten(),size)
    else:
        print('VIRIAL PARAMETER ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # SOURCE SIZE CAN BE A SINGLE VALUE EVEN IF THERE ARE MUTIPLE MODELS
    if isinstance(rs,(list,np.ndarray)) == False:
        df_pdf['rs_pc'] = rs
    elif isinstance(rs,(list,np.ndarray)):
        df_pdf['rs_pc'] = np.repeat(rs.flatten(),size)
    else:
        print('SOURCE SIZE ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # TEMP CAN BE A SINGLE VALUE EVEN IF THERE ARE MUTIPLE MODELS
    if isinstance(temp,(list,np.ndarray)) == False:
        df_pdf['temp'] = temp
        df_pdf['c_s'] = c_s
    elif isinstance(temp,(list,np.ndarray)):
        df_pdf['temp'] = np.repeat(temp.flatten(),size)
        df_pdf['c_s'] = np.repeat(c_s.flatten(),size)
    else:
        print('TEMP ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return
                
    return df_pdf


def get_PDFs_simple(sigma_v,
                    temp,
                    Sigma,
                    rs,
                    kappa=1.5,
                    size=1000,
                    b=0.4,
                    beta=np.inf,
                    n_ref = 10**4.5,
                    mass_conserving=False,
                    lognorm=False,
                    s_crit=False,
                    s_const=False,
                    alpha_vir = 1.,
                    correct_dens=False,
                    r_ref = 1.):


    N_surface = Sigma*u.M_sun/u.pc**2/(2.33*(const.m_p+const.m_e))
    N_surface = N_surface.to(u.cm**-2).value
    
    sigma_v = np.sqrt(3)*sigma_v
    c_s = sound_speed(temp)
    mach = sigma_v/c_s
    alpha_s, alpha_N, alpha_n, alpha_eta = calc_slopes(kappa)

    if isinstance(beta,np.ndarray):
        sigma_s = LN_width_alfven(b, mach, beta)
    else:
        if np.isfinite(beta):
            sigma_s = LN_width_alfven(b, mach, beta)
        else:
            sigma_s = LN_width(b, mach)

    # 3D + 2D - LOG + LINEAR STD
    sigma_n = np.sqrt(np.exp(sigma_s**2)-1)
    A = 0.11
    sigma_eta = np.sqrt(A)*sigma_s
    sigma_N = np.sqrt(np.exp(sigma_eta**2)-1)
    ks = sigma_n/(sigma_N+sigma_n)
    #ks = np.repeat(1.,len(Sigma))

    # ESTIMATE MEAN DENSITY
    Mc_const = 4*np.pi*Sigma*u.M_sun/u.pc**2*(rs*u.pc)**2
    Mc_const = Mc_const.to(u.M_sun)
    rho = Mc_const/(4/3*np.pi)/(rs*u.pc)**3
    rho = rho.to(u.kg/u.cm**3).value
    n_surface = rho/(2.33*(const.m_p+const.m_e)).value
    #n_src = n_surface*np.exp(sigma_s**2/2)
    #N_src = N_surface*np.exp(sigma_eta**2/2)
    n_src = n_surface*sigma_n**2/2
    N_src = N_surface*sigma_N**2/2
    #n_src = n_surface
    #N_src = N_surface
    Mc = 4/3*np.pi*(n_src*(2.33*(const.m_p+const.m_e))*u.cm**-3)*(rs*u.pc)**3
    Mc = Mc.to(u.M_sun).value
    fcorr = Mc_const/Mc

    rho_surface = n_surface*(2.33*(const.m_p+const.m_e))*u.cm**-3
    rho_surface = rho_surface.to(u.kg/u.cm**3).value
            
    # TRANSITION
    eta_t = 0.5*(2*abs(alpha_eta)-1)*sigma_eta**2
    N_t = N_src*np.exp(eta_t)
    s_t = transition_density(alpha_s,sigma_s)
    n_t = n_src*np.exp(s_t)
    r_t = (N_t/n_t)*u.cm.to(u.pc)

    try:
        C = ((n_t/n_surface)*(r_t/rs)**(kappa/(ks-kappa)))
    except:
        C=0.
    if lognorm:
        C=0
    rnorm = abs(rs - C*r_t)/abs(C-1)
    nnorm = n_t*(r_t/rnorm)**-kappa*(1+r_t/rnorm)**(kappa-ks)
    #Nnorm = nnorm*u.cm**-3*rnorm*u.pc
    #Nnorm = Nnorm.to(u.cm**-2).value
    Nnorm = N_t*(r_t/rnorm)**-(kappa+1)*(1+r_t/rnorm)**(kappa-ks)

    #r = np.logspace(6,-6,size)
    #n_array = nnorm*(r/rnorm)**-kappa*(1+r/rnorm)**(kappa-ks)
    #N_array = Nnorm*(r/rnorm)**-(kappa+1)*(1+r/rnorm)**(kappa-ks)

    # 3D
    if isinstance(n_src,(list,np.ndarray)):
        sigma_s = sigma_s.reshape(-1,1)
        s_t = s_t.reshape(-1,1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)

    # 2D
    if isinstance(sigma_eta,(list,np.ndarray)):
        sigma_eta = sigma_eta.reshape(-len(sigma_eta),1)
        eta_t = eta_t.reshape(-len(eta_t),1)
    if isinstance(kappa,(list,np.ndarray)):
        alpha_s = alpha_s.reshape(-1,1)
        alpha_eta = alpha_eta.reshape(-1,1)

    # INIT DENSITY ARRAYS
    s_0 = mean_log_density(sigma_s)
    if isinstance(s_0,(list,np.ndarray)): 
        s_0 = s_0.reshape(-1,1)
    s_array = np.squeeze(np.linspace(s_0-sigma_s*10,s_0+sigma_s*10,size).T)
    eta_0 = -0.5*sigma_eta**2
    if isinstance(eta_0,(list,np.ndarray)):
        eta_0 = eta_0.reshape(-1,1)
    eta = np.squeeze(np.linspace(eta_0-sigma_eta*10,eta_0+sigma_eta*10,size).T)
    
    #s_array = np.log(n_array/n_src)
    #eta = np.log(N_array/N_src)
    
    # 3D
    #s_0 = mean_log_density(sigma_s)
    C = PL_amplitude(alpha_s,sigma_s)
    N = normalization(C,s_t,alpha_s,sigma_s)
    LN = lognormal(N,sigma_s,s_array,s_0)
    PL = power_law(N,C,alpha_s,s_array)
    if lognorm == False:
        PDF = combine_PL_LN(s_t,PL,LN,s_array)
    elif lognorm == True:
        PDF = LN

    i0 = np.squeeze((s_array > s_0) & (np.log10(PDF) < -5))
    i1 = np.squeeze((s_array < s_0) & (np.log10(PDF) < -5))
    if isinstance(s_0,(list,np.ndarray)):
        s_max = []
        eta_max = []
        for i,s,eta_ in zip(i0,s_array,eta):
            s_max.append(s[i][0])
            eta_max.append(eta_[i][0])
        s_max = np.array(s_max).reshape(-1,1)
        eta_max = np.array(eta_max).reshape(-1,1)
        
        s_min = []
        eta_min = []
        for i,s,eta_ in zip(i1,s_array,eta):
            s_min.append(s[i][-1])
            eta_min.append(eta_[i][-1])
        s_min = np.array(s_min).reshape(-1,1)
        eta_min = np.array(eta_min).reshape(-1,1)
    else:
        s_max = s_array[i0][0]
        s_min = s_array[i1][-1]
        eta_min = eta[i0][0]
        eta_max = eta[i1][-1]
    #print(s_max)

    # DENSITY ARRAYS
    #if isinstance(n_src,(list,np.ndarray)):
    s_array = np.squeeze(np.linspace(s_min.reshape(-1,1),s_max.reshape(-1,1),size).T)
    eta = np.squeeze(np.linspace(eta_min.reshape(-1,1),eta_max.reshape(-1,1),size).T)
    #else:
    #    s_array = np.squeeze(np.linspace(s_min,s_max,size))
    #    eta = np.squeeze(np.linspace(eta_min,eta_max,size))

    n_array = n_src*np.exp(s_array)
    N_array = N_src*np.exp(eta)
    r = N_array/n_array*u.cm.to(u.pc)
    #n_array = n_src*(r/rs)**-kappa
    #N_array = N_src*(r/rs)**-(kappa+1)
    #n_array = n_surface*(r/rs)**-kappa
    #N_array = N_surface*(r/rs)**-(kappa+1)
    
    #s_array = np.log(n_array/n_src)
    #eta = np.log(N_array/N_src)

    # 3D
    #s_0 = mean_log_density(sigma_s)
    C = PL_amplitude(alpha_s,sigma_s)
    N = normalization(C,s_t,alpha_s,sigma_s)
    LN = lognormal(N,sigma_s,s_array,s_0)
    PL = power_law(N,C,alpha_s,s_array)
    if lognorm == False:
        PDF = combine_PL_LN(s_t,PL,LN,s_array)
    elif lognorm == True:
        PDF = LN

    # 2D
    #eta_0 = -0.5*sigma_eta**2
    C_N = PL_amplitude(alpha_eta,sigma_eta)
    N_N = normalization(C_N,eta_t,alpha_eta,sigma_eta)
    LN_N = lognormal(N_N,sigma_eta,eta,eta_0)
    PL_N = power_law(N_N,C_N,alpha_eta,eta)
    if lognorm == False:
        PDF_N = combine_PL_LN(eta_t,PL_N,LN_N,eta)
    elif lognorm == True:
        PDF_N = LN_N

    if mass_conserving:
        s_s = conservation_shift(N,C,alpha_s,s_t,sigma_s)
        s_array = s_array+np.nan_to_num(s_s)
        eta_s = conservation_shift(N_N,C_N,alpha_eta,eta_t,sigma_eta)
        eta = eta-np.nan_to_num(eta_s)

    if lognorm:
        s_fix = np.log(n_ref/n_src)
        if s_crit == 'KM05':
            s_crit = calc_scrit_KM05(mach, alpha_vir)
            if isinstance(s_crit,(np.ndarray,list)):
                fcrit = []
                #s_crit = s_crit.reshape(-len(s_crit),1)
                for st_, LN_, s_ in zip(s_crit,LN,s_array):
                    fcrit.append(mass_frac_above_crit(st_,LN_,s_))
                    #plt.plot(s_,LN_)
                fcrit = np.array(fcrit)
            else:
                fcrit = mass_frac_above_crit(s_crit,LN,s_array)
        elif s_crit == 'PN11':
            #theta = np.sqrt(1/(np.pi**2/15/0.067))
            s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
            if isinstance(s_crit,(np.ndarray,list)):
                fcrit = []
                #s_crit = s_crit.reshape(-len(s_crit),1)
                for st_, LN_, s_ in zip(s_crit,LN,s_array):
                    fcrit.append(mass_frac_above_crit(st_,LN_,s_))
                    #plt.plot(s_,LN_)
                fcrit = np.array(fcrit)
            else:
                fcrit = mass_frac_above_crit(s_crit,LN,s_array)
        if isinstance(s_fix,(np.ndarray,list)):
            ffix = []
            #s_fix = s_fix.reshape(-len(s_fix),1)
            for st_, LN_, s_ in zip(s_fix,LN,s_array):
                ffix.append(mass_frac_above_crit(st_,LN_,s_))
            ffix = np.array(ffix)
        else:
            ffix = mass_frac_above_crit(s_fix,LN,s_array)
        if s_const == True:
            s_crit = s_fix
            fcrit = ffix
                
            
    elif lognorm == False:
        #s_choice = s_crit
        s_crit = calc_scrit_PN11(mach,np.inf,alpha_vir)
        s_fix = np.log(n_ref/n_src)
        #fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)
        #ffix = mass_frac_above_crit(s_fix,PDF,s_array)
        #f_t = mass_frac_above_crit(s_t,PDF,s_array)
        if isinstance(s_crit,(np.ndarray,list)):
            fcrit = calc_fdense_Burk2019(sigma_s.reshape(-len(sigma_s),1),alpha_s)
            fcrit = np.array(fcrit)
        if isinstance(s_fix,(np.ndarray,list)):
            ffix = []
            s_fix = s_fix.reshape(-len(s_fix),1)
            for sf_,PDF_, s_ in zip(s_fix,PDF,s_array):
                ffix.append(mass_frac_above_crit(sf_,PDF_,s_))
            ffix = np.array(ffix)
        if isinstance(s_t,(np.ndarray,list)):
            f_t = []
            s_t = s_t.reshape(-len(s_t),1)
            for sf_,PDF_, s_ in zip(s_t,PDF,s_array):
                f_t.append(mass_frac_above_crit(sf_,PDF_,s_))
            f_t = np.array(f_t)
        else:
            fcrit = calc_fdense_Burk2019(sigma_s,alpha_s)
            ffix = mass_frac_above_crit(s_fix,PDF,s_array)
            f_t = mass_frac_above_crit(s_t,PDF,s_array)
        if (s_const == True):
            fcrit = ffix

    ##############################
    # SAVING PDF(S) TO DATAFRAME #
    ##############################

    #plt.plot((s_array[100].T),np.log10(PDF[100].T))
    #plt.show()
        
    df_pdf = pd.DataFrame()
    if isinstance(n_src,(list,np.ndarray)):
        nmodel = np.arange(1,len(n_src.flatten())+1)
    elif isinstance(n_src,(list,np.ndarray)) == False:
        nmodel = 0.
    else:
        print('CANNOT PRODUCE MODEL INDEX')
        return


    # DENSITY ARRAYS
    df_pdf['s_array'] = s_array.flatten()
    df_pdf['eta'] = eta.flatten()
    df_pdf['log n_arr'] = np.log10(n_array.flatten())
    df_pdf['log N_arr'] = np.log10(N_array.flatten())
    # RADII
    df_pdf['r_arr_pc'] = r.flatten()
    

    # 3D PDF MODEL ARRAYS
    df_pdf['LN'] = LN.flatten()
    df_pdf['PL'] = PL.flatten()
    df_pdf['PDF'] = PDF.flatten()
    # 2D PDF MODEL ARRAYS
    df_pdf['LN_eta'] = LN_N.flatten()
    df_pdf['PL_eta'] = PL_N.flatten()
    df_pdf['PDF_eta'] = PDF_N.flatten()

    # SLOPES
    if isinstance(alpha_s,(list,np.ndarray)) == False:
        df_pdf['kappa'] = kappa
        df_pdf['alpha_s'] = alpha_s
        df_pdf['alpha_N'] = alpha_N
        df_pdf['alpha_n'] = alpha_n
        df_pdf['alpha_eta'] = alpha_eta
    elif isinstance(alpha_s,(list,np.ndarray)):
        df_pdf['kappa'] = np.repeat(kappa,size)
        df_pdf['alpha_s'] = np.repeat(alpha_s.flatten(),size)
        df_pdf['alpha_N'] = np.repeat(alpha_N.flatten(),size)
        df_pdf['alpha_n'] = np.repeat(alpha_n.flatten(),size)
        df_pdf['alpha_eta'] = np.repeat(alpha_eta.flatten(),size)
        df_pdf['model'] = np.repeat(nmodel,size)
    else:
        print('ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return
        
    # PDF INPUTS AND NORMALIZATION CONSTANTS
    if isinstance(sigma_s,(list,np.ndarray)) == False:
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = sigma_v
        df_pdf['mach'] = mach
        df_pdf['N_surface'] = N_surface
        df_pdf['N_src'] = N_src
        df_pdf['n_src'] = n_src
        df_pdf['Mc_Msun'] = Mc
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
        if lognorm == False:
            df_pdf['f_t'] = f_t
            # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
            # EVEN CALCULATED FOR LN-ONLY MODELS
            df_pdf['eta_t'] = eta_t
            df_pdf['s_t'] = s_t
            df_pdf['N_t cm-2'] = N_t
            df_pdf['log n_t'] = np.log10(n_t)
            df_pdf['r_t pc'] = r_t
        # SPATIAL SLOPE OF OUTER PL
        df_pdf['ks'] = ks
        # DENSITY CORRECTION
        df_pdf['fcorr'] = fcorr
        # CRITICAL DENSITY
        df_pdf['s_crit'] = s_crit
        # NORMALIZATION DENSITY & CORRESPONDING VALUES
        #df_pdf['r_norm_pc'] = rnorm
        #df_pdf['N_norm cm-2'] = Nnorm
        #df_pdf['n_norm cm-3'] = nnorm
    elif isinstance(sigma_s,(list,np.ndarray)):
        # GENERAL INPUTS
        df_pdf['sigma_v_3D'] = np.repeat(sigma_v.flatten(),size)
        df_pdf['mach'] = np.repeat(mach.flatten(),size)
        df_pdf['N_src'] = np.repeat(N_src.flatten(),size)
        df_pdf['n_src'] = np.repeat(n_src.flatten(),size)
        df_pdf['Mc_Msun'] = np.repeat(Mc.flatten(),size)
        # 3D PDF
        df_pdf['sigma_s'] = np.repeat(sigma_s.flatten(),size)
        df_pdf['sigma_n'] = np.repeat(sigma_n.flatten(),size)
        df_pdf['s_0'] = np.repeat(s_0.flatten(),size)
        df_pdf['C norm'] = np.repeat(C.flatten(),size)
        df_pdf['N norm'] = np.repeat(N.flatten(),size)
        # 2D PDF
        df_pdf['sigma_eta'] = np.repeat(sigma_eta.flatten(),size)
        df_pdf['sigma_N'] = np.repeat(sigma_N.flatten(),size)
        df_pdf['eta_0'] = np.repeat(eta_0.flatten(),size)
        df_pdf['N_N norm'] = np.repeat(C_N.flatten(),size)
        df_pdf['N_N norm'] = np.repeat(N_N.flatten(),size)
        # FRACTIONS
        df_pdf['fcrit'] = np.repeat(fcrit.flatten(),size)
        df_pdf['f_above_fix'] = np.repeat(ffix.flatten(),size)
        # TRANSITION DENSITY & OTHER QUANTITIES AT TRANSITION DENSITY
        # EVEN CALCULATED FOR LN-ONLY MODELS
        df_pdf['eta_t'] = np.repeat(eta_t.flatten(),size)
        df_pdf['s_t'] = np.repeat(s_t.flatten(),size)
        df_pdf['N_t cm-2'] = np.repeat(N_t.flatten(),size)
        df_pdf['log n_t'] = np.log10(np.repeat(n_t.flatten(),size))
        df_pdf['r_t pc'] = np.repeat(r_t.flatten(),size)
        # SPATIAL SLOPE OF OUTER PL
        if isinstance(ks,(np.ndarray,list)):
            df_pdf['ks'] = np.repeat(ks.flatten(),size)
        else:
            df_pdf['ks'] = np.repeat(ks,size)
        # DENSITY CORRECTION
        df_pdf['fcorr'] = np.repeat(fcorr.flatten(),size)
        # CRITICAL DENSITY
        df_pdf['s_crit'] = np.repeat(s_crit.flatten(),size)
        # NORMALIZATION DENSITY & CORRESPONDING VALUES
        #df_pdf['r_norm_pc'] = np.repeat(rnorm.flatten(),size)
        #df_pdf['N_norm cm-2'] = np.repeat(Nnorm.flatten(),size)
        #df_pdf['n_norm cm-3'] = np.repeat(nnorm.flatten(),size)
    else:
        print('ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # WE CAN SET VIRIAL PARAMETER MANUALLY
    # VIRIAL PARAMETER
    if isinstance(alpha_vir,(list,np.ndarray)) == False:
        df_pdf['alpha_vir'] = alpha_vir
    elif isinstance(alpha_vir,(list,np.ndarray)):
        df_pdf['alpha_vir'] = np.repeat(alpha_vir.flatten(),size)
    else:
        print('VIRIAL PARAMETER ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # SOURCE SIZE CAN BE A SINGLE VALUE EVEN IF THERE ARE MUTIPLE MODELS
    if isinstance(rs,(list,np.ndarray)) == False:
        df_pdf['rs_pc'] = rs
    elif isinstance(rs,(list,np.ndarray)):
        df_pdf['rs_pc'] = np.repeat(rs.flatten(),size)
    else:
        print('SOURCE SIZE ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return

    # TEMP CAN BE A SINGLE VALUE EVEN IF THERE ARE MUTIPLE MODELS
    if isinstance(temp,(list,np.ndarray)) == False:
        df_pdf['temp'] = temp
        df_pdf['c_s'] = c_s
    elif isinstance(temp,(list,np.ndarray)):
        df_pdf['temp'] = np.repeat(temp.flatten(),size)
        df_pdf['c_s'] = np.repeat(c_s.flatten(),size)
    else:
        print('TEMP ARRAY LENGTH INCOMPATIBLE WITH DATAFRAME')
        return
                
    return df_pdf

def get_df_with_SF_props(sigma_v,temp,n_src,r_src,
                         Sigma,alpha=2.0,
                         size=500,b=0.4,beta=np.inf,vwind=10.,
                         mass_conserving=False,lognorm=False,
                         match_temp_to_SF=False):

    kappa = 3./alpha
    df_pdf = get_PDFs(sigma_v,temp,Sigma,n_src,r_src,
                      kappa=3./alpha,size=size,b=b,
                      mass_conserving=mass_conserving,
                      lognorm=lognorm)

    alpha = df_pdf['alpha_s'].values[0]
    kappa = df_pdf['kappa'].values[0]
    rs = df_pdf['rs_pc'].values[0]
    n_src = df_pdf['n_src'].values[0]
    N_src = df_pdf['N_src'].values[0]
    sigma_v = df_pdf['sigma_v_3D'].values[0]/np.sqrt(3)
    fdense = df_pdf['fcrit'].values[0]
    sigma_s = df_pdf['sigma_s'].values[0]
    avir = df_pdf['alpha_vir'].values[0]
    
    dv = np.sqrt(8*np.log(2))*sigma_v*u.km/u.s
    dvdr = dv*n_src*u.cm**-3/N_src/u.cm**-2
    dvdr = dvdr.to(u.km/u.s/u.pc).value
    t_grav = temp_grav(n_src,dvdr, kappa)
    t_turb = temp_turb(sigma_v, rs, n_src, dvdr)

    Sigma = N_src*u.cm**-2*(2.33*(const.m_p+const.m_e))
    Sigma = Sigma.to(u.M_sun/u.pc**2).value
    rho_src = n_src*(2.33*(const.m_p+const.m_e)).value
    tff = np.sqrt(3.*np.pi/(32*const.G))*np.sqrt(1/(rho_src*u.kg*u.cm**-3))
    tff = tff.to(u.yr)

    if match_temp_to_SF == True:
        dsf = 1
        dt = 1
        SFR_ff = 1e6
        niter=0
        while ((dsf >= 1e-6) and (dt >= 1e-6)) and (niter < 10):
        
            temp_last = temp
            Sigma_SFR_last = SFR_ff
            
            fdense = df_pdf['fcrit'].values[0]
            sigma_s = df_pdf['sigma_s'].values[0]
            s_crit = df_pdf['s_crit'].values[0]
            
            SFR_ff = calc_SFR_composite(sigma_s, alpha, s_crit, eff = fdense)
            Sigma_SFR = SFR_ff*(N_src*u.cm**-2*(2.33*(const.m_p+const.m_e))).to(u.M_sun/u.pc**2)/tff.to(u.Myr)
            Sigma_SFR = Sigma_SFR.value
            
            t_cr = temp_cr(Sigma_SFR,n_src,dvdr,vwind=vwind)
            temp = np.nanmax([t_turb,t_cr,t_grav])
            
            df_pdf = get_PDFs(sigma_v,temp,
                              n_src,Sigma,
                              kappa=kappa,size=size,
                              b=b,mass_conserving=mass_conserving,
                              lognorm=lognorm)
            SFR_ff = calc_SFR_composite(sigma_s, alpha, s_crit, eff = fdense)
            Sigma_SFR = SFR_ff*(N_src*u.cm**-2*(2.33*(const.m_p+const.m_e))).to(u.M_sun/u.pc**2)/tff.to(u.Myr)
            Sigma_SFR = Sigma_SFR.value
            
            t_cr = temp_cr(Sigma_SFR,n_src,dvdr,vwind=vwind)
            temp = np.nanmax([t_turb,t_cr,t_grav])
            
            dsf = abs((Sigma_SFR-Sigma_SFR_last))
            dt = abs((temp-temp_last))
            niter=niter+1
            
        df_pdf['temp'] = temp
        df_pdf['temp_cr'] = t_cr
        df_pdf['temp_turb'] = t_turb
        df_pdf['temp_grav'] = t_grav

    s_crit = df_pdf['s_crit'].values[0]
    SFR_ff = calc_SFR_composite(sigma_s, alpha, s_crit, eff = fdense)
    Sigma_SFR = SFR_ff*(N_src*u.cm**-2*(2.33*(const.m_p+const.m_e))).to(u.M_sun/u.pc**2)/tff.to(u.Myr)
    Sigma_SFR = Sigma_SFR.value
    
    df_pdf['SFR_ff'] = SFR_ff
    df_pdf['Sigma_SFR'] = Sigma_SFR
    df_pdf['tff_yr'] = tff.value

    return df_pdf



def calc_fcorr(sigma_v,temp,Sigma,rs,
               kappa=1.5,
               b=0.4,
               beta=np.inf):

    alpha = 3/kappa
    N_src = Sigma*u.M_sun/u.pc**2/(2.33*(const.m_p+const.m_e))
    N_src = N_src.to(u.cm**-2).value
    sigma_v = np.sqrt(3)*sigma_v
    c_s = sound_speed(temp)
    mach = sigma_v/c_s
    alpha_s, alpha_N, alpha_n, alpha_eta = calc_slopes(kappa)

    Mc_const = 4*np.pi*Sigma*u.M_sun/u.pc**2*(rs*u.pc)**2
    Mc_const = Mc_const.to(u.M_sun)
    rho_src = Mc_const/(4/3*np.pi)/(rs*u.pc)**3
    rho_src = rho_src.to(u.kg/u.cm**3).value
    n_src = rho_src/(2.33*(const.m_p+const.m_e)).value

    if isinstance(beta,np.ndarray):
        sigma_s = LN_width_alfven(b, mach, beta)
    else:
        if np.isfinite(beta):
            sigma_s = LN_width_alfven(b, mach, beta)
        else:
            sigma_s = LN_width(b, mach)

    # 3D + 2D - LOG + LINEAR STD
    sigma_n = np.sqrt(np.exp(sigma_s**2)-1)
    A = 0.11
    sigma_eta = np.sqrt(A)*sigma_s
    sigma_N = np.sqrt(np.exp(sigma_eta**2)-1)
    ks = 2.*sigma_n/(sigma_N+sigma_n)
    
    #ks = 2*np.log(n_up/n_lo)/(np.log(N_up/N_lo)+np.log(n_up/n_lo))

    # TRANSITION
    eta_t = 0.5*(2*abs(alpha_eta)-1)*sigma_eta**2
    N_t = N_src*np.exp(eta_t)
    s_t = transition_density(alpha,sigma_s)
    n_t = n_src*np.exp(s_t)
    r_t = (N_t/n_t)*u.cm.to(u.pc)*abs(1-kappa)

    try:
        C = ((n_t/n_src)*(r_t/rs)**(kappa/(kappa-ks)))
    except:
        C=0.
    rnorm = abs(rs - C*r_t)/abs(C-1)
    #print(r_t)
    nnorm = n_t*(r_t/rnorm)**kappa*(1+r_t/rnorm)**(kappa-ks)
    Nnorm = nnorm*u.cm**-3*rnorm*u.pc/abs(1-ks)
    Nnorm = Nnorm.to(u.cm**-2).value
    #print(nnorm)

    fcorr = 1/(3-ks)*(1-(r_t/rs)**(3-ks))+1/(3-kappa)*(r_t/rs)**(3-kappa)
    return n_src, fcorr

