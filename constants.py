import numpy as np

#(* Constants *)
gcm2 = 0.000208836 # added
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
hbar = 6.62606957 * 1e-27/(2 * np.pi) # \[HBar] = hbar; \\[Pi] = np.pi
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
