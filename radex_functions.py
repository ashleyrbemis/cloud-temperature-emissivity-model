
# Some RADEX-related scripts
from decimal import Decimal
import pandas as pd
from random import random
import numpy as np
import os
import shutil
from PDF_model_functions import sound_speed

def write_input(n_array,temp,mole,flow,fupp,bw,tbg,cdmol,dv,tmp):
    '''
    Write input file for a set of RADEX runs
    '''
    infile = open(tmp+'.inp',mode='a')
    i = 0
    for n_,cd_ in zip(n_array,cdmol):
        nh2 = '%.2E' % Decimal(n_)
        cdh2 = '%.2E' % Decimal(cd_)
        temp = '%.2E' % Decimal(temp)
        dv = '%.2F' % Decimal(dv)
        infile = open(tmp+'.inp','a')
        infile.write(mole+'.dat\n')
        infile.write('radex.'+tmp+'\n')
        infile.write(str(flow*(1-bw))+' '+str(fupp/(1-bw))+'\n')
        infile.write(str(temp)+'\n')
        infile.write('1\n')
        infile.write('H2\n')
        infile.write(str(nh2)+'\n')
        infile.write(str(tbg)+'\n')
        infile.write(str(cdh2)+'\n')
        infile.write(str(dv)+'\n')
        if i < len(n_array):
            infile.write('1\n')
        i=i+1
    infile.write('0\n')
    infile.close()

def write_input_temp(n_array,temp,mole,flow,fupp,bw,tbg,cdmol,dv,tmp):
    '''
    Write input file for a set of RADEX runs
    '''
    infile = open(tmp+'.inp',mode='a')
    i = 0
    for n_,cd_,t_ in zip(n_array,cdmol,temp):
        nh2 = '%.2E' % Decimal(n_)
        cdh2 = '%.2E' % Decimal(cd_)
        temp = '%.2E' % Decimal(t_)
        dv = '%.2F' % Decimal(dv)
        infile = open(tmp+'.inp','a')
        infile.write(mole+'.dat\n')
        infile.write('radex.'+tmp+'\n')
        infile.write(str(flow*(1-bw))+' '+str(fupp/(1-bw))+'\n')
        infile.write(str(temp)+'\n')
        infile.write('1\n')
        infile.write('H2\n')
        infile.write(str(nh2)+'\n')
        infile.write(str(tbg)+'\n')
        infile.write(str(cdh2)+'\n')
        infile.write(str(dv)+'\n')
        if i < len(n_array):
            infile.write('1\n')
        i=i+1
    infile.write('0\n')
    infile.close()
    
    

def write_input_dv(n_array,tkin,mole,flow,fupp,bw,tbg,cdmol,dv,tmp):
    '''
    Write input file for a set of RADEX runs
    '''
    
    infile = open(tmp+'.inp','a')
    
    i = 0
    for n_,cd_,dv_ in zip(n_array,cdmol,dv):
        nh2 = '%.2E' % Decimal(n_)
        cdh2 = '%.2E' % Decimal(cd_)
        temp = '%.2E' % Decimal(tkin)
        dv_ = '%.2F' % Decimal(dv_)
        infile.write(mole+'.dat\n')
        infile.write('radex.'+tmp+'\n')
        infile.write(str(flow*(1-bw))+' '+str(fupp/(1-bw))+'\n')
        infile.write(str(temp)+'\n')
        infile.write('1\n')
        infile.write('H2\n')
        infile.write(str(nh2)+'\n')
        infile.write(str(tbg)+'\n')
        infile.write(str(cdh2)+'\n')
        infile.write(str(dv_)+'\n')
        if i <= len(n_array):
            infile.write('1\n')
        i=i+1
    infile.write('0\n')
    infile.close()

def write_input_one(n_,tkin,mole,flow,fupp,bw,tbg,cd_,dv_,tmp):
    '''
    Write input file for a single RADEX run
    '''
    infile = open(tmp+'.inp','a')
    nh2 = '%.2E' % Decimal(n_)
    cdh2 = '%.2E' % Decimal(cd_)
    infile.write(mole+'.dat\n')
    infile.write('radex.'+tmp+'\n')
    infile.write(str(flow*(1-bw))+' '+str(fupp/(1-bw))+'\n')
    infile.write(str(tkin)+'\n')
    infile.write('1\n')
    infile.write('H2\n')
    infile.write(str(nh2)+'\n')
    infile.write(str(tbg)+'\n')
    infile.write(str(cdh2)+'\n')
    infile.write(str(dv_)+'\n')
    infile.write('0\n')
    infile.close()

def get_fup_flo_trans(mol,jup,jlo):
    if mol == 'co':
        if (jup == 1):
            fup = 116
        elif (jup == 2):
            fup = 231
        elif (jup == 3):
            fup = 346
        elif (jup == 4):
            fup = 462
        if (jlo == 0):
            flo = 114
        elif (jlo == 1):
            flo = 229
        elif(jlo == 2):
            flo = 244
        elif(jlo == 3):
            flo = 460
    elif mol == 'hcn':
        if (jup == 1):
            fup = 90
        elif (jup == 2):
            fup = 179
        elif (jup == 3):
            fup = 268
        elif (jup == 4):
            fup = 357
        if (jlo == 0):
            flo = 88
        elif (jlo == 1):
            flo = 177
        elif(jlo == 2):
            flo = 265
        elif(jlo == 3):
            flo = 354
    elif mol == 'hco+':
        if (jup == 1):
            fup = 100
        elif (jup == 2):
            fup = 180
        elif (jup == 3):
            fup = 270
        elif (jup == 4):
            fup = 360
        if (jlo == 0):
            flo = 80
        elif (jlo == 1):
            flo = 170
        elif(jlo == 2):
            flo = 260
        elif(jlo == 3):
            flo = 350
    return flo, fup

def parse_outfile_Nlines(f,nlines):
    
    lines = open(f).readlines()
    lines = [l.strip() for l in lines]
    lines = np.hstack(lines)

    ind0 = [i+2 for i in np.arange(len(lines)) if 'FLUX' in lines[i]]
    ind1 = np.array(ind0)+nlines
    ind = [list(np.arange(ind0[i], ind1[i])) for i in np.arange(len(ind0))]

    out_lines = []
    for li in lines[ind]:
        for l in li:
            lsplit = l.split()
            line_id = str(int(lsplit[0]))+' '+str(int(lsplit[2]))
            out_lines.append((line_id+' '+' '.join(lsplit[3:])).split())
    
    dens = np.repeat([l.split(' ')[-1] for l in lines if 'Density of H2  [cm-3]:' in l],nlines)
    cd = np.repeat([l.split(' ')[-1] for l in lines if '* Column density [cm-2]:' in l],nlines)
    dv = np.repeat([l.split(' ')[-1] for l in lines if '* Line width     [km/s]:' in l],nlines)
    tkin = np.repeat([l.split(' ')[-1] for l in lines if '* T(kin)            [K]:' in l],nlines)

    cols = lines[ind0[0]-2].split()[1:]
    cols.insert(0,'JUP')
    cols.insert(1,'JLO')
    units = lines[ind0[0]-1].split()
    units.insert(0,'')
    units.insert(1,'')
    i = int(np.argwhere(np.array(cols) == 'TAU'))
    units.insert(i,'')
    units = [u.strip('()').replace('*','') for u in units]
    colnames = [' '.join((c,u)).strip().replace(' ','_') for c,u in zip(cols,units)]
    
    df_radex = pd.DataFrame(data = out_lines, columns = colnames)
    df_radex['nH2_cm-3'] = dens
    df_radex['Nmol_cm-2'] = cd
    df_radex['dv_km/s'] = dv
    df_radex['temp_K'] = tkin
    df_radex = df_radex.apply(pd.to_numeric,errors='coerce')
    
    return df_radex


def run_radex(mole,cdmol,n_array,jup,jlo,temp,dv,bw,tbg,model_num=1):
    flow, fupp = get_fup_flo_trans(mole,jup,jlo)
    nlines = jup-jlo

    tmp = str(int(random()*1e6))
    if isinstance(temp,(float,int)):
        write_input(n_array,temp,mole,flow,fupp,bw,tbg,cdmol,dv,tmp)
    elif isinstance(temp,(np.ndarray,list)):
        write_input_temp(n_array,temp,mole,flow,fupp,bw,tbg,cdmol,dv,tmp)
    
    os.system('radex < '+tmp+'.inp'+' > /dev/null 2>&1')
    df_radex = parse_outfile_Nlines('radex.'+tmp,nlines)
    
    os.system('rm -rf radex.'+tmp)
    os.system('rm -rf '+tmp+'.inp')
    return df_radex

from subprocess import call
import subprocess as sp
def run_radex_one(mole,cdmol,n_array,jup,jlo,temp,dv,bw,tbg,nlines):
    flow, fupp = get_fup_flo_trans(mole,jup,jlo)
    tmp = str(int(random()*1e6))
    try:
        call("rm "+"radex."+tmp+" > /dev/null 2>&1",shell=True)
        call("rm "+tmp+".inp"+" > /dev/null 2>&1",shell=True)
    except:
        pass

    df_out = None
    #try:
    write_input(n_array,temp,mole,flow,fupp,bw,tbg,cdmol,dv,tmp)
    call('radex < '+tmp+'.inp'+' > /dev/null 2>&1',shell=True) #
    df_out = parse_outfile_Nlines('radex.'+tmp,n_array,nlines)
    call("rm "+"radex."+tmp+" > /dev/null 2>&1",shell=True)
    call("rm "+tmp+".inp"+" > /dev/null 2>&1",shell=True)
    drop = [c_ for c_ in df_out.columns if ('--' in str(c_)) or ('Unnamed' in str(c_))]
    df_out = df_out.drop(labels=drop,axis='columns')
    drop = [c_ for c_ in df_out.columns if 'JLO_' in str(c_)]
    df_out = df_out.drop(labels=drop,axis='columns')
    drop = [c_ for c_ in df_out.columns if 'JUP_' in str(c_)]
    df_out = df_out.drop(labels=drop,axis='columns')
    #except:
    #    print('fail')
    #    pass

   
    i = 0
    try:
        n_max = np.nanmax(df_out.dropna()['nH2_cm-3'].values)
        i = np.argmax(df_out.dropna()['nH2_cm-3'].values)
        #print(df_out['FLUX_Kkm/s'].iloc[i])
    except:
        pass

    '''
    left = len(n_array)-1 - i
    #print(left)
    count = 0
    while True:
        if (i >= len(n_array)-1) or (count >= left):
            break
        try:
            #print('i ',i)
            write_input(n_array[i:],temp,mole,flow,fupp,bw,tbg,cdmol[i:],dv,tmp)
            call('radex < '+tmp+'.inp > /dev/null 2>&1',shell=True)
            df_ = parse_outfile_Nlines('radex.'+tmp,n_array,nlines)
            df_out = df_out.append(df_)
            n_max = np.nanmax(df_out.dropna()['nH2_cm-3'].values)
            i = np.argmin(abs(1-n_max/n_array))
            call("rm "+"radex."+tmp+" > /dev/null 2>&1",shell=True)
            call("rm "+tmp+".inp"+" > /dev/null 2>&1",shell=True)
        except:
            count = count + 1
            #print('iter ',count)
            pass
    
    try:
        call("rm "+"radex."+tmp+" > /dev/null 2>&1",shell=True)
        call("rm "+tmp+".inp"+" > /dev/null 2>&1",shell=True)
    except:
        pass
    '''
            
    return df_out

def run_multiline(n_array, N_array, temp = 10.,
                  X_hcn=1e-8,X_co=1.4e-4,X_hcop=1e-8,
                  dv = 1.,jup=4,jlo=0,hcop=True,model_num=1):
    
    bw    = 0.001 # bandwidth
    tbg   = 0. #2.73 # background radiation temperature

    match = ['nH2_cm-3','JUP','JLO','temp_K','dv_km/s']
    
    # CO
    mole = 'co'
    cdmol = X_co*N_array
    df_co = run_radex(mole,cdmol,n_array,jup,jlo,temp,dv,bw,tbg,model_num=model_num)
    cols = df_co.columns
    cols = list(set(cols)-set(match))
    newcols = [c+'_'+mole for c in cols]
    newcols = newcols + match
    d = {}
    for key,value in zip(cols,newcols):
        d[key] = value
    df_co = df_co.rename(columns = d)
    
    # HCN
    mole = 'hcn'
    cdmol = X_hcn*N_array
    df_hcn = run_radex(mole,cdmol,n_array,jup,jlo,temp,dv,bw,tbg,model_num=model_num)
    cols = df_hcn.columns
    cols = list(set(cols)-set(match))
    newcols = [c+'_'+mole for c in cols]
    newcols = newcols + match
    d = {}
    for key,value in zip(cols,newcols):
        d[key] = value
    df_hcn = df_hcn.rename(columns = d)

    df_out = df_co.merge(df_hcn,on = match,how='outer')

    if hcop == True:
        # HCO+
        mole = 'hco+'
        cdmol = X_hcop*N_array
        df_hcop = run_radex(mole,cdmol,n_array,jup,jlo,temp,dv,bw,tbg,model_num=model_num)
        cols = df_hcop.columns
        cols = list(set(cols)-set(match))
        newcols = [c+'_'+mole for c in cols]
        newcols = newcols + match
        d = {}
        for key,value in zip(cols,newcols):
            d[key] = value
        df_hcop = df_hcop.rename(columns = d)

        df_out = df_out.merge(df_hcop,on = match,how='outer')
    
    return df_out




