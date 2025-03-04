import os
import glob

import numpy as np
import pandas as pd

import hypervolume as hvb

def main():

    fpath_in = "../../../gvissio/tilman/results2/"
    fpath_out = "/work/users/mtorrassa/biofire-idh/data/"

    NPs = [10, 50]

    # species survive with b >= 0.001
    dc = 3
    bmin = 10**(-dc)
    vc = 10**dc

    # Compositional diversity (Species Richness, Inverse Simpson Index)
    df_totN= pd.DataFrame(index=['N','biome','ncom','srichness']).T
    df_tot = pd.DataFrame()

    Cmin, Cmax = 0.0, 0.3
    Mmin, Mmax = 0.0, 0.09
    Lmin, Lmax = 0.002, 0.5

    # Dataframe representing the composition of the simulated communities, to estimate the hypervolumes metrices

    for NP in NPs:

        print(f'\nMediterranean - N={NP}\n')

        filelist = glob.glob(os.path.join(fpath_in, f'coefficients_{NP}_2025-*.txt'))
        k_com = 0

        for k, info_file in enumerate(filelist):
            # print(k_com)

            arr = np.loadtxt(info_file)
            
            for i,a in enumerate(arr):
                i_com = i+k_com
                # print(i_com)

                n_com = np.ones(NP) * int(arr[i][0])
                ind = np.arange(0,NP)+1

                list_c = [4*j+1 for j in range(0,NP)]
                C = arr[i][list_c]
                list_m = [4*j+2 for j in range(0,NP)]
                M = arr[i][list_m]
                list_r = [4*j+3 for j in range(0,NP)]
                R = arr[i][list_r]
                list_l = [4*j+4 for j in range(0,NP)]
                L = arr[i][list_l]

                beq = equil_til(NP, C, M)

                # Compositional Diversity Indicators

                sr = np.count_nonzero(beq)
                isi = 1/np.sum(np.power(beq/np.sum(beq),2))

                # Functional Diversity Indicators
                fd_rich = 0
                fd_div = 0

                if sr > 1:
                    cov = np.round(beq*vc).tolist()
                    df = pd.DataFrame([ind, C, M, R, L], index=['I','C','M','R','L'], columns=ind).T
                    I1 = df['I'].repeat(cov)
                    C1 = df['C'].repeat(cov)
                    M1 = df['M'].repeat(cov)
                    R1 = df['R'].repeat(cov)
                    L1 = df['L'].repeat(cov)
                    df1 = pd.concat([I1, C1, M1, R1, L1], axis=1)

                    df1['I'] = (df1['I'] - 1) / (NP-1)
                    df1['C'] = (df1['C'] - Cmin) / (Cmax - Cmin)
                    df1['M'] = (df1['M'] - Mmin) / (Mmax - Mmin)
                    df1['L'] = (df1['L'] - Lmin) / (Lmax - Lmin) 
                    np_temp = df1.to_numpy()[:,1:]

                    try:
                        hv = hvb.hypervolume(np_temp, verbose=False)
                        fd_rich = hvb.kernel_alpha(hv)
                        fd_div = hvb.kernel_dispersion(hv)
                        # fd_reg = hvb.kernel_evenness(hv, mins=mins, maxs=maxs)
                        
                    except Exception as e:
                        print(beq)
                        fd_rich = 0.0
                        fd_div = 0.0

                df_res = pd.DataFrame([NP, 'med', i_com, sr, isi, fd_rich, fd_div], index=['N','biome','ncom','srichness', 'isimpson', 'frichness', 'fdivergence']).T

                df_totN = pd.concat([df_totN, df_res])

            k_com = i_com+1

    # Append the boreal simulation (that are named differently...)
        print(f'\nBoreal - N={NP}\n')

        filelist = glob.glob(os.path.join(fpath_in, f'bor_coefficients_{NP}_2025-*.txt'))
        k_com = 0

        for k, info_file in enumerate(filelist):
            # print(k_com)

            arr = np.loadtxt(info_file)
            
            for i,a in enumerate(arr):
                i_com = i+k_com
                # print(i_com)

                n_com = np.ones(NP) * int(arr[i][0])
                ind = np.arange(0,NP)+1

                list_c = [4*j+1 for j in range(0,NP)]
                C = arr[i][list_c]

                list_m = [4*j+2 for j in range(0,NP)]
                M = arr[i][list_m]

                beq = equil_til(NP, C, M)

                sr = np.count_nonzero(beq)
                isi = 1/np.sum(np.power(beq/np.sum(beq),2))

                # Functional Diversity Indicators
                fd_rich = 0
                fd_div = 0

                if sr > 1:
                    cov = np.round(beq*vc).tolist()
                    df = pd.DataFrame([ind, C, M, R, L], index=['I','C','M','R','L'], columns=ind).T
                    I1 = df['I'].repeat(cov)
                    C1 = df['C'].repeat(cov)
                    M1 = df['M'].repeat(cov)
                    R1 = df['R'].repeat(cov)
                    L1 = df['L'].repeat(cov)
                    df1 = pd.concat([I1, C1, M1, R1, L1], axis=1)

                    df1['I'] = (df1['I'] - 1) / (NP-1)
                    df1['C'] = (df1['C'] - Cmin) / (Cmax - Cmin)
                    df1['M'] = (df1['M'] - Mmin) / (Mmax - Mmin)
                    df1['L'] = (df1['L'] - Lmin) / (Lmax - Lmin)

                    np_temp = df1.to_numpy()[:,1:]
                    try:
                        hv = hvb.hypervolume(np_temp, verbose=False)
                        fd_rich = hvb.kernel_alpha(hv)
                        fd_div = hvb.kernel_dispersion(hv)
                        # fd_reg = hvb.kernel_evenness(hv, mins=mins, maxs=maxs)
                        
                    except Exception as e:
                        print(beq)
                        fd_rich = 0.0
                        fd_div = 0.0

                df_res = pd.DataFrame([NP, 'bor', i_com, sr, isi, fd_rich, fd_div], index=['N','biome','ncom','srichness', 'isimpson', 'frichness', 'fdivergence']).T

                df_totN = pd.concat([df_totN, df_res])

            k_com = i_com+1

    # create a dataset with community identification, fire return time and biodiversity indices
    df_totN.to_csv(os.path.join(fpath_out, 'tilman-diversity.csv'))

    print('FiresModel-Tilman difference')

    df_bp = pd.DataFrame(columns=['eco-type', 'group', 'diversity', 'value'])

    df1 = pd.read_csv(os.path.join(fpath_out, 'coms-fire-bioindex-fd.csv'))
    df1['eco-type'] = df1['biome'] + df1['N'].astype(str)
    df1['dynamic'] = 'Fires'

    df2 = pd.read_csv(os.path.join(fpath_out, 'tilman-diversity.csv'))
    df2['eco-type'] = df2['biome'] + df2['N'].astype(str)
    df2['dynamic'] = 'Tilman'
    df2['ncom'] += 1

    divindex = ['srichness', 'isimpson', 'frichness', 'fdivergence']

    df_diff = pd.DataFrame(columns=['eco-type', 'srichness', 'isimpson', 'frichness', 'fdivergence'])

    for indx, row in df1.iterrows():

        print(row['eco-type'], row['ncom'])

        df_til = df2[(df2['eco-type']==row['eco-type']) & (df2['ncom']==row['ncom'])]

        df_diff.at[indx,'eco-type'] = row['eco-type']
        df_diff.at[indx,'srichness'] = row['srichness'] - df_til['srichness'].values[0]
        df_diff.at[indx,'isimpson'] = row['isimpson'] - df_til['isimpson'].values[0]
        df_diff.at[indx,'frichness'] = row['frichness'] - df_til['frichness'].values[0]
        df_diff.at[indx,'fdivergence'] = row['fdivergence'] - df_til['fdivergence'].values[0]

    df_diff.to_csv(os.path.join(fpath_out, 'biodindex-difference.csv'))


#--------------- (2.3) ANALYSES FUNCTIONS ---------------

def equil_til(NP, Clist, Mlist):
    """
    Computes the equilibrium vegetation cover for a system with no fire (Tilman 1994)
    
    Returns
    -------
    beq : ndarray
    """
    C = np.array(Clist)
    M = np.array(Mlist)
    # PARAMS FOR IMPERFECT HIERARCHY
    A = np.ones(NP)
    A[0] = 0.0
    # print('Portion occupied at equilibrium (Tilman):')
    # print('i', 'b_eq')
    beq = np.zeros(NP)
    for ii in range(NP):
        beq[ii] = 1 - M[ii]/C[ii] - A[ii]*np.sum(beq[0:ii]*(1+C[0:ii]/C[ii]))
        if beq[ii] < 0:
            beq[ii] = 0.0
        # cond = C[ii] - M[ii] > 0
        # print(ii+1, beq[ii])
    # print('occupied space at equilibrium:', np.sum(beq),'\n')
    return beq

def richness(b, bmin=1e-05):
    """
    Computes the Species Richness Index for one or more input communities
    
    Parameters
    -------
    b : array_like
        vegetation cover of the plants of the communities
    bmin : float, optional
        minimum vegetation cover value to consider a plant present in the community. The default value is 1e-05
    
    Returns
    -------
    sr : int or 1-D ndarray
        number of species coexisting in the input communities
    """
    
    nd = np.ndim(b)
    if nd==1:
        sr = np.sum(b>bmin)
    elif nd==2:
        sr = np.sum(b>bmin, axis=1)
    else:
        print('TROPPE DIMENSIONI, restituirò un NAN!')
        sr = np.nan
    return sr

def simpson(b, nnp):
    """
    Computes the Inverse Simpson Index for one or more given communities
    
    Parameters
    -------
    b : 1-D or 2-D array_like
        vegetation cover of the plants of the communities
    nnp : int
        maximum number of species in the community
    
    Returns
    -------
    isi : int or 1-D ndarray
        Inverse Simpson Index of the input communities
    """
    
    nd = np.ndim(b)
    if nd==1:
        p = b / np.sum(b)
        p2 = np.power(p,2)
        isi = 1 / np.sum(p2)
    elif nd==2:
        p = b/np.repeat(np.expand_dims(np.sum(b, axis=1),1), nnp, axis=1) #relative abundance
        p2 = np.power(p,2)
        isi = 1 / np.sum(p2, axis=1)
    else:
        print('TROPPE DIMENSIONI, restituirò un NAN!')
        isi = np.nan
    return isi

# Call script from external library
if __name__ == "__main__":
    main()
    
# ----------------------------------------------------------------------------