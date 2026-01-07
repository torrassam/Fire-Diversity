import os
import glob
import logging
import argparse

import numpy as np
import pandas as pd

import hypervolumes as hvb

def main():

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="fpath_in", required=True)
    parser.add_argument("--out", dest="fpath_out", required=True)
    args = parser.parse_args()
    
    fpath_in = args.fpath_in
    fpath_out = args.fpath_out

    logging.info("Input directory: %s", fpath_in)
    logging.info("Output directory: %s", fpath_out)

    # fpath_in = "/home/gvissio/tilman/results_idh"
    # fpath_out = "/work/users/mtorrassa/biofire-idh/data_new2/"

    NPs = [10, 50]

    # species survive with b >= 0.001
    dc = 3
    vc = 10**dc

    # Compositional diversity (Species Richness, Inverse Simpson Index)
    df_totN= pd.DataFrame(index=['N','biome','ncom','srichness']).T

    # Dataframe representing the composition of the simulated communities, to estimate the hypervolumes metrices

    for NP in NPs:

        logging.info(f'\nMediterranean - N={NP}\n')

        filelist = sorted(glob.glob(os.path.join(fpath_in, f'coefficients_{NP}_2025-*.txt')))
        k_com = 1

        for k, info_file in enumerate(filelist):
            # logging.info(k_com)

            arr = np.loadtxt(info_file)
            
            for i,a in enumerate(arr):
                i_com = i+k_com
                # logging.info(i_com)

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

                    if NP==10:
                        Cmin, Cmax = 0.0, 0.27 # 0.0, 20.6
                        Mmin, Mmax = 0.0, 0.1 # 0.0, 2.1
                    elif NP==50:
                        Cmin, Cmax = 0.0, 0.3 # 0.0, 22.5
                        Mmin, Mmax = 0.0, 0.1 # 0.0, 2.1
                    else:
                        Cmin, Cmax = 0.0, 0.5 # dummy values
                        Mmin, Mmax = 0.0, 0.1 # dummy values
                    
                    Lmin, Lmax = 0.002, 0.5

                    df['C'] = (df['C'] - Cmin) / (Cmax - Cmin)
                    df['M'] = (df['M'] - Mmin) / (Mmax - Mmin)
                    df['L'] = (df['L'] - Lmin) / (Lmax - Lmin)

                    np_temp = df1[['C','R','L']].to_numpy()

                    try:
                        hv = hvb.hypervolume(np_temp, verbose=False)
                        fd_rich = hvb.kernel_alpha(hv)
                        fd_div = hvb.kernel_dispersion(hv)
                        # fd_reg = hvb.kernel_evenness(hv, mins=mins, maxs=maxs)
                        
                    except Exception as e:
                        logging.info(beq)
                        fd_rich = 0.0
                        fd_div = 0.0

                df_res = pd.DataFrame([NP, 'med', i_com, sr, isi, fd_rich, fd_div], index=['N','biome','ncom','srichness', 'isimpson', 'frichness', 'fdivergence']).T

                df_totN = pd.concat([df_totN, df_res])

            k_com = i_com+1

    # Append the boreal simulation (that are named differently...)
        logging.info(f'\nBoreal - N={NP}\n')

        filelist = sorted(glob.glob(os.path.join(fpath_in, f'bor_coefficients_{NP}_2025-*.txt')))
        k_com = 1

        for k, info_file in enumerate(filelist):
            # logging.info(k_com)

            arr = np.loadtxt(info_file)
            
            for i,a in enumerate(arr):
                i_com = i+k_com
                # logging.info(i_com)

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

                    Cmin, Cmax = 0.0, 0.18 # 0.0, 2.1
                    Mmin, Mmax = 0.0, 0.05 # 0.0, 0.1          
                    Lmin, Lmax = 0.002, 0.5

                    df1['C'] = (df1['C'] - Cmin) / (Cmax - Cmin)
                    df1['M'] = (df1['M'] - Cmin) / (Cmax - Cmin)
                    df1['L'] = (df1['L'] - Lmin) / (Lmax - Lmin)

                    np_temp = df1[['C','R','L']].to_numpy()
                    
                    try:
                        hv = hvb.hypervolume(np_temp, verbose=False)
                        fd_rich = hvb.kernel_alpha(hv)
                        fd_div = hvb.kernel_dispersion(hv)
                        # fd_reg = hvb.kernel_evenness(hv, mins=mins, maxs=maxs)
                        
                    except Exception as e:
                        logging.info(beq)
                        fd_rich = 0.0
                        fd_div = 0.0

                df_res = pd.DataFrame([NP, 'bor', i_com, sr, isi, fd_rich, fd_div], index=['N','biome','ncom','srichness', 'isimpson', 'frichness', 'fdivergence']).T

                df_totN = pd.concat([df_totN, df_res])

            k_com = i_com+1

    # create a dataset with community identification, fire return time and biodiversity indices
    df_totN.to_csv(os.path.join(fpath_out, 'tilman-diversity.csv'))

    logging.info('FiresModel-Tilman difference')

    df1 = pd.read_csv(os.path.join(fpath_out, 'coms-fire-bioindex-fd.csv'))
    df1['eco-type'] = df1['biome'] + df1['N'].astype(str)
    df1['dynamic'] = 'Fires'

    df2 = pd.read_csv(os.path.join(fpath_out, 'tilman-diversity.csv'))
    df2['eco-type'] = df2['biome'] + df2['N'].astype(str)
    df2['dynamic'] = 'Tilman'
    # df2['ncom'] += 1

    # Create the new dataframes
    df3 = df1.copy().drop(columns=['Unnamed: 0','frichness','fdivergence','srichness','isimpson','dynamic'])
    df_diff = pd.DataFrame(columns=['eco-type', 'srichness', 'isimpson', 'frichness', 'fdivergence'])

    for indx, row in df1.iterrows():

        logging.info(f"{row['eco-type']}, {row['ncom']}")

        df_til = df2[(df2['eco-type']==row['eco-type']) & (df2['ncom']==row['ncom'])]

        df_diff.at[indx,'eco-type'] = row['eco-type']
        df_diff.at[indx,'srichness'] = row['srichness'] - df_til['srichness'].values[0]
        df_diff.at[indx,'isimpson'] = row['isimpson'] - df_til['isimpson'].values[0]
        df_diff.at[indx,'frichness'] = row['frichness'] - df_til['frichness'].values[0]
        df_diff.at[indx,'fdivergence'] = row['fdivergence'] - df_til['fdivergence'].values[0]

        df3.at[indx,'eco-type'] = row['eco-type']
        df3.at[indx, 'srich_fire'] = row['srichness']
        df3.at[indx, 'isimp_fire'] = row['isimpson']
        df3.at[indx, 'frich_fire'] = row['frichness']
        df3.at[indx, 'fdiv_fire'] = row['fdivergence']
        
        df3.at[indx, 'srich_comp'] = df_til['srichness'].values[0]
        df3.at[indx, 'isimp_comp'] = df_til['isimpson'].values[0]
        df3.at[indx, 'frich_comp'] = df_til['frichness'].values[0]
        df3.at[indx, 'fdiv_comp'] = df_til['fdivergence'].values[0]

    df_diff.to_csv(os.path.join(fpath_out, 'biodindex-difference.csv'))
    df3.to_csv(os.path.join(fpath_out, 'comp-vs-fire-diversity.csv'))

    logging.info("Program completed successfully.\n\n")

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
    # logging.info('Portion occupied at equilibrium (Tilman):')
    # logging.info('i \t b_eq')
    beq = np.zeros(NP)
    for ii in range(NP):
        beq[ii] = 1 - M[ii]/C[ii] - A[ii]*np.sum(beq[0:ii]*(1+C[0:ii]/C[ii]))
        if beq[ii] < 0:
            beq[ii] = 0.0
        # cond = C[ii] - M[ii] > 0
        # logging.info(f'{ii+1} \t {beq[ii]})
    # logging.info(f'occupied space at equilibrium: {np.sum(beq)}\n')
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
        logging.warning('TROPPE DIMENSIONI, restituirò un NAN!')
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
        logging.warning('TROPPE DIMENSIONI, restituirò un NAN!')
        isi = np.nan
    return isi

# Call script from external library
if __name__ == "__main__":
    main()
    
# ----------------------------------------------------------------------------