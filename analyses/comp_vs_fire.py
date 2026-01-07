import os
import glob
import logging
import argparse

import numpy as np
import pandas as pd

def main():

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="fpath_in", required=True)
    parser.add_argument("--out", dest="fpath_out", required=True)
    args = parser.parse_args()
    
    fpath_in = args.fpath_in
    fpath_out = args.fpath_out

    logging.info(f"Input directory: {fpath_in}")
    logging.info(f"Output directory: {fpath_out}")

    # fpath_in = "/home/gvissio/tilman/results_idh"
    # fpath_out = "/work/users/mtorrassa/biofire-idh/data_new2/"

    NPs = [10, 50]

    # species survive with b >= 0.001
    dc = 3
    vc = 10**dc

    # Reas the already existing files
    df1 = pd.read_csv(os.path.join(fpath_out, 'coms-fire-bioindex-fd.csv'), index_col=0)
    df2 = pd.read_csv(os.path.join(fpath_out, 'tilman-diversity.csv'))

    # Create the new dataframe
    df3 = df1.copy().drop(columns=['Unnamed: 0','frichness','fdivergence','srichness','isimpson'])

    # Dataframe representing the composition of the simulated communities, to estimate the hypervolumes metrices

    for NP in NPs:

        logging.info(f'\nMediterranean - N={NP}\n')
        
        biome = 'med'
        filelist = sorted(glob.glob(os.path.join(fpath_in, f'coefficients_{NP}_2025-*.txt')))
        k_com = 1

        for k, info_file in enumerate(filelist):
            # logging.info(k_com)

            arr = np.loadtxt(f'{info_file}')
            
            for i,a in enumerate(arr):
                i_com = i+k_com
                logging.info(f'{i_com}')

                ind = np.arange(0,NP)+1

                list_c = [4*j+1 for j in range(0,NP)]
                C = arr[i][list_c]
                list_m = [4*j+2 for j in range(0,NP)]
                M = arr[i][list_m]
                list_r = [4*j+3 for j in range(0,NP)]
                R = arr[i][list_r]

                # Compositional Diversity Indicators - Competition effect
                sr_comp = df2[(df2['ncom']==i_com)&(df2['biome']==biome)&(df2['N']==NP)]['srichness'].values[0]
                isi_comp = df2[(df2['ncom']==i_com)&(df2['biome']==biome)&(df2['N']==NP)]['isimpson'].values[0]

                # Compositional Diversity Indicators - Fire response effect
                for index,row in df3[(df3['ncom']==i_com)&(df3['biome']==biome)&(df3['N']==NP)].iterrows():
                    Tf = row['frt']

                    bfire = equil_til_fires(NP, C, M, R, Tf)
                    sr_fire = np.count_nonzero(bfire)
                    isi_fire = 1/np.sum(np.power(bfire/np.sum(bfire),2))

                    df3.at[index, 'srich_fire'] = sr_fire
                    df3.at[index, 'isimp_fire'] = isi_fire
                    df3.at[index, 'srich_comp'] = sr_comp
                    df3.at[index, 'isimp_comp'] = isi_comp

            k_com = i_com+1

    # Append the boreal simulation (that are named differently...)
        logging.info(f'\nBoreal - N={NP}\n')

        biome = 'bor'
        filelist = sorted(glob.glob(os.path.join(fpath_in, f'bor_coefficients_{NP}_2025-*.txt')))
        k_com = 1

        for k, info_file in enumerate(filelist):
            # logging.info(k_com)

            arr = np.loadtxt(f'{info_file}')
            
            for i,a in enumerate(arr):
                i_com = i+k_com
                logging.info(f'{i_com}')

                ind = np.arange(0,NP)+1

                list_c = [4*j+1 for j in range(0,NP)]
                C = arr[i][list_c]
                list_m = [4*j+2 for j in range(0,NP)]
                M = arr[i][list_m]
                list_r = [4*j+3 for j in range(0,NP)]
                R = arr[i][list_r]

                # Compositional Diversity Indicators - Competition effect
                sr_comp = df2[(df2['ncom']==i_com)&(df2['biome']==biome)&(df2['N']==NP)]['srichness'].values[0]
                isi_comp = df2[(df2['ncom']==i_com)&(df2['biome']==biome)&(df2['N']==NP)]['isimpson'].values[0]

                # Compositional Diversity Indicators - Fire response effect
                for index,row in df3[(df3['ncom']==i_com)&(df3['biome']==biome)&(df3['N']==NP)].iterrows():
                    Tf = row['frt']

                    bfire = equil_til_fires(NP, C, M, R, Tf)
                    sr_fire = np.count_nonzero(bfire)
                    isi_fire = 1/np.sum(np.power(bfire/np.sum(bfire),2))

                    df3.at[index, 'srich_fire'] = sr_fire
                    df3.at[index, 'isimp_fire'] = isi_fire
                    df3.at[index, 'srich_comp'] = sr_comp
                    df3.at[index, 'isimp_comp'] = isi_comp

            k_com = i_com+1

    df3.to_csv(os.path.join(fpath_out, 'comp-vs-fire-diversity.csv'))
    
    logging.info('Program Completed Successfully')


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


def equil_til_fires(NP, Clist, Mlist, Rlist, Tf):
    """

    """
    C = np.array(Clist)
    M = np.array(Mlist)
    R = np.array(Rlist)

    # total mortality (with fire response effect)
    M_tot = M - np.log(R)/Tf

    # PARAMS FOR IMPERFECT HIERARCHY
    A = np.ones(NP)
    A[0] = 0.0

    beq = np.zeros(NP)
    for ii in range(NP):
        beq[ii] = 1 - M_tot[ii]/C[ii] - A[ii]*np.sum(beq[0:ii]*(1+C[0:ii]/C[ii]))
        if beq[ii] < 0:
            beq[ii] = 0.0
        # cond = C[ii] - M[ii] > 0
        # logging.info(f'{ii+1} \t {beq[ii]})
    # logging.info(f'occupied space at equilibrium: {np.sum(beq)}\n')
    return beq

# Call script from external library
if __name__ == "__main__":
    main()
    
# ----------------------------------------------------------------------------