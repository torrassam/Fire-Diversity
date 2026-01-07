import os
import glob
import logging
import warnings
import argparse

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

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

    # fpath_in = "/home/gvissio/tilman/results2" # "/home/gvissio/tilman/results_idh" for the exponential fit simulations
    # fpath_out = "/work/users/mtorrassa/biofire-idh/data_new/" # "/work/users/mtorrassa/biofire-idh/data_new2/" for the exponential fit simulations results

    hv_communities = True # save the dataframes to estimate the hypervolumes metrics or not
    NPs = [10, 50]

    # species survive with b >= 0.001
    dc = 3
    bmin = 10**(-dc)
    vc = 10**dc

    # Compositional diversity (Species Richness, Inverse Simpson Index)
    df_totN = pd.DataFrame()

    for NP in NPs:

        logging.info('Mediterranean - N%s', NP)

        kcom = 0
        filelist = sorted(glob.glob(os.path.join(fpath_in, f'coefficients_{NP}_2025-*.txt')))

        for k, info_file in enumerate(filelist):

            logging.info(f'{k}, {info_file}')

            nc_i = info_file.split("coefficients")[1]
            coeff = np.loadtxt(info_file)
                    
            bave_file = os.path.join(fpath_in, f'fixed_points{nc_i}')
            fire_file = os.path.join(fpath_in, f'firetimes{nc_i}')

            blist = [f'b{i+1}' for i in range(NP)]
            names = ['ncom', 'init']+[f'b{i+1}' for i in range(NP)]
            df_bave = pd.read_csv(bave_file, sep='\s+',names=names)
            df_bave = df_bave.where(df_bave>bmin).fillna(0)

            df_fr = pd.read_csv(fire_file, sep='\s+', names=['ncom','init','frt'])

            df_bave['ncom'] += kcom
            kcom = df_bave['ncom'].max()

            df_res = df_bave.copy()
            df_res['N'] = NP
            df_res['biome'] = 'med'
            df_res['frt']=df_fr['frt']
            df_res = df_res.set_index(['biome','N','ncom','init',])#.groupby(['biome','N','ncom','sim']).mean()
            # df_temp = df_res.reset_index() # this will be used later
            df_tot = df_res.drop(columns=blist)

            df_sr = pd.DataFrame(df_res[blist].ge(bmin).sum(axis=1))
            df_tot['srichness']=df_sr

            df_isi = df_res[blist].div(df_res[blist].sum(axis=1), axis=0)**2
            df_isi = 1 / df_isi.sum(axis=1)
            df_tot['isimpson'] = df_isi

            df_tot.reset_index(inplace=True)

            for index, row in df_bave.iterrows():
                i_com = int(row['ncom'])
                init = int(row['init'])
                bmean = row[2:].to_numpy()

                # Remove duplicates
                duplicate=False
                for ic in range(init-1):
                    bmean_2 = df_bave[(df_bave['ncom']==i_com)&(df_bave['init']==ic+1)].values[0][2:]
                    if abs(bmean - bmean_2).max()<0.001:
                        # logging.info(f"duplicate of {i_com}-{ic+1}")
                        df_tot.drop(index=df_tot[(df_tot['ncom']==i_com)&(df_tot['init']==init)].index, inplace=True)
                        duplicate=True
                        break
                
                if duplicate==False and hv_communities and df_tot[(df_tot['ncom']==i_com)&(df_tot['init']==init)]['srichness'].values[0]>1:
                    i = i_com - 1 - kcom
                    a = coeff[i]

                    # n_com = np.ones(NP) * int(arr[i][0])
                    ind = np.arange(0,NP)+1

                    list_c = [4*j+1 for j in range(0,NP)]
                    C = coeff[i][list_c]

                    list_m = [4*j+2 for j in range(0,NP)]
                    M = coeff[i][list_m]

                    list_r = [4*j+3 for j in range(0,NP)]
                    R = coeff[i][list_r]

                    list_l = [4*j+4 for j in range(0,NP)]
                    L = coeff[i][list_l]

                    vegcover = np.round(bmean*vc)

                    df = pd.DataFrame([ind, C, M , R, L], index=['I','C','M','R','L'], columns=ind).T
                    
                    if np.count_nonzero(vegcover) < 2:
                        logging.info("one species")
                        logging.info(vegcover)
                    
                    else:
                        # logging.info('save_hv_com')
                        cov = vegcover.tolist()

                        I1 = df['I'].repeat(cov)
                        C1 = df['C'].repeat(cov)
                        M1 = df['M'].repeat(cov)
                        R1 = df['R'].repeat(cov)
                        L1 = df['L'].repeat(cov)

                        df1 = pd.concat([I1, C1, M1, R1, L1], axis=1)
                        df1.to_csv(os.path.join(fpath_out,f'comp_med{NP}/coms-n{NP}-med-{i_com}-{init}.csv'))

            df_totN = pd.concat([df_totN, df_tot])


    # Append the boreal simulation (that are named differently...)
        logging.info(f'Boreal - N{NP}')

        kcom = 0
        filelist = sorted(glob.glob(os.path.join(fpath_in, f'bor_coefficients_{NP}_2025-*.txt')))

        for k, info_file in enumerate(filelist):

            logging.info(f'{k}, {info_file}')

            nc_i = info_file.split("coefficients")[1]
            coeff = np.loadtxt(info_file)
                    
            bave_file = os.path.join(fpath_in, f'bor_fixed_points{nc_i}')
            fire_file = os.path.join(fpath_in, f'bor_firetimes{nc_i}')

            blist = [f'b{i+1}' for i in range(NP)]
            names = ['ncom', 'init']+[f'b{i+1}' for i in range(NP)]
            df_bave = pd.read_csv(bave_file, sep='\s+',names=names)
            df_bave = df_bave.where(df_bave>bmin).fillna(0)

            df_fr = pd.read_csv(fire_file, sep='\s+', names=['ncom','init','frt'])

            df_bave['ncom'] += kcom
            kcom = df_bave['ncom'].max()

            df_res = df_bave.copy()
            df_res['N'] = NP
            df_res['biome'] = 'bor'
            df_res['frt']=df_fr['frt']
            df_res = df_res.set_index(['biome','N','ncom','init',])#.groupby(['biome','N','ncom','sim']).mean()
            # df_temp = df_res.reset_index() # this will be used later
            df_tot = df_res.drop(columns=blist)

            df_sr = pd.DataFrame(df_res[blist].ge(bmin).sum(axis=1))
            df_tot['srichness']=df_sr

            df_isi = df_res[blist].div(df_res[blist].sum(axis=1), axis=0)**2
            df_isi = 1 / df_isi.sum(axis=1)
            df_tot['isimpson'] = df_isi

            df_tot.reset_index(inplace=True)

            for index, row in df_bave.iterrows():
                i_com = int(row['ncom'])
                init = int(row['init'])
                bmean = row[2:].to_numpy()

                # Remove duplicates
                duplicate=False
                for ic in range(init-1):
                    bmean_2 = df_bave[(df_bave['ncom']==i_com)&(df_bave['init']==ic+1)].values[0][2:]
                    if abs(bmean - bmean_2).max()<0.001:
                        # logging.info(f"duplicate of {i_com}-{ic+1}")
                        df_tot.drop(index=df_tot[(df_tot['ncom']==i_com)&(df_tot['init']==init)].index, inplace=True)
                        duplicate=True
                        break
                
                if duplicate==False and hv_communities and df_tot[(df_tot['ncom']==i_com)&(df_tot['init']==init)]['srichness'].values[0]>1:
                    i = i_com - 1 - kcom
                    a = coeff[i]

                    # n_com = np.ones(NP) * int(arr[i][0])
                    ind = np.arange(0,NP)+1

                    list_c = [4*j+1 for j in range(0,NP)]
                    C = coeff[i][list_c]

                    list_m = [4*j+2 for j in range(0,NP)]
                    M = coeff[i][list_m]

                    list_r = [4*j+3 for j in range(0,NP)]
                    R = coeff[i][list_r]

                    list_l = [4*j+4 for j in range(0,NP)]
                    L = coeff[i][list_l]

                    vegcover = np.round(bmean*vc)

                    df = pd.DataFrame([ind, C, M , R, L], index=['I','C','M','R','L'], columns=ind).T
                    
                    if np.count_nonzero(vegcover) < 2:
                        logging.info("one species")
                        logging.info(vegcover)
                    
                    else:
                        # logging.info('save_hv_com')
                        cov = vegcover.tolist()

                        I1 = df['I'].repeat(cov)
                        C1 = df['C'].repeat(cov)
                        M1 = df['M'].repeat(cov)
                        R1 = df['R'].repeat(cov)
                        L1 = df['L'].repeat(cov)

                        df1 = pd.concat([I1, C1, M1, R1, L1], axis=1)
                        df1.to_csv(os.path.join(fpath_out,f'comp_bor{NP}/coms-n{NP}-bor-{i_com}-{init}.csv'))
            
            df_totN = pd.concat([df_totN, df_tot])

    # create a dataset with community identification, fire return time and biodiversity indices
    df_totN.to_csv(os.path.join(fpath_out, 'coms-fire-bioindex.csv'))

    logging.info("Program completed successfully.\n\n")

# ----------------------------------------------------------------------------

# Call script from external library
if __name__ == "__main__":
    main()
    
# ----------------------------------------------------------------------------