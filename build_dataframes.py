import os
import sys
import glob

import numpy as np
import pandas as pd

sys.path.insert(1, '..')

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

fpath_in = "../../../gvissio/tilman/results2/"
fpath_out = "/work/users/mtorrassa/biofire-idh/data/"

hv_communities = True # save the dataframes to estimate the hypervolumes metrics or not
NPs = [10, 50]

# species survive with b >= 0.001
dc = 3
bmin = 10**(-dc)
vc = 10**dc

# Compositional diversity (Species Richness, Inverse Simpson Index)
df_totN = pd.DataFrame()

# Dataframe representing the composition of the simulated communities, to estimate the hypervolumes metrices
df_comp = pd.DataFrame(index=['I','C','M','R','L']).T

for NP in NPs:

    print(f'Mediterranean - N{NP}')

    kcom = 0
    filelist = glob.glob(os.path.join(fpath_in, f'coefficients_{NP}_2025-*.txt'))

    for k, info_file in enumerate(filelist):

        print(k)

        nc_i = info_file.split("coefficients")[1]
                    
        bave_file = os.path.join(fpath_in, f'fixed_points{nc_i}')
        fire_file = os.path.join(fpath_in, f'firetimes{nc_i}')

        blist = [f'b{i+1}' for i in range(NP)]
        names = ['ncom', 'init']+[f'b{i+1}' for i in range(NP)]
        df_bave = pd.read_csv(bave_file, sep='\s+',names=names)
        df_bave = df_bave.where(df_bave>bmin).fillna(0)

        df_fr = pd.read_csv(fire_file, sep='\s+', names=['ncom','init','frt'])
        # df_fr = df_fr.groupby(['ncom','sim']).mean().reset_index()

#         df_stat = pd.DataFrame(df_bave.ge(bmin))
#         df_bave['init'] = df_stat.groupby(df_stat.columns.tolist()).ngroup() + 1

        df_res = df_bave
        df_res['ncom'] += kcom
        kcom = df_res['ncom'].max()
        
        df_res['N'] = NP
        df_res['biome'] = 'med'
        df_res['frt']=df_fr['frt']
        df_res = df_res.set_index(['biome','N','ncom','init',])#.groupby(['biome','N','ncom','sim']).mean()
        df_temp = df_res.reset_index() # this will be used later
        df_tot = df_res.drop(columns=blist)

        df_sr = pd.DataFrame(df_res[blist].ge(bmin).sum(axis=1))
        df_tot['srichness']=df_sr

        df_isi = df_res[blist].div(df_res[blist].sum(axis=1), axis=0)**2
        df_isi = 1 / df_isi.sum(axis=1)
        df_tot['isimpson'] = df_isi

        df_totN = pd.concat([df_totN, df_tot])

        # Dataset containing the communities composition that will be used for the estimation of the HV functional diversity metrices
        if hv_communities:
            df_tot = df_tot.reset_index()
            arr = np.loadtxt(info_file)
            for indx, row in df_tot[df_tot['srichness']>1].iterrows():
                
                N = int(row['N'])
                i_com = int(row['ncom'])
                init = int(row['init'])

                i = i_com - 1 - kcom
                a = arr[i]

                # n_com = np.ones(NP) * int(arr[i][0])
                ind = np.arange(0,N)+1

                list_c = [4*j+1 for j in range(0,N)]
                C = arr[i][list_c]

                list_m = [4*j+2 for j in range(0,N)]
                M = arr[i][list_m]

                list_r = [4*j+3 for j in range(0,N)]
                R = arr[i][list_r]

                list_l = [4*j+4 for j in range(0,N)]
                L = arr[i][list_l]

                bmean = df_temp[df_temp.index==indx].drop(columns=['N','biome','ncom','init','frt']).to_numpy()
                # bmean = df_temp[df_temp.index==indx].to_numpy()[:,2:]
                vegcover = np.round(bmean*vc)

                df = pd.DataFrame([ind, C, M , R, L], index=['I','C','M','R','L'], columns=ind).T

                for j,v in enumerate(vegcover):

                    if np.count_nonzero(v) < 2:
                        print("one species")
                        print(v)
                    
                    else:
                        cov = v.tolist()

                        I1 = df['I'].repeat(cov)
                        C1 = df['C'].repeat(cov)
                        M1 = df['M'].repeat(cov)
                        R1 = df['R'].repeat(cov)
                        L1 = df['L'].repeat(cov)

                        df1 = pd.concat([I1, C1, M1, R1, L1], axis=1)
                        df1.to_csv(os.path.join(fpath_out,f'comp_med{NP}/coms-n{NP}-med-{i_com}-{init}.csv'))


# Append the boreal simulation (that are named differently...)
    print(f'Boreal - N{NP}')

    kcom = 0
    filelist = glob.glob(os.path.join(fpath_in, f'bor_coefficients_{NP}_2025-*.txt'))

    for k, info_file in enumerate(filelist):

        print(k)

        nc_i = info_file.split("coefficients")[1]
                    
        bave_file = os.path.join(fpath_in, f'bor_fixed_points{nc_i}')
        fire_file = os.path.join(fpath_in, f'bor_firetimes{nc_i}')

        blist = [f'b{i+1}' for i in range(NP)]
        names = ['ncom', 'init']+[f'b{i+1}' for i in range(NP)]
        df_bave = pd.read_csv(bave_file, sep='\s+',names=names)
        df_bave = df_bave.where(df_bave>bmin).fillna(0)

        df_fr = pd.read_csv(fire_file, sep='\s+', names=['ncom','init','frt'])
        # df_fr = df_fr.groupby(['ncom','sim']).mean().reset_index()

#         df_stat = pd.DataFrame(df_bave.ge(bmin))
#         df_bave['init'] = df_stat.groupby(df_stat.columns.tolist()).ngroup() + 1

        df_res = df_bave
        df_res['ncom'] += kcom
        kcom = df_res['ncom'].max()
        
        df_res['N'] = NP
        df_res['biome'] = 'bor'
        df_res['frt']=df_fr['frt']
        df_res = df_res.set_index(['biome','N','ncom','init',])#.groupby(['biome','N','ncom','sim']).mean()
        df_temp = df_res.reset_index() # this will be used later
        df_tot = df_res.drop(columns=blist)

        df_sr = pd.DataFrame(df_res[blist].ge(bmin).sum(axis=1))
        df_tot['srichness']=df_sr

        df_isi = df_res[blist].div(df_res[blist].sum(axis=1), axis=0)**2
        df_isi = 1 / df_isi.sum(axis=1)
        df_tot['isimpson'] = df_isi

        df_totN = pd.concat([df_totN, df_tot])

        # Dataset containing the communities composition that will be used for the estimation of the HV functional diversity metrices
        if hv_communities:
            df_tot = df_tot.reset_index()
            arr = np.loadtxt(info_file)
            for indx, row in df_tot[df_tot['srichness']>1].iterrows():
                
                N = int(row['N'])
                i_com = int(row['ncom'])
                init = int(row['init'])

                i = i_com - 1 - kcom
                a = arr[i]

                # n_com = np.ones(NP) * int(arr[i][0])
                ind = np.arange(0,N)+1

                list_c = [4*j+1 for j in range(0,N)]
                C = arr[i][list_c]

                list_m = [4*j+2 for j in range(0,N)]
                M = arr[i][list_m]

                list_r = [4*j+3 for j in range(0,N)]
                R = arr[i][list_r]

                list_l = [4*j+4 for j in range(0,N)]
                L = arr[i][list_l]

                bmean = df_temp[df_temp.index==indx].drop(columns=['N','biome','ncom','init','frt']).to_numpy()
                # bmean = df_temp[df_temp.index==indx].to_numpy()[:,2:]
                vegcover = np.round(bmean*vc)

                df = pd.DataFrame([ind, C, M , R, L], index=['I','C','M','R','L'], columns=ind).T

                for j,v in enumerate(vegcover):

                    if np.count_nonzero(v) < 2:
                        print("one species")
                        print(v)
                    
                    else:
                        cov = v.tolist()

                        I1 = df['I'].repeat(cov)
                        C1 = df['C'].repeat(cov)
                        M1 = df['M'].repeat(cov)
                        R1 = df['R'].repeat(cov)
                        L1 = df['L'].repeat(cov)
                    
                        df1 = pd.concat([I1, C1, M1, R1, L1], axis=1) # dataframe for the hypervolume estimation
                        df1.to_csv(os.path.join(fpath_out,f'comp_bor{NP}/coms-n{NP}-bor-{i_com}-{init}.csv'))

# create a dataset with community identification, fire return time and biodiversity indices
df_totN.to_csv(os.path.join(fpath_out, 'coms-fire-bioindex.csv'))