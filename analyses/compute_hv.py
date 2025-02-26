import os
import sys

import pandas as pd

sys.path.insert(1, '..')

import hypervolume_bat as hvb

fpath_in = "../../../gvissio/tilman/results2/"
fpath_out = "/work/users/mtorrassa/biofire-idh/data/"

cbio_file = os.path.join(fpath_out, 'coms-fire-bioindex.csv')
df_cbio = pd.read_csv(cbio_file)

Cmin, Cmax = 0.0, 0.3
Mmin, Mmax = 0.0, 0.09
Lmin, Lmax = 0.002, 0.5

df_cbio['frichness'] = 0.0
df_cbio['fdivergence'] = 0.0

for index, row in df_cbio[df_cbio['srichness']>1].iterrows():

    NP = row['N']
    biome = row['biome']
    i_com = int(row['ncom'])
    init = int(row['init'])

    print(f'{biome} - N={NP}')

    df = pd.read_csv(os.path.join(fpath_out,f'comp_{biome}{NP}/coms-n{NP}-{biome}-{i_com}-{init}.csv'))
    # Normalization (Range Transformation)
    df['I'] = (df['I'] - 1) / (NP-1)
    df['C'] = (df['C'] - Cmin) / (Cmax - Cmin)
    df['M'] = (df['M'] - Mmin) / (Mmax - Mmin)
    df['L'] = (df['L'] - Lmin) / (Lmax - Lmin)

    np_temp = df.to_numpy()[:,1:]

    try:
        hv = hvb.hypervolume(np_temp, verbose=True)
        fd_rich = hvb.kernel_alpha(hv)
        fd_div = hvb.kernel_dispersion(hv)
        
    except Exception as e:
        fd_rich = 0.0
        fd_div = 0.0

    df_cbio.at[index,'frichness'] = fd_rich
    df_cbio.at[index,'fdivergence'] = fd_div

# create a dataset with the communities composition
df_cbio.to_csv(os.path.join(fpath_out, 'coms-fire-bioindex-fd.csv'))