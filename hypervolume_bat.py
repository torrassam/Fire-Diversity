import os
from math import ceil

import numpy as np
import pandas as pd
import scipy.stats as scs

from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    
    np.random.seed(0)

    fpath_old = ""
    fpath_out = ""
    fname = 'coms-fire-hv-comparison.csv'

    hvr_file = os.path.join(fpath_old, 'coms-fire-hv_old.csv')
    df_hvr = pd.read_csv(hvr_file)

    # df_hvr2 = df_hvr[df_hvr['NP']==10].drop(columns=['pevenness', 'isimpson', 'cvar_C', 'cvar_R', 'cvar_L', 'idisp_C', 'idisp_R', 'idisp_L'])
    df_hvr2 = df_hvr.drop(columns=['cvar_C', 'cvar_R', 'cvar_L', 'idisp_C', 'idisp_R', 'idisp_L'])

    df_hvr2['fd_rich_new'] = 0.0
    df_hvr2['fd_div_new'] = 0.0
    df_hvr2['fd_reg_new'] = 0.0

    for index, row in df_hvr2[(df_hvr2['srichness']>1)].iterrows():

        NP = row['NP']
        X = row['X']
        subexp = row['sub.experiment']

        df = pd.read_csv(os.path.join(fpath_old,f'communities/coms-{subexp}-n{int(NP)}-{int(X)}.csv'))
        df['I'] = np.log(df['I'])
        df['C'] = np.log(df['C'])
        df = df.drop(columns=['M']).dropna()
        np_temp = df.to_numpy()[:,1:]
        
        mins = np.array([np.log(1), np.log(0.001), 0.001, 0.001])
        maxs = np.array([np.log(NP), np.log(150), 1.0, 1.0])
        
        try:       
            hv = hypervolume(np_temp, verbose=True)
            fd_rich = kernel_alpha(hv)
            fd_div = kernel_dispersion(hv)
            fd_reg = kernel_evenness(hv, mins=mins, maxs=maxs)
        except Exception as e:
            fd_rich = 0.0
            fd_div = 0.0
            fd_reg = 0.0

        df_hvr2.at[index,'fd_rich_new'] = fd_rich
        df_hvr2.at[index,'fd_div_new'] = fd_div
        df_hvr2.at[index,'fd_reg_new'] = fd_reg

    # create a dataset with the communities composition
    df_hvr2.to_csv(os.path.join(fpath_out, fname))

    plot_fig(fpath_out, fname)

# -------------------------------------------------------------------------------------

def plot_fig(fpath, fname):

    hv_file = os.path.join(fpath, fname)
    df_hv = pd.read_csv(hv_file)

    df_comp = df_hv[df_hv['srichness']>1]#[df_hv['NP']==10]
    df_comp=df_comp.rename(columns={"fd_reg": "fd_eve", "fd_reg_new": "fd_eve_new"})

    fig, ax2 = plt.subplots(1,2, figsize=(12,5), dpi=100)
    fig.suptitle("Hypervolume functional diversity metrics comparison (R-Python)")

    sns.regplot(data=df_comp, x='fd_rich', y='fd_rich_new', ax=ax2[0])
    c, p = scs.pearsonr(df_comp['fd_rich'],df_comp['fd_rich_new'])
    ax2[0].set_title(f'corr={round(c,2)}, p-value={round(p,2)}')

    sns.regplot(data=df_comp, x='fd_div', y='fd_div_new', ax=ax2[1])
    c, p = scs.pearsonr(df_comp['fd_div'],df_comp['fd_div_new'])
    ax2[1].set_title(f'corr={round(c,2)}, p-value={round(p,2)}')
        
    axs = ax2.flatten()
    for ax in axs:
        ax.grid(True)

    plt.show()
        
    # ax2[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)

# -------------------------------------------------------------------------------------

def silverman_bandwidth(data):
    """
    Estimate the bandwidth using Silverman's rule of thumb for multivariate data.
    
    Parameters:
    - data: np.array, shape (m_samples, n_features)
      The data representing the functional traits of species.
    
    Returns:
    - bandwidths: np.array, shape (n_features,)
      The bandwidth vector for each dimension.
    """
    m, n = data.shape
    
    # Calculate the constant factor (4/(n+2))^(1/(n+4)) * m^(-1/(n+4))
    constant = (4 / (n + 2)) ** (1 / (n + 4)) * m ** (-1 / (n + 4))
    
    # Calculate the bandwidth for each dimension
    bandwidths = constant * np.std(data, axis=0)
    
    return bandwidths

def hypervolume(data, bandwidth=None, samples=None, verbose=False):

    random_seed = 0
    np.random.seed(random_seed)

    if bandwidth is None:
        bandwidth = silverman_bandwidth(data)
    
    scaled_data = data / bandwidth

    # Use scaled bandwidths in KDE
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')  # Bandwidth=1.0 since data is scaled
    kde.fit(scaled_data)
    
    
    if samples is None:
        samples = ceil((10**(3 + np.sqrt(np.shape(data)[1])))/np.shape(data)[0])
    
    if verbose:
        print('bandwidths:', bandwidth)
        print('sample:', samples)

    # Randomly sample points from the KDE
    sampled_points = kde.sample(n_samples=samples, random_state=random_seed)
    # Re-scale the sampled points back to the original scale
    rescaled_sampled_points = sampled_points * bandwidth

    return rescaled_sampled_points



def kernel_alpha(sampled_points):
    """
    Estimate alpha diversity (functional richness) using KDE-based hypervolumes.
    
    Returns:
    - float: Estimated functional richness (volume of the hypervolume).
    """
    
    # Compute the convex hull of the sampled points
    hull = ConvexHull(sampled_points)
    
    return hull.volume

def kernel_dispersion(sampled_points):
    """
    Estimate functional dispersion (average distance to centroid) within the KDE-based hypervolume.
    
    Returns:
    - float: Estimated functional dispersion.
    """
    
    # Compute the centroid of the sampled points
    centroid = np.mean(sampled_points, axis=0)
    
    # Compute the average distance of each point to the centroid
    distances = cdist(sampled_points, centroid.reshape(1, -1))
    
    return np.mean(distances)

# This function looks like the most promosing one, the only limitation is that I have to adjust the traits-space based on the experiment I'm considering
def kernel_evenness(sampled_points, mins=None, maxs=None):
    """
    Estimate functional evenness within the KDE-based hypervolume.
    
    Returns:
    - float: Estimated functional evenness
    """
    
    if np.any(mins)==None:
        mins = np.min(sampled_points, axis=0).to_numpy
    if np.any(maxs)==None:
        maxs = np.max(sampled_points, axis=0).to_numpy

    # df_traits = pd.read_csv(os.path.join(fpath_out,f'coms-n10-traits.csv')).drop(columns=['M', 'Unnamed: 0', 'n_com'])
    # df_traits['I'] = np.log(df_traits['I'])
    # mins = np.min(df_traits, axis=0).to_numpy()
    # maxs = np.max(df_traits, axis=0).to_numpy()
    sampled_theorethical = np.random.uniform(mins, maxs, size=(sampled_points.shape[0], sampled_points.shape[1]))

    n_bins=12
    H_th_tot = 0
    H_tot = 0
    H_ovl_tot = 0

    for dim in range(0,4):
        h, e = np.histogram(sampled_theorethical[:,dim], bins=n_bins, range=(mins[dim], maxs[dim]), density=True)
        H_th_tot += h.sum()
        h, e = np.histogram(sampled_points[:,dim], bins=n_bins, range=(mins[dim], maxs[dim]), density=True)
        H_tot += h.sum()

    for dim in range(0,4):
        H_th, e = np.histogram(sampled_theorethical[:,dim], bins=n_bins, range=(mins[dim], maxs[dim]), density=True)
        H_th = H_th / H_th_tot

        H, e = np.histogram(sampled_points[:,dim], bins=n_bins, range=(mins[dim], maxs[dim]), density=True)
        H = H / H_tot

        H_ovl = np.minimum(H,H_th)
        H_ovl_tot += H_ovl.sum()
    
    return H_ovl_tot

# -------------------------------------------------------------------------------------

# Call script from external library
if __name__ == "__main__":
    main()
    
# ----------------------------------------------------------------------------