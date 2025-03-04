import os
from math import ceil
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

def main():

    fpath_dir = "/work/users/mtorrassa/biofire-idh/data/"

    cbio_file = os.path.join(fpath_dir, 'coms-fire-bioindex.csv')
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

        df = pd.read_csv(os.path.join(fpath_dir,f'comp_{biome}{NP}/coms-n{NP}-{biome}-{i_com}-{init}.csv'))
        # Normalization (Range Transformation)
        df['I'] = (df['I'] - 1) / (NP-1)
        df['C'] = (df['C'] - Cmin) / (Cmax - Cmin)
        df['M'] = (df['M'] - Mmin) / (Mmax - Mmin)
        df['L'] = (df['L'] - Lmin) / (Lmax - Lmin)

        np_temp = df.to_numpy()[:,1:]

        try:
            hv = hypervolume(np_temp, verbose=True)
            fd_rich = kernel_alpha(hv)
            fd_div = kernel_dispersion(hv)
            
        except Exception as e:
            fd_rich = 0.0
            fd_div = 0.0

        df_cbio.at[index,'frichness'] = fd_rich
        df_cbio.at[index,'fdivergence'] = fd_div

    # create a dataset with the communities composition
    df_cbio.to_csv(os.path.join(fpath_dir, 'coms-fire-bioindex-fd.csv'))

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
    """
    Computes the hypervolume of the given data using Kernel Density Estimation (KDE).
    
    Parameters:
    - data (array-like)
      The input data for which the hypervolume is to be computed.
    - bandwidth (float, optional)
      The bandwidth for the KDE. If None, Silverman's rule of thumb is used to estimate it.
    - samples (int, optional)
      The number of samples to draw from the KDE. If None, a default value based on the data dimensions is used.
    - verbose (bool, optional)
      If True, prints additional information about the bandwidth and sample size.
    
    Returns:
    - array-like: The rescaled sampled points representing the hypervolume.
    """

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
    
    Parameters:
    - sampled_points: np.array, shape (n_samples, n_features)
      The points sampled from the KDE representing the hypervolume.
    
    Returns:
    - float: Estimated functional richness (volume of the hypervolume).
    """
    
    # Compute the convex hull of the sampled points
    hull = ConvexHull(sampled_points)
    
    return hull.volume


def kernel_dispersion(sampled_points):
    """
    Estimate functional dispersion (average distance to centroid) within the KDE-based hypervolume.
    
    Parameters:
    - sampled_points: np.array, shape (n_samples, n_features)
      The points sampled from the KDE representing the hypervolume.
      
    Returns:
    - float: Estimated functional dispersion.
    """
    
    # Compute the centroid of the sampled points
    centroid = np.mean(sampled_points, axis=0)
    
    # Compute the average distance of each point to the centroid
    distances = cdist(sampled_points, centroid.reshape(1, -1))
    
    return np.mean(distances)

# ----------------------------------------------------------------------------

# Call script from external library
if __name__ == "__main__":
    main()
    
# ----------------------------------------------------------------------------