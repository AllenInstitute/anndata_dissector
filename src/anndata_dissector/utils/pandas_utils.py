from anndata._io.specs import read_elem
import h5py


def read_df_from_h5ad(
        h5ad_path,
        df_name):
    """
    read a DataFrame ('obs' or 'var') from an h5ad file.

    Parameters
    ---------
    h5ad_path:
        path to the h5ad file
    df_name:
        either "obs" or "var" (whichever dataframe you want)

    Returns
    -------
    a pandas DataFrame
    """
    with h5py.File(h5ad_path, 'r', swmr=True) as in_file:
        df = read_elem(in_file[df_name])
    return df
