import numpy as np
import pandas as pd

def save_results_python(results, p, output_filename="results.csv"):
    """
    Save the results of the spline fitter in a CSV/txt file.
    Args:
        results (np.ndarray): results from the spline fitter
        p (dict): parameters from the spline fitter
        output_filename (str): output filename
    Returns:
        None
    """
    
    N = results.shape[0]
    results_extended = np.zeros((N, 16), dtype=results.dtype)
    results_extended[:, :12] = results

    # x, y... values times pixelsize (in nm)
    results_extended[:, [12, 14]] = results_extended[:, [1, 6]] * p["pixelsize"]
    results_extended[:, [13, 15]] = results_extended[:, [2, 7]] * p["pixelsize"]
    results = results_extended
    
    if p['isspline']:   # always isspline (raw traduction)
        columns = [
            'frame', 'x_pix', 'y_pix', 'z_nm',
            'photons', 'background',
            'crlb_x', 'crlb_y', 'crlb_z',
            'crlb_photons', 'crlb_background',
            'logLikelyhood',
            'x_nm', 'y_nm',
            'crlb_xnm', 'crlb_ynm'
        ]
        
        resultstable = pd.DataFrame(results, columns=columns)        
        resultstable.to_csv(output_filename, sep='\t', index=False)
        print(f"Results saved in {output_filename}")