import katarchive
import katfile
import scape

import matplotlib.pyplot as plt
import numpy as np


def stitch(datasets, freq, fixedLO=4e9, n_channels=512, df=1e6, smooth=False):
    """
            Combine measurements with different centre frequencies. For now all inputs must
            be sorted by increasing fLO.
            @param datasets: list of (2D) spectra for each fLO (i.e. power by frequency)
            @param fLOs: list of VLO frequencies (in Hz) for each of the elements in datasets.
            @return: consolidated_frequencies, consolidated_dataset
        """
    # First translate to RF
    dsets = []
    freqs = []
    for i in range(min([len(datasets), len(freq)])):
        dsets.append(np.flipud(datasets[i]).tolist())
        freqs.append(np.flipud(freq[i]).tolist())
    
        # Now stitch together
    d_consolidated = dsets[0]
    f_consolidated = freqs[0]
    for i in range(len(freqs)-1):
        f_currentstop = freqs[i][-1]
        f_nextstart = freqs[i+1][0]
        n_overlap = round((f_currentstop-f_nextstart)/df + 1)
        if (n_overlap < 0): # Extend current data to bridge gaps in coverage
            for f_currentstop in np.arange(f_currentstop, f_nextstart, df):
                d_consolidated.append(d_consolidated[-1])
        elif (n_overlap > 0): # Cut overlapping data
            n_a = int(n_overlap/2) # Number of channels to remove from the left side data
            if (smooth): # Find a spot close to n_a which minimizes the difference between left & right side
                diff = [abs(d_consolidated[n_a-x]-dsets[i+1][n_a+1]) for x in range(-n_a//4,n_a//4)]
                n_a = n_a - diff.index(min(diff))
            n_b = int(n_overlap - n_a) # Number of channels to remove from the right side data
            # Also average 3 values around the join point
            d_consolidated = d_consolidated[:-n_a-2] + \
                                 ((np.asarray(d_consolidated[-n_a-2:-n_a+1])+np.asarray(dsets[i+1][n_b-2:n_b+1]))/2.).tolist() + \
                              dsets[i+1][n_b+1:]
            f_consolidated = f_consolidated[:-n_a] + freqs[i+1][n_b:]
    d_consolidated = np.asarray(d_consolidated)
    f_range = np.arange(freqs[0][0], freqs[-1][-1]+df, df) # Sometimes out by 1
    if (len(f_range) != len(d_consolidated)):
        print("WARN: poor stitching")
        f_consolidated = np.asarray(f_consolidated)
    else:
        f_consolidated = f_range
          
    return f_consolidated, d_consolidated
      
