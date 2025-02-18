import pandas as pd
import numpy as np

def read_trcPALMTracer(file):
    """
    Parse trcPALMTracer text file.
    Input: trcPALMTracer file pathway
    Output: list of tracks at Napari format (track_id, t, z, y, x)
    
    Only read trcPALMTracer.txt
    Skip header: only read tracks data
    """
    data = pd.read_csv(file, sep="\t", skiprows=2)
    data["Track"] = [int(i) for i in data["Track"]]
    data["Plane"] = [int(i) for i in data["Plane"]]
    tracks = np.array([(data["Track"][i], data['Plane'][i], 0, data['CentroidY(px)'][i], data["CentroidX(px)"][i]) for i in range(len(data))])
    return tracks