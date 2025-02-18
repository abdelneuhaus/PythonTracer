import pandas as pd

def read_locPALMTracer(file):
    """
    Parse locPALMTracer text file.
    Input: locPALMTracer file pathway
    Output: pd.Dataframe for locs file and list of loc coordinates
    
    Only read locPALMTracer.txt
    Skip header: only read localization data
    """
    data = pd.read_csv(file, sep="\t", skiprows=2)
    data["id"] = [int(i) for i in data["id"]]
    data["Plane"] = [int(i) for i in data["Plane"]]
    data["Index"] = [int(i) for i in data["Index"]]
    coordinates = [(data['Plane'][i]-1, data['CentroidX(px)'][i], data["CentroidY(px)"][i]) for i in range(len(data))]
    return data, coordinates