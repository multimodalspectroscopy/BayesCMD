"""Functions to import and parse real data for use in model."""
import csv
import sys


def import_actual_data(fname):
    """Parse a csv into the correct format to compare with simulated data.

    Parameters
    ----------
    fname : str
        Filepath/filename of the actual data to parse

    Returns
    -------
    actual_data : dict
        Dictionary of actual data. Keys are column headers, values are lists
        of data.

    """
    actual_data = {}
    with open(fname, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            for k in row.keys():
                try:
                    actual_data.setdefault(k.strip(), []).append(float(row[k]))
                except ValueError as e:
                    print(e)
                    sys.exit(1)
    return actual_data


def inputParse(d0, inputs):
    """Parse dictionary of actual data for use with ModelBCMD.

    Parameters
    ----------
    d0 : dict
        Dictionary created from import_actual_data

    inputs : list
        List of model input names which should all be :obj:`str` objects.

    Returns
    -------
    dict
        Dictionary of inputs as used by :class:`bayescmd.bcmd_model.ModelBCMD`

    """
    values = [l for k, l in d0.items() if k in inputs]
    return {"names": [k for k in d0.keys() if k in inputs],
            "values": list(map(list, zip(*values)))}
