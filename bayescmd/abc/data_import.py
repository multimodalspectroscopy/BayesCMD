import csv


def import_actual_data(fname):
    """
    Method to parse a csv of data into the correct format for comparing with simulated data.

    Inputs:
    ======
    :param fname: Filepath/filename of the actual data to parse
    :type fname: str

    Returns:
    =======
    :return: Dictionary of actual data. Keys are columns headers, values are lists of data.
    :rtype: dict
    """

    actual_data = {}
    with open(fname, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            for k in row.keys():
                try:
                    actual_data.setdefault(k, []).append(float(row[k]))
                except ValueError as e:
                    print(e)
                    sys.exit(1)
    return actual_data
