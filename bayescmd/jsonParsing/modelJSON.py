import json
import os.path
# import re
import pprint
# import subprocess
import sys
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.join("..", ".."))))
from ..bcmdModel.bcmd_model import ModelBCMD  # noqa
from ..util import findBaseDir  # noqa
from io import StringIO  # noqa
BASEDIR = findBaseDir(os.environ['BASEDIR'])


def float_or_str(n):
    """Determine if a string can be returned as a float.

    Parameters
    ----------
    n : str
        String to convert to float if possible

    Returns
    -------
    s: float
        `n` as float, or str if not a number.

    """
    try:
        s = float(n)
    except ValueError:
        s = n
    return s


def get_model_name(fpath):
    """Return model name from file path.

    Parameters
    ----------
    fpath: str
        Path to model def file.

    Returns
    -------
    str
        Name of model, as determined by modeldef file.

    """
    fname = os.path.split(fpath)[-1]
    return fname.split('.')[0]


def json_writer(model_name, dictionary):
    """Write a python dictionary to JSON.

    Parameters
    ----------
    model_name: str
        Name of BCMD model.
    dictionary dict:
        Dict to write to file.

    Returns
    -------
    None
        Writes to JSON file in ``data/`` dir with name `model_name`.json

    """
    with open(
            os.path.join(
                os.path.dirname(__file__), 'data', '%s.json' % model_name),
            'w') as fp:
        json.dump(dictionary, fp)
    return None


def modeldefParse(fpath):
    """Process a modeldef file to extract information.

    Function reads a modeldef file and extracts default model inputs, outputs
    and parameters.

    Parameters
    ----------
    fpath: str
        Path to modeldef file.

    Returns
    -------
    model_data: dict
        Dictionary of model information. Has form::
        {
            model_name: name of model, obtained via :func:`get_model_name`
            input: :obj:`list` of :obj:`str` for each model input.
            output: :obj:`list` of :obj:`str` for each model output.
            parameters: :obj:`dict` of param_name (:obj:`str`):
            param_value(:obj:`str`)
        }

    """
    model_data = {'params': {}}

    with open(fpath) as f:
        model_data['model_name'] = get_model_name(fpath)
        for line in filter(None, (line.rstrip() for line in f)):
            li = line.lstrip()
            li = li.split()
            if li[0][:1] == '@':
                model_data.setdefault(li[0][1:],
                                      []).extend([item for item in li[1:]])
    model = ModelBCMD(model_data['model_name'])
    result = StringIO(model.get_defaults().stdout.decode())
    model_data['params'].update({
        line.strip('\n').split('\t')[0]: line.strip('\n').split('\t')[1]
        for line in result
    })
    json_writer(model_data['model_name'], model_data)

    return model_data


def getDefaultFilePath(model_name):
    """Given a model name, return the default path to the modeldef file.

    Parameters
    ----------
    model_name: str
        Name of model

    Returns
    -------
    modelPath: str
        Path to modeldef file

    """
    def_path = os.path.join(BASEDIR, 'examples')
    modelPath = None
    for root, dirs, files in os.walk(def_path, topdown=True):
        for file in files:
            if file == model_name + '.modeldef':
                modelPath = os.path.join(root, file)
                print(modelPath)
    return modelPath


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Process some integers.')
    ap.add_argument('model_file', metavar='FILE', help='the modeldef file')
    args = ap.parse_args()
    pprint.pprint(modeldefParse(args.model_file), depth=2)
