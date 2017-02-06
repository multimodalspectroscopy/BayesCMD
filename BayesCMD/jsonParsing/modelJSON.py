import json
import os.path
import re


def float_or_str(n):
    try:
        s = float(n)
        return s
    except:
        return n


def get_model_name(fpath):
    fname = os.path.split(fpath)[-1]
    return fname.split('.')[0]


def modeldefParse(fpath):
    model_data = dict()

    with open(fpath) as f:
        model_data['model_name'] = get_model_name(fpath)
        for line in filter(None, (line.rstrip() for line in f)):
            if ':=' in line:
                line = line.strip('\n')
                model_data['params'][line.split(':=')[0].strip(' ')] =
                ''.join(line.split(':=')[1].strip(' '))
            li = line.lstrip()
            li = li.split()
            if li[0][:1] == '@':
                model_data.setdefault(li[0][1:],
                                      []).extend([item for item in li[1:]])
    return model_data


def param_replacement(param_dict, expression_string):
    """
    Function that will go through the dictionary of parameters, find them in
    expressions and replace with the appropriate number.

    : param param_dict: Dictionary of parameters.
    : param expression_string: String to find and replace inside.
    """
    pattern = re.compile(r'\b(' + '|'.join(param_dict.keys()) + r')\b')

    result = pattern.sub(lambda x: str(d[x.group()]), expression_string)

    return result


def json_writer(model_name, dictionary):
    with open('%s.json' % model_name, 'w') as fp:
        json.dump(sample, fp)

if __name__ == '__main__':
    import argparse
    import pprint
    ap = argparse.ArgumentParser(description='Process some integers.')
    ap.add_argument('model_file', metavar='FILE', help='the modeldef file')
    args = ap.parse_args()
    pprint.pprint(modeldefParse(args.model_file), depth=2)
