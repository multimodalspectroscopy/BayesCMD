import time
import subprocess
import select
import argparse


def tail(filename, n):
    f = subprocess.Popen(['tail','-n',str(n),filename],\
            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    p = select.poll()
    p.register(f.stdout)

    def _byte_converter(bo):
        s = bo.decode("utf-8").strip().split(":")
        s[0] = '"'+s[0]+'"'
        s = ":".join(s)
        return s

    while True:
        if p.poll(1):
            return "{" + ",".join([_byte_converter(x) for x in f.stdout.readlines()])+"},"


if __name__=="__main__":

    parser = argparse.ArgumentParser("Pass in file to be converted to JSON and printed")
    parser.add_argument('filename', help="file to be processed")
    parser.add_argument('n_params', help="Number of parameters being optimised")

    args = parser.parse_args()

    print(tail(args.filename, args.n_params))
