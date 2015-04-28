"""Simple script to filter a list of households with temporarily absent
persons by the pixel count value."""
import sys


def main():
    with open(sys.argv[1]) as infile:
        lines = [x.split() for x in infile.readlines() if 'missing' not in x]
        value = int(sys.argv[2])
        filtered = (x for x in lines if 'error' in x[1] or int(x[1]) > value)
        for line in filtered:
            print " ".join(line)


main()
