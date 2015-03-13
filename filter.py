import sys


def main():
    with open(sys.argv[1]) as infile:
        lines = [x.split() for x in infile.readlines() if 'missing' not in x
                 and 'error' not in x]
        value = int(sys.argv[2])
        filtered = (x for x in lines if int(x[1]) > value)
        for line in filtered:
            print " ".join(line)

main()
