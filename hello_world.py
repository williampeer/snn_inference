import sys


def main(argv):
    args = [arg for arg in argv if not arg.startswith("-")]
    print('hello! args: {}'.format(args))


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
