import benchml as bml
from benchml.models import compile_and_filter

log = bml.log
log.setLevel("error")


RST_HEADER = """======
Models
======
"""


def main():
    models = compile_and_filter()
    print(RST_HEADER)
    for m in models:
        print(m.__doc__)


if __name__ == "__main__":
    main()
