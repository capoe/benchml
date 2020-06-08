collections = {}
from . import mod_basic

for register in [ mod_basic.register_all ]:
    collections.update(register())

def compile(groups, **kwargs):
    selected = [ model \
        for group in groups \
            for model in collections[group](**kwargs) ]
    return selected

