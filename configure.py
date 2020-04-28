#! /usr/bin/env python
import os
cwd = os.getcwd()
cwd = os.path.abspath(cwd)

ifs = open('benchml/BENCHMLRC.in', 'r')
ofs = open('bin/BENCHMLRC', 'w')
for ln in ifs.readlines():
    ln = ln.replace('@CMAKE_INSTALL_PREFIX@', cwd)
    ofs.write(ln)
ofs.close()
ifs.close()
