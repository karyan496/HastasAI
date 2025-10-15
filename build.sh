#!/usr/bin/env bash
curl https://www.python.org/ftp/python/3.10.15/Python-3.10.15.tgz | tar xz
cd Python-3.10.15
./configure --prefix=/opt/python-3.10.15
make
make install
export PATH=/opt/python-3.10.15/bin:$PATH