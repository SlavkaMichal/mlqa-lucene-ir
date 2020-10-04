# mlqa-lucene-ir
Information retrieval using Lucene-BM25 for MLQA dataset

## Installation steps:

### Download pylucene
    wget https://downloads.apache.org/lucene/pylucene/pylucene-8.6.1-src.tar.gz.sha256
    wget https://downloads.apache.org/lucene/pylucene/pylucene-8.6.1-src.tar.gz
    sha256sum -c pylucene-8.6.1-src.tar.gz.sha256
    tar xvf pylucene-8.6.1-src.tar.gz

### JCC installation
    module add ant-1.9
    module add jdk-8
    module add gcc-4.8.4
    module add python-3.6.2-gcc
    cd pylucene-8.6.1/jcc/
    # only meta
    export JCC_JDK=/packages/run/jdk-8/current/
    # knot python3
    python setup.py build
    python setup.py install --user

### pylucene instalation
Add to Makefile:
meta:
    PREFIX=/storage/brno2/home/xslavk01/.local/bin
    PREFIX_PYTHON=/software/python-3.6.2/gcc
    ANT=JAVA_HOME=/packages/run/jdk-8/current /software/ant-1.9/1.9.4/bin/ant
    PYTHON=$(PREFIX_PYTHON)/bin/python
    JCC=$(PYTHON) -m jcc --shared
    NUM_FILES=10

knot:
    PREFIX=/home/xslavk01/workspace/.local/bin
    PREFIX_PYTHON=/usr
    ANT=JAVA_HOME=/usr/lib/jvm/java-13-oracle /usr/bin/ant
    PYTHON=$(PREFIX_PYTHON)/bin/python3
    JCC=$(PYTHON) -m jcc --shared
    NUM_FILES=10

## Runinng
    export PYTHONPATH=/storage/brno2/home/xslavk01/.local/lib/python3.6/site-package/
