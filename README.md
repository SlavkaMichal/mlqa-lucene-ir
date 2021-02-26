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
meta:

    module add python-3.6.2-gcc

    PYTHONPATH=/storage/brno2/home/xslavk01/.local/lib/python3.6/site-packages/
    PYTHONPATH=$PYTHONPATH:/storage/plzen1/home/xslavk01/.local/lib/python3.6/site-packages/
    export PYTHONPATH

## Usage main.py
    python main.py -h
    usage: main.py [-h] [-i INDEX] [-p PATH] -d {mlqa_dev,mlqa_test,wiki}
                   [-e {dev,test}] -l {en,es,de,multi} [-a {en,es,de,standard}]
                   [-q QUERY] [-c] [-m {dist,hit@k,qa_f1,review}]
                   [-r {reader,retriever}] [-s RAM_SIZE] [--progress_bar] [--dry]
                   [--test]

    Creating and Searching index files

    optional arguments:
      -h, --help            show this help message and exit
      -i INDEX, --index INDEX
                            Path to index file
      -p PATH, --path PATH  Path to data
      -d {mlqa_dev,mlqa_test,wiki}, --dataset {mlqa_dev,mlqa_test,wiki}
                            Dataset for indexing
      -e {dev,test}, --eval-dataset {dev,test}
                            Dataset for evaluation with answers
      -l {en,es,de,multi}, --language {en,es,de,multi}
                            Context language
      -a {en,es,de,standard}, --analyzer {en,es,de,standard}
                            Select analyzer
      -q QUERY, --query QUERY
                            Query data
      -c, --create          Create new index
      -m {dist,hit@k,qa_f1,review}, --metric {dist,hit@k,qa_f1,review}
                            Compute metric
      -r {reader,retriever}, --run {reader,retriever}
                            Run interactively
      -s RAM_SIZE, --ram_size RAM_SIZE
                            Ram size for indexing
      --progress_bar        Show progress bar while indexing TODO
      --dry                 Test run TODO
      --test                Test run TODO
