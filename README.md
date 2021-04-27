This repo contains experiment code accompanying:

> Tianze Shi, Adrian Benton, Igor Malioutov, and Ozan Irsoy. "Diversity-Aware Batch Active Learning for Dependency Parsing". NAACL. 2021.


## How to Train

We provide an example training script in `scripts/train.sh`. Please edit paths to train/dev/test conllu files accordingly. There are toggles to indicate which quality and diversity measures to use through the variables `QUALITY` and `DIVERSITY`. `COMBO_METHOD` variable indicates the use of DPPs or a naive top-K strategy.

### Example

Run:

```bash
# Download EWT
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu

sh train_ewt_amp-dpp_sample.sh
```

to train and evaluate a dependency parser on the [UD English Web Treebank](https://github.com/UniversalDependencies/UD_English-EWT) using the AMP w/ DPP active learning strategy.  It is assumed that you are training on a GPU machine with a CUDA-enabled torch installation.

## Code Structure

The main entry point for the model definition is `deppar/parser.py`. Active learning sampling strategies are implemented in `deppar/acquisition.py` for generic sample strategies, and `deppar/dpp.py` contains functions to sample batches using DPPs.

## LICENSE

All files are licensed by

> Copyright 2021 Bloomberg Finance L.P.
> [Apache 2.0 License](LICENSES/Apache_2.0_License)

unless otherwise noted in [CONTENTS](CONTENTS).
