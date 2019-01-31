# the LGM package

by _Yuesong Shen_

This repository contains the demo code (as a python package) for the paper:

"Probabilistic Discriminative Learning with Layered Graphical Models" by
Yuesong Shen, Tao Wu, Csaba Domokos and Daniel Cremers

(TODO: arxiv link and bibtex)

If you find this code useful for your research, please consider citing the
above paper.

The code is released under GPL v3 or later. For any questions please contact:
yuesong.shen@tum.de

## setup instructions:

Tested environment: Ubuntu 16.04; Python 3.6; gcc 5.4.0.

Required dependencies: Python 3.5+ along with pip; ABI compatible C++ compiler.

- In terminal, change to current directory.

- Install dependencies: "pip install -r requirements.txt"

- Install locally the demo package: "pip install -e ."


## usage instructions:

Demo scripts are inside the folder "example/".

- "demo_lgm.py" is the demo script for LGM models

  Run "python demo_lgm.py -h" for possible arguments

  Examples:

  - Run Local model with sequential TRW and FashionMNIST. Use cuda:

    "python demo_lgm.py -m local -i seqtrw -d FashionMNIST -g"

  - run Dense model with LBP (2 inference iterations) and MNIST for 10 epochs.
    Use cpu only:

    "python demo_lgm.py -m dense -i loopy -n 2 -d MNIST -e 10"

- "demo_nn.py" is the demo script for NN baselines

  Run "python demo_nn.py -h" for possible arguments

  Examples:

  - Run Local model with FashionMNIST and sigmoid activation. Use cuda:

    "python demo_nn.py -m local -a sigmoid -d FashionMNIST -g"

  - run Dense model with relu and MNIST for 10 epochs. Use cpu only:

    "python demo_nn.py -m dense -a relu -d MNIST -e 10"
