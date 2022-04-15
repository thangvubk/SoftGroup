## Installation

1\) Environment requirements

* Python 3.x
* Pytorch 1.11
* CUDA 9.2 or higher

The following installation guild suppose ``python=3.7`` ``pytorch=1.11`` and ``cuda=10.2``. You may change them according to your system.

Create a conda virtual environment and activate it.
```
conda create -n softgroup python=3.7
conda activate softgroup
```


2\) Clone the repository.
```
git clone https://github.com/thangvubk/SoftGroup.git
```


3\) Install the dependencies.
```
conda install pytorch cudatoolkit=10.2 -c pytorch
pip install spconv-cu102
pip install -r requirements.txt
```

4\) Install build requirement.

```
sudo apt-get install libsparsehash-dev
```

5\) Setup
```
python setup.py build_ext develop
```
