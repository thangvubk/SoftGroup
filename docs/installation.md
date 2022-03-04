## Installation

1\) Environment requirements

* Python 3.x
* Pytorch 1.1 or higher
* CUDA 9.2 or higher
* gcc-5.4 or higher

Create a conda virtual environment and activate it.
```
conda create -n softgroup python=3.7
conda activate softgroup
```


2\) Clone the repository.
```
git clone https://github.com/thangvubk/SoftGroup.git --recursive
```

  
3\) Install the requirements.
```
cd SoftGroup
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

4\) Install spconv 


*  Install the spconv dependencies.
```
sudo apt-get install libboost-all-dev
sudo apt-get install libsparsehash-dev
```

* Compile the spconv library.
```
cd SoftGroup/lib/spconv
python setup.py bdist_wheel
pip install dist/{WHEEL_FILE_NAME}.whl
```


5\) Compile the external C++ and CUDA ops.
```
cd SoftGroup/lib/softgroup_ops
python setup.py build_ext develop
```

Alternative installation guide can be found in [here](https://github.com/hustvl/HAIS).
