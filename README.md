
# Writing kernels on NVIDIA GPUs using numba cuda libraries

In this repository, I will be developing some basic signal processing/algebraic kernels which can be executed on NVIDIA GPUs and support numba cuda libraries. The purpose of maintaining this repo is to showcase the parallel processing capability of GPGPUs (General Purpose Graphic Processing Units) in implementing complex algorithms. I will be comparing the compute/execution time of various algorithms on CPU vs GPU.


## Installation
In the following section, I will be describing the necessary tools and installations needed to set up an NVIDIA GPU laptop for writing numba cuda kernels in python.

Install Python version = 3.7.6

Install Anaconda3 version = 2020.02 from the below link:
https://repo.anaconda.com/archive/#:~:text=Anaconda3%2D2020.02%2DWindows%2Dx86_64,6b02c1c91049d29fc65be68f2443079a

After installing Anaconda, from the Anaconda Navigator, install Spyder version = 4.0.1

Download and install NVIDIA toolkit verison = 10.2 and its 2 patches from the below link:
https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork

Open the Anaconda shell and install the following packages:

cupy-cuda102
```bash
  pip install cupy-cuda102==10.0.0
```

numba
```bash
  pip install numba==0.50.1
```

llvmlite: This package aids in compilation of numba kernels. It is a very important package for Nvidia GPU installations
```bash
  conda install llvmlite==0.33.0
```







    