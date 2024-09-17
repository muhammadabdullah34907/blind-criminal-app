
## Cloning the Repo

To download the code, navigate to a folder of your choosing on the Jetson.  First, make sure git and cmake are installed:

``` bash
$ sudo apt-get update
$ sudo apt-get install git cmake
```

Then clone the `jetson-inference` project:

``` bash
$ git clone https://github.com/dusty-nv/jetson-inference
$ cd jetson-inference
$ git submodule update --init
```

Remember to run the `git submodule update --init` step (or clone with the `--recursive` flag).

### Python Development Packages

The Python functionality of this project is implemented through Python extension modules that provide bindings to the native C++ code using the Python C API.  While configuring the project, the repo searches for versions of Python that have development packages installed on the system, and will then build the bindings for each version of Python that's present (e.g. Python 2.7, 3.6, 3.8).

``` bash
$ sudo apt-get install libpython3-dev python3-numpy
``` 

Then after the `sudo make install` step, the [`jetson_inference`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html) and [`jetson_utils`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.utils.html) modules should be able to be imported in Python.

### Configuring with CMake

Next, create a build directory within the project and run `cmake` to configure the build.  When `cmake` is run, a script is launched ([`CMakePreBuild.sh`](../CMakePreBuild.sh)) that will install any required dependencies and download DNN models for you.

``` bash
$ cd jetson-inference    # omit if working directory is already jetson-inference/ from above
$ mkdir build
$ cd build
$ cmake ../
```

> **note**: this command will launch the [`CMakePreBuild.sh`](../CMakePreBuild.sh) script which asks for sudo privileges while installing some prerequisite packages on the Jetson. The script also downloads pre-trained networks from web services.


## Installing PyTorch

If you are using JetPack 4.2 or newer, another tool will now run that can optionally install PyTorch on your Jetson please skip this step.
Leave the options un-selected, and it will skip the installation of PyTorch. 

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-inference/master/docs/images/pytorch-installer.jpg" width="650">

## Compiling the Project

Make sure you are still in the `jetson-inference/build` directory, created above in step #3.

Then run `make` followed by `sudo make install` to build the libraries, Python extension bindings, and code samples:

``` bash
$ cd jetson-inference/build          # omit if working directory is already build/ from above
$ make
$ sudo make install
$ sudo ldconfig
```

