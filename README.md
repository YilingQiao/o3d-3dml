
## 3D machine learning module for Open3D

### Intro

This is only a temporary repo for 3D machine learning in Open3D. It will be merged into Open3D project in the future.


### Usage

a. Create a conda virtual environment and activate it.

```shell
conda create -n o3d-ml3d python=3.7 -y
conda activate o3d-ml3d
```

b. Install packages.
```shell
pip install -r requirements.txt
pip install -v -e .
```

c. Run the example scripts
```shell
python example/script_inference.py
```
