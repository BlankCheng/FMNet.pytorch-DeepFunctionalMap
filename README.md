# FMNet.pytorch
A pytorch implementation of Deep Functional Maps (FMNet).

## Introduction
This is a pytorch implementation of [Deep Functional Maps](https://arxiv.org/abs/1704.08686). Groundtruth labels of FAUST correspondence are not used. For efficiency, 2048 points are randomly sampled from 6890 points on original meshes. The results may not be bijective.

## Usage
Build shot calculator:
~~~
cd utils/shot
cmake .
make
~~~
Calculate eigenvectors, geodesic maps, shot descriptors of trained models, save in .mat format:
~~~
python preprocess.py
~~~
Train:
~~~
python train.py --dataset=faust
~~~
Test(temporarily use trained data to test, for visualization):
~~~
python test.py --dataset=faust --model_name=epoch300.pth
~~~
Visualize correspondence:
~~~
python visualize.py
~~~

## Visualization
![pair1](https://github.com/BlankCheng/FMNet.pytorch/raw/master/imgs/ScreenCapture_2020-02-17-13-23-52.png)
![pair2](https://github.com/BlankCheng/FMNet.pytorch/raw/master/imgs/ScreenCapture_2020-02-17-13-24-07.png)
![pair3](https://github.com/BlankCheng/FMNet.pytorch/raw/master/imgs/ScreenCapture_2020-02-17-13-25-17.png)

