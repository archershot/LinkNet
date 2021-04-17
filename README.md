# linkNet implementation in Jittor
This is the Jittor implementation for LinkNet, developed based on [FuseNet](https://github.com/tum-vision/fusenet) code.
We currently release the demo code of linknet backbone, a 2D-3D multi-model segmentation network with encoder-decoder framework as fusenet.
Full version will be released soon!

## Requirements
- Ubuntu 20.04
- Python 3.8.0
- GPU + CUDA CuDNN

you may configure LinkNet requirements as:
```bash
git clone https://github.com/archershot/linkNet
cd linkNet
pip install -r requirements.txt
```

## Getting Started
We provide a demo script and a sample scan.
Please download our [pretrained weight](https://drive.google.com/file/d/113MLmm3Z02NOVQZrzaEGowRll-_lgoxc/view?usp=sharing) and move it to folder weights.
Then, you can run the results as:
```bash
unzip sample.zip
python demo.py --dataroot ./sample/scenenet/0 --use_dhac
```

## Results

* Result on SceneNet

<table>
</table>

|    SceneNet-val     |  OA  | mAcc | mIoU | Beds | Books | Ceiling | Chair | Floor | Furniture | Objects | Picture | Sofa | Table |  TV   | Wall | Window |
|---------------------|------|------|------|------|-------|---------|-------|-------|-----------|---------|---------|------|-------|-------|------|--------|
| RGB-DEPTH (FuseNet) | 82.1 | 63.4 | 46.1 | 46.2 |   -   |  79.3   |  53.7 |  75.1 |    36.9   |   54.5  |   51.0  | 22.6 |  45.6 |  28.3 | 80.5 |  25.7  |
| RGB-DHAC (LinkNet)  | 86.6 | 73.3 | 58.3 | 60.9 |   -   |  83.4   |  63.2 |  83.2 |    59.2   |   68.0  |   66.8  | 29.7 |  66.5 |  61.5 | 83.3 |  31.7  |

More detailed results will be released soon.


## Acknowledgments
Code is inspired by [FuseNet](https://github.com/tum-vision/fusenet) and [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/).
