## 说明

本项目基于pytorch复现srgan模型，并在celeba数据集下训练

## 环境

```
pip install -r requirements.txt
```

## 训练

```
python train.py
```

## 测试_生成图像

```
python generate.py --image_name image/XXX.jpg(png)
```

## 结果展示

从左往右依次是LR、超分、HR图像

![](image/000001_com.png)

## 感谢

本代码主要来源https://www.kaggle.com/code/balraj98/single-image-super-resolution-gan-srgan-pytorch 和 https://github.com/leftthomas/SRGAN ，在此基础上做了大量注释，generate.py做了大量修改
