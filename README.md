# OOD OVD
## SetUp
```bash
conda create -n ovd python=3.9 -y
conda activate ovd

pip install -r requirements.txt
```
## Directory
```bash
OVD
--data
    --coco
    --ood_coco
```
## Dataset
### COCO-O
#### DATA Install
```bash
gdown 1aBfIJN0zo_i80Hv4p7Ch7M8pRzO37qbq
```

### COCO-C
```bash
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```