# FasterVideo: efficient joint detection and tracking
This repository is based on Detectron2 from FAIR 
https://github.com/facebookresearch/detectron2

This code performs joint detection and tracking for object tracking tasks.
Using Faster R-CNN and an additional Embeddings head (trained using triplet loss) the method is trained and tested on several datasets and benchmarks (KITTI, MOT17, MOT20)

### Requirements
- Linux or macOS
- Python >= 3.6 [preferably a conda environment]
- PyTorch 1.7
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- pycocotools: `pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
- GCC >= 4.9
- Motmetrics: `pip install motmetrics`
- Jupyter notebook to run the notebooks

### Installation
Running the following command `python -m pip install -e fastervideo`
### Usage:
See the notebooks for details
### Data
You need to download the datasets and store them in the datasets folder under the correct subfolder maintaining the default hierarchy.
#### KITTI: http://www.cvlibs.net/download.php?file=data_object_image_2.zip
#### MOT17: https://motchallenge.net/data/MOT17.zip
#### MOT20: https://motchallenge.net/data/MOT20.zip
### Trained Model Weights
weights of trained models can be found on this [link](https://unigeit-my.sharepoint.com/:u:/g/personal/s4554705_studenti_unige_it/EQBXBXsLrINHljE7W1oipLwBZPPFaB7J5RSPjjYusuKYUA?e=8ZWi1n)
## Results:

### KITTI
|Method|MOTA|MOTP|P|R|IDs|FPS|
|---|---|---|---|---|----|----|
|FasterVideo|79.3|78.6|94.6|87.5|287|8.8|
|Tracktor++|80.2|82.1|97.9|84.4|68|2.8|
|MOTBP|84.2|85.7|98|90.5|293|1.6*|
|TuSimple|86.6|84|97.9|88.8|468|3.3*|
|SORT|54.2|77.57|92.87|60.80|1|454*|


### MOT17
|Method|IDF1|MOTA|MOTP|P|R|IDs|FPS|
|---|---|---|---|---|----|----|----|
|FasterVideo|49.9|45.1|77|88.3|58.1|5589|9|
|Tracktor++|52.3|53.9|78.9|96.2|54.9|2152|1.8|
|SORT|43.1|39.8|77.8|90.7|49|4852|143*|

### MOT20
|Method|IDF1|MOTA|MOTP|P|R|IDs|FPS|
|---|---|---|---|---|----|----|----|
|FasterVideo|44.7|39.1|76.2|92.5|49.5|4171|0.8|
|Tracktor++|50.8|52.1|76.8|84.7|62.7|2751|0.2*|
|SORT|42.7|45.1|78.5|90.2|48.8|4470|57.3*|
