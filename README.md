# FasterVideo for efficient joint detection and tracking
This repository is based on Detectron2 from FAIR 
https://github.com/facebookresearch/detectron2

This code performs joint detection and tracking for object tracking tasks.
Using Faster R-CNN and an additional Embeddings head (trained using triplet loss) the method is trained and tested on several datasets and benchmarks (KITTI, MOT17, MOT20)

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

## Usage:
Create your python environment, requirements are usual packages, make sure to include pytorch and torchvision
Take a look at the notebooks available in the folder notebooks, which will guide through training and evaluation of the  method
### Data
You need to download the datasets and store them in the datasets folder under the correct subfolder maintaining the default hierarchy.
### Model Weights
weights of trained models can be found on this [link](https://unigeit-my.sharepoint.com/:u:/g/personal/s4554705_studenti_unige_it/EQBXBXsLrINHljE7W1oipLwBZPPFaB7J5RSPjjYusuKYUA?e=8ZWi1n)
