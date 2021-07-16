# data preparation

### NYU
We use a [preprocessed version](https://drive.google.com/drive/folders/1_1HpmoCsshNCMQdXhSNOq8Y-deIDcbKS?usp=sharing) provided [here](https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch#data-preparation). Just download the file and extract it to `data/nyu_labeled` to use.

The official NYU Depth V2 data can be downloaded [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). If you prefer to use the official data, you need to extract the depth and RGB images from the mat file and save them to `data/nyu_labeled/Depth/*.npy` and `data/nyu_labeled/RGB/*.jpg` respectively.



### MiddleBury & Lu

For these two datasets, we follow [Su et al. (Depth Enhancement via Low-rank Matrix Completion)](http://web.cecs.pdx.edu/~fliu/project/depth-enhance/) and use the data provided [here](http://web.cecs.pdx.edu/~fliu/project/depth-enhance/Depth_Enh.zip). Download it and extract it to `data/depth_enhance` to use.

### 

### Noisy MiddleBury

For the three images (`Art, Books, Moebius`) used in the noisy super-resolution experiment, we download the RGB images from the official [middlebury 2005 datasets site](https://vision.middlebury.edu/stereo/data/scenes2005/). For the GT depth, we follow [Park et al. (High Quality Depth Map Upsampling for 3D-TOF Cameras)](http://jaesik.info/publications/depthups/index.html) and use the data provided [here](http://jaesik.info/publications/depthups/iccv11_dataset.zip). Download the RGBs (view1) and the GT depths, then put them under `data/noisy_depth/middlebury/rgb/` and `data/noisy_depth/middlebury/gt/` respectively. Also, the file names should be modified to match each pair of RGB and GT depth.

We also provide a copy of the processed data [here](https://drive.google.com/file/d/1Bz0NcFdRzjN2CnWZlJzNSBFOBsnRZNKE/view?usp=sharing). Download it and extract to `data/noisy_depth/middlebury` to use.
