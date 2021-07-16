set -euxo pipefail

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint ./pretrained/jiif_4.pth --name jiif_4 --model JIIF --dataset Middlebury --scale 4 --interpolation bicubic --data_root ./data/depth_enhance/01_Middlebury_Dataset
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint ./pretrained/jiif_8.pth --name jiif_8 --model JIIF --dataset Middlebury --scale 8 --interpolation bicubic --data_root ./data/depth_enhance/01_Middlebury_Dataset
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint ./pretrained/jiif_16.pth --name jiif_16 --model JIIF --dataset Middlebury --scale 16 --interpolation bicubic --data_root ./data/depth_enhance/01_Middlebury_Dataset

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint ./pretrained/jiif_4.pth --name jiif_4 --model JIIF --dataset Lu --scale 4 --interpolation bicubic --data_root ./data/depth_enhance/03_RGBD_Dataset
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint ./pretrained/jiif_8.pth --name jiif_8 --model JIIF --dataset Lu --scale 8 --interpolation bicubic --data_root ./data/depth_enhance/03_RGBD_Dataset
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint ./pretrained/jiif_16.pth --name jiif_16 --model JIIF --dataset Lu --scale 16 --interpolation bicubic --data_root ./data/depth_enhance/03_RGBD_Dataset

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint ./pretrained/jiif_4.pth --name jiif_4 --model JIIF --dataset NYU --scale 4 --interpolation bicubic --data_root ./data/nyu_labeled
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint ./pretrained/jiif_8.pth --name jiif_8 --model JIIF --dataset NYU --scale 8 --interpolation bicubic --data_root ./data/nyu_labeled
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint ./pretrained/jiif_16.pth --name jiif_16 --model JIIF --dataset NYU --scale 16 --interpolation bicubic --data_root ./data/nyu_labeled