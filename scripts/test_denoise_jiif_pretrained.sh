set -euxo pipefail

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint ./pretrained/denoise_jiif_4.pth --name denoise_jiif_4 --model JIIF --dataset NoisyMiddlebury --scale 4 --interpolation bicubic --data_root ./data/noisy_depth/middlebury --batched_eval --report_per_image
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint ./pretrained/denoise_jiif_8.pth --name denoise_jiif_8 --model JIIF --dataset NoisyMiddlebury --scale 8 --interpolation bicubic --data_root ./data/noisy_depth/middlebury --batched_eval --report_per_image
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint ./pretrained/denoise_jiif_16.pth --name denoise_jiif_16 --model JIIF --dataset NoisyMiddlebury --scale 16 --interpolation bicubic --data_root ./data/noisy_depth/middlebury --batched_eval --report_per_image

