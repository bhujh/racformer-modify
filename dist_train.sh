CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node 4 train.py --config configs/racformer_r50_nuimg_704x256_f8.py