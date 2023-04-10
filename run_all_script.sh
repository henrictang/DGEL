CUDA_VISIBLE_DEVICES=0 python main.py --dataset wikipedia --embedding_dim 32 --sample_length 100 --epochs 30
CUDA_VISIBLE_DEVICES=0 python evaluate_all.py --dataset wikipedia --embedding_dim 32 --sample_length 100 --epochs 30

CUDA_VISIBLE_DEVICES=0 python main.py --dataset Foursquare --embedding_dim 32 --sample_length 200 --epochs 30
CUDA_VISIBLE_DEVICES=0 python evaluate_all.py --dataset Foursquare --embedding_dim 32 --sample_length 200 --epochs 30

CUDA_VISIBLE_DEVICES=0 python main.py --dataset reddit --embedding_dim 64 --sample_length 150 --epochs 30
CUDA_VISIBLE_DEVICES=0 python evaluate_all.py --dataset redit --embedding_dim 64 --sample_length 150 --epochs 30

