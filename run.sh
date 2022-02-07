python train.py --model lstm --batch-size 64 --max-length 100 --wandb --max_epochs 10 --gpus 1
python train.py --model transformer --batch-size 64 --max-length 100 --wandb --max_epochs 10 --gpus 1
python train.py --model switch --batch-size 64 --max-length 100 --wandb --max_epochs 10 --gpus 1
