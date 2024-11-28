OPTIMIZER=$1
DEVICES=0
CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
    -d "../data/90-10-1-q.cnf" -l\
    --optimizer "mse"\
    --lr 10e0 \
    --problem_type "cnf" \
    --num_steps 5 \
    --batch_size 100 \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/90-10-2-q.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 100 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/90-10-3-q.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 100 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/90-10-4-q.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 100 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/90-10-5-q.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 100 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/90-10-6-q.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 100 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/90-10-7-q.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 100 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/90-10-8-q.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 100 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/90-10-9-q.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 100 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
    -d "../data/90-10-10-q.cnf" -l\
    --optimizer "mse"\
    --lr 10e0 \
    --problem_type "cnf" \
    --num_steps 5 \
    --batch_size 100 \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"