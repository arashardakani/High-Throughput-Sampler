OPTIMIZER=$1
DEVICES=0
# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/prod-1s.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 500 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/prod-2s.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 400 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/prod-3s.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 200 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/prod-4s.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 150 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/prod-8s.cnf" -l\
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
#     -d "../data/prod-2.cnf" -l\
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
#     -d "../data/prod-4.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 500 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
    -d "../data/prod-8.cnf" -l\
    --optimizer "mse"\
    --lr 10e0 \
    --problem_type "cnf" \
    --num_steps 5 \
    --batch_size 200 \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/prod-16.cnf" -l\
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
    -d "../data/prod-20.cnf" -l\
    --optimizer "mse"\
    --lr 10e0 \
    --problem_type "cnf" \
    --num_steps 5 \
    --batch_size 75 \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/prod-24.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 75 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

# CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
#     -d "../data/prod-28.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 70 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
    -d "../data/prod-32.cnf" -l\
    --optimizer "mse"\
    --lr 10e0 \
    --problem_type "cnf" \
    --num_steps 5 \
    --batch_size 60 \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"