OPTIMIZER=$1
DEVICES=0
CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
    -d "../data/or-50-10-7-UC-10.cnf" -l\
    --optimizer "mse"\
    --lr 10e0 \
    --problem_type "cnf" \
    --num_steps 10 \
    --batch_size 7777 \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"

CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
    -d "../data/or-60-20-10-UC-10.cnf" -l\
    --optimizer "mse"\
    --lr 10e0 \
    --problem_type "cnf" \
    --num_steps 5 \
    --batch_size 100 \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"

CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
    -d "../data/or-70-5-5-UC-10.cnf" -l\
    --optimizer "mse"\
    --lr 10e0 \
    --problem_type "cnf" \
    --num_steps 5 \
    --batch_size 100 \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"

CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
    -d "../data/or-100-20-8-UC-10.cnf" -l\
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
#     -d "../data/or-50-10-7-UC-10.cnf" -l\
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
#     -d "../data/or-50-10-7-UC-20.cnf" -l\
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
#     -d "../data/or-50-10-7-UC-30.cnf" -l\
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
#     -d "../data/or-50-10-7-UC-40.cnf" -l\
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
#     -d "../data/or-60-20-10-UC-10.cnf" -l\
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
#     -d "../data/or-60-20-10-UC-20.cnf" -l\
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
#     -d "../data/or-60-20-10-UC-30.cnf" -l\
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
#     -d "../data/or-60-20-10-UC-40.cnf" -l\
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
#     -d "../data/or-70-5-5-UC-10.cnf" -l\
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
#     -d "../data/or-70-5-5-UC-20.cnf" -l\
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
#     -d "../data/or-70-5-5-UC-30.cnf" -l\
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
#     -d "../data/or-70-5-5-UC-40.cnf" -l\
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
#     -d "../data/or-100-20-8-UC-10.cnf" -l\
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
#     -d "../data/or-100-20-8-UC-20.cnf" -l\
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
#     -d "../data/or-100-20-8-UC-30.cnf" -l\
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
#     -d "../data/or-100-20-8-UC-40.cnf" -l\
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
#     -d "../data/or-100-20-8-UC-50.cnf" -l\
#     --optimizer "mse"\
#     --lr 10e0 \
#     --problem_type "cnf" \
#     --num_steps 5 \
#     --batch_size 100 \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "debug" \
#     --wandb_tags "seed=0"

