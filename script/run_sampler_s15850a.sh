OPTIMIZER=$1
DEVICES=7
CUDA_VISIBLE_DEVICES=$DEVICES python ../src/pytorch/run.py \
    -d "../data/s15850a_3_2.cnf" -l\
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
    -d "../data/s15850a_7_4.cnf" -l\
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
    -d "../data/s15850a_15_7.cnf" -l\
    --optimizer "mse"\
    --lr 10e0 \
    --problem_type "cnf" \
    --num_steps 5 \
    --batch_size 100 \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"

