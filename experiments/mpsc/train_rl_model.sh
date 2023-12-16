#!/bin/bash

SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_3D'

TASK='stab'
# TASK='track'

ALGO='ppo'
# ALGO='sac'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Removed the temporary data used to train the new unsafe model.
rm -r -f ./unsafe_rl_temp_data/

# Train the unsafe controller/agent.
python3 ../../safe_control_gym/experiments/execute_rl_controller.py \
    --algo ${ALGO} \
    --task ${SYS_NAME} \
    --overrides \
        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
    --output_dir ./ \
    --tag unsafe_rl_temp_data/ \
    --seed 2 \
    --kv_overrides \
        task_config.init_state=None

# Move the newly trained unsafe model.
mv ./unsafe_rl_temp_data/seed2_*/model_latest.pt ./models/rl_models/${ALGO}_model_${SYS}_${TASK}.pt

# Removed the temporary data used to train the new unsafe model.
rm -r -f ./unsafe_rl_temp_data/
