#!/bin/bash

#### train part ####

# python train.py --task=Humanoid  --seed 11  --algo=ppo --num_envs=2048 --headless
# python train.py --task=Ant  --seed 0  --algo=ppo --num_envs=2048 --headless
# python train.py --task=Cartpole --seed 0  --algo=ppo --num_envs=2048 --headless
# python train.py --task=BallBalance --seed 0  --algo=ppo --num_envs=2048 --headless
# python train.py --task=BallBalance --seed 0  --algo=ppo_collect --num_envs=2048 --headless
# python train.py --task=BallBalance --seed 0  --algo=td3_bc --num_envs=80 --headless --datatype=expert
# python train.py --task=BallBalance --seed 0  --algo=gail --num_envs=80 --headless --datatype=expert

# python train.py --task=Cartpole --seed 0  --algo=ppo_collect --num_envs=2048 --headless
# python train.py --task=Swimmer --seed 0  --algo=ppo --num_envs=2048 
# python train.py --task=Walker --seed 0  --algo=ppo --num_envs=2048 --headless
# python train.py --task=Hopper --seed 0  --algo=ppo --num_envs=2048 --headless
# python train.py --task=Hopper --seed 0  --algo=ppo_collect --num_envs=2048 --headless
# python train.py --task=Hopper --seed 0  --algo=gail --num_envs=2048 --headless --datatype=expert #(没调通)

# python train.py --task=Humanoid  --seed 11  --algo=ppo_collect --num_envs=2048 --headless
# python train.py --task=Ant  --seed 0  --algo=ppo_collect --num_envs=2048 --headless

# python train.py --task=Humanoid  --seed 11  --algo=gail --num_envs=2048 --headless --datatype=expert
# python train.py --task=Ant  --seed 0  --algo=gail --num_envs=2048  --headless --datatype=expert
# python train.py --task=Cartpole --seed 0  --algo=gail --num_envs=2048 --headless --datatype=expert


# python train.py --task=FrankaCabinet --seed 0  --algo=ppo --num_envs=2048 --headless
# python train.py --task=FrankaCabinet --seed 0  --algo=gail --num_envs=2048 --headless --datatype=expert

#######################
# python train.py --task=ShadowHandCatchOverarm --seed 21 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=15000
# python train.py --task=ShadowHandCatchAbreast --seed 21 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=10000
# python train.py --task=ShadowHandCatchOver2Underarm --seed 21 --algo=ppo_pnorm --num_envs=2048 --headless --max_iterations=10000
# python train.py --task=ShadowHandCatchUnderarm --seed 21 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=15000
# python train.py --task=ShadowHandOverOverarm --seed 21 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=10000

# python train.py --task=ShadowHandMetaML1 --seed 0 --num_envs=2048 --algo=ppo_stability --headless --max_iterations=10000

# python train.py --task=ShadowHandCatchOverarmRandom --seed 0 --num_envs 2048 --algo=ppo_lyp --headless --max_iterations=15000

# python train.py --task=ShadowHandMetaML1Random --seed 0 --algo=ppo_lyp --num_envs 256 --headless --max_iterations 15000 --rl_device="cuda:0" --sim_device="cuda:0"
# python train.py --task=ShadowHandMetaML1Random --seed 2 --algo=ppo_lyp --num_envs 256 --headless --max_iterations 15000 --rl_device="cuda:1" --sim_device="cuda:1"
# python train.py --task=ShadowHandMetaML1Random --seed 3 --algo=ppo_lyp --num_envs 256 --headless --max_iterations 15000 --rl_device="cuda:0" --sim_device="cuda:0"
# python train.py --task=ShadowHandMetaML1Random --seed 4 --algo=ppo_lyp --num_envs 256 --headless --max_iterations 15000 --rl_device="cuda:1" --sim_device="cuda:1"
# python train.py --task=ShadowHandMetaML1Random --seed 5 --algo=ppo_lyp --num_envs 256 --headless --max_iterations 15000 --rl_device="cuda:0" --sim_device="cuda:0"
# python train.py --task=ShadowHandMetaML1Random --seed 0 --algo=sac --num_envs 64 --headless --max_iterations 10000 --rl_device="cuda:1" --sim_device="cuda:1"
python train.py --task=ShadowHandMetaML1Random --seed 0 --algo=ppo_add_intrisic --num_envs 256 --headless --max_iterations 10000 --rl_device="cuda:0" --sim_device="cuda:0"
# python train.py --task=ShadowHandMetaML1Random --seed 3 --algo=sac --num_envs 64 --headless --max_iterations 10000 --rl_device="cuda:1" --sim_device="cuda:1"
# python train.py --task=ShadowHandMetaML1Random --seed 4 --algo=sac --num_envs 64 --headless --max_iterations 10000 --rl_device="cuda:1" --sim_device="cuda:1"
python train.py --task=ShadowHandMetaML1Random --seed 5 --algo=sac --num_envs 64 --headless --max_iterations 10000 --rl_device="cuda:0" --sim_device="cuda:0"

# python train.py --task=ShadowHandMetaML1Random --seed 0 --algo=ppo_lyp_constrain --num_envs 64 --headless --max_iterations 10000 --rl_device="cuda:0" --sim_device="cuda:0"

# python train.py --task=ShadowHandMetaML1Random --seed 6 --algo=trpo --num_envs 256 --headless --max_iterations 10000 --rl_device="cuda:1"
# python train.py --task=ShadowHandLiftUnderarm --seed 0 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=15000
# python train.py --task=ShadowHandBottleCap --seed 0 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=15000

# python train.py --task=ShadowHandCatchOverarm --seed 8 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=15000
# python train.py --task=ShadowHandCatchAbreast --seed 8 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=10000
# python train.py --task=ShadowHandCatchOver2Underarm --seed 8 --algo=ppo_pnorm --num_envs=2048 --headless --max_iterations=10000
# python train.py --task=ShadowHandCatchUnderarm --seed 8 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=15000
# python train.py --task=ShadowHandOverOverarm --seed 8 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=10000

# python train.py --task=ShadowHandCatchOverarm --seed 9 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=15000
# python train.py --task=ShadowHandCatchAbreast --seed 9 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=10000
# python train.py --task=ShadowHandCatchOver2Underarm --seed 9 --algo=ppo_pnorm --num_envs=2048 --headless --max_iterations=10000
# python train.py --task=ShadowHandCatchUnderarm --seed 9 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=15000
# python train.py --task=ShadowHandOverOverarm --seed 9 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=10000
#################

# python train.py --task=ShadowHandCatchOverarm --seed 0 --num_envs=2048 --algo=ppo_pnorm --headless --max_iterations=6000

# python train.py --task=ShadowHandCatchOver2Underarm --seed 0 --num_envs=2048 --algo=gail --headless --max_iterations=6000 --datatype=expert
# python train.py --task=ShadowHandCatchOverarm --seed 11 --algo=ppo_pnorm --num_envs=2048 --test --model_dir="/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/ShadowHandCatchOverarm/ppo_pnorm/ppo_pnorm_seed11/model_15000.pt"
# python train.py --task=ShadowHandCatchAbreast --seed 11 --algo=ppo_pnorm --num_envs=2048 --test --model_dir="/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/shadow_hand_catch_abreast/ppo_pnorm/ppo_pnorm_seed11/model_10000.pt"
# python train.py --task=ShadowHandCatchOver2Underarm --seed 11 --algo=ppo_pnorm --num_envs=2048 --test --model_dir="/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/shadow_hand_catch_over2underarm/ppo_pnorm/ppo_pnorm_seed11/model_10000.pt"
# python train.py --task=ShadowHandCatchUnderarm --seed 11 --algo=ppo_pnorm --num_envs=2048 --test --model_dir="/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/shadow_hand_catch_underarm/ppo_pnorm/ppo_pnorm_seed11/model_15000.pt"
# python train.py --task=ShadowHandOverOverarm --seed 11 --algo=ppo_pnorm --num_envs=2048 --test --model_dir="/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/shadow_hand_over_overarm/ppo_pnorm/ppo_pnorm_seed11/model_10000.pt"
# python train.py --task=ShadowHandMetaML1 --seed 0 --algo=ppo_pnorm --num_envs=2048 --test --model_dir="/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/ShadowHandMetaML1/ppo_pnorm/ppo_pnorm_seed0/model_1000.pt"


# python train.py --task=ShadowHandMetaML1Random --seed 0 --algo=ppo_lyp --num_envs=2 --test --model_dir="/media/xuht/sdb/lan/isaacgym_bench/isaacgymbench/logs/ShadowHandMetaML1Random/ppo_lyp/ppo_lyp_seed0_2dim_9obj/model_10000.pt"
# overarm gesture test
# python train.py --task=ShadowHandMetaML1Random --seed 0 --algo=ppo_lyp --num_envs=2 --test --model_dir="/media/xuht/sdb/lan/isaacgym_bench/isaacgymbench/logs/ShadowHandMetaML1Random/overarm/compare/ppo_lyp/ppo_lyp_seed0/model_10000.pt"


# python train.py --task=ShadowHandCatchAbreast --seed 11 --algo=ppo_pnorm --num_envs=2048 --test --model_dir="logs/shadow_hand_catch_abreast/ppo_pnorm/ppo_pnorm_seed11/model_10000.pt"
# python train.py --task=ShadowHandCatchOverarm --seed 21 --algo=ppo_pnorm --num_envs=2048 --test --model_dir="/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/ShadowHandCatchOverarm/ppo_pnorm/ppo_pnorm_seed21/model_20000.pt"
# python train.py --task=ShadowHandCatchAbreast --seed 21 --algo=ppo_pnorm --num_envs=2048 --test --model_dir="/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/shadow_hand_catch_abreast/ppo_pnorm/ppo_pnorm_seed21/model_20000.pt"
# python train.py --task=ShadowHandCatchOver2Underarm --seed 21 --algo=ppo_pnorm --num_envs=2048 --test --model_dir="/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/shadow_hand_catch_over2underarm/ppo_pnorm/ppo_pnorm_seed21/model_20000.pt"
# python train.py --task=ShadowHandCatchUnderarm --seed 21 --algo=ppo_pnorm --num_envs=2048 --test --model_dir="/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/shadow_hand_catch_underarm/ppo_pnorm/ppo_pnorm_seed21/model_20000.pt"
# python train.py --task=ShadowHandOverOverarm --seed 21 --algo=ppo_pnorm --num_envs=2048 --test --model_dir="/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/shadow_hand_over_overarm/ppo_pnorm/ppo_pnorm_seed21/model_20000.pt"


# python train.py --task=ShadowHandMetaML1Random --seed 3 --algo=ppo_lyp --num_envs=2 --test --model_dir="/media/xuht/sdb/lan/isaacgym_bench/isaacgymbench/logs/ShadowHandMetaML1Random/abreast/compare/ppo_lyp/ppo_lyp_seed3/model_10000.pt"
#### plot part ####
# python utils/logger/plotter.py --root-dir='logs/ShadowHandCatchOverarm' --output-path='/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/ShadowHandCatchOverarm/ppo_pnorm.png' --shaded-std 
# python utils/logger/plotter.py --root-dir='logs/ShadowHandCatchOverarm' --output-path='/home/shengjie/RL/isaacgym_bench/isaacgymbench/logs/ShadowHandCatchOverarm/ppo_pnorm.png' --shaded-std --smooth 30 --legend-pattern="ppo_pnorm"



# python train.py --task=ShadowHandCatchOverarm --seed 31 --num_envs=1 --algo=ppo_pnorm --max_iterations=20000 --minibatch_size=1 --headless 

