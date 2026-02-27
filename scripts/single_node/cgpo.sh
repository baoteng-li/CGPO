# CGPO
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29505 scripts/train_sd3_cur.py --config config/grpo.py:geneval_sd3

# CGPO-Fast
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29505 scripts/train_sd3_fast.py --config config/grpo.py:geneval_sd3_fast


