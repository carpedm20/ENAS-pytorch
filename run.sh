#!/bin/sh
# Entropy decrease fast but maintaining compared to other --ema=0.95
#python main.py --ema_baseline_decay=0.9 --shared_initial_step=150 --reward_c=800 &
#sleep 2

# BAD # live until 5k entropy was high so shared_loss was unstable
#python main.py --ema_baseline_decay=0.9 --shared_initial_step=150 --reward_c=80 &
#sleep 2

# BAD # Entropy decrease fast and increase continuously
#python main.py --ema_baseline_decay=0.95 --shared_initial_step=150 --reward_c=800 &
#sleep 2

# BAD explode but alive longer then upper reward_c=800 one
# Entropy decrease fast and increase continuously
#python main.py --ema_baseline_decay=0.95 --shared_initial_step=150 --reward_c=80 &
#sleep 2

#python main.py --ema_baseline_decay=0.9 --shared_initial_step=150 --reward_c=80 --ppl_square=True &
sleep 2

#python main.py --ema_baseline_decay=0.92 --shared_initial_step=150 --reward_c=80 --ppl_square=True &
sleep 2

python main.py --ema_baseline_decay=0.95 --shared_initial_step=150 --reward_c=80 --ppl_square=True &
sleep 2
