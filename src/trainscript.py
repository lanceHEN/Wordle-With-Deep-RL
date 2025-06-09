from envs.WordleGame import WordleGame
from envs.wordleenv import WordleEnv
from envs.batchedenv import BatchedWordleEnv
from envs.LoadList import load_word_list
from utils.wordtoonehot import word_to_onehot
from models.letterencoder import LetterEncoder
from models.wordencoder import WordEncoder
from models.observationencoder import ObservationEncoder
from models.sharedencoder import SharedEncoder
from models.policyhead import PolicyHead
from models.valuehead import ValueHead
from models.wordleactorcritic import WordleActorCritic
from training.trajectorycollector import generate_trajectory
from training.ppotrainer import ppo_update
from training.trainloop import training_loop
from training.eval import evaluate_policy_on_all_answers
import torch
import random
from collections import deque
from torch.optim.lr_scheduler import CosineAnnealingLR

#torch.autograd.set_detect_anomaly(True)

device = torch.device("mps")

answer_list = load_word_list('../data/5letteranswersshuffled.txt')[:]
#word_list = load_word_list('../data/5letterwords.txt')
#answer_list = load_word_list('../data/5letteranswers.txt')
word_list = list(set(answer_list + load_word_list('../data/5letterwords.txt')[:200]))
#word_list = answer_list



word_matrix = torch.stack([word_to_onehot(w) for w in word_list]).to(device)  # shape: [vocab_size, 130]

letter_encoder = LetterEncoder().to(device)
observation_encoder = ObservationEncoder(letter_encoder, vocab_size=len(word_list)).to(device)
#grid, meta = oe(obs)
#print(grid)
#print(meta)
shared_encoder = SharedEncoder().to(device)
#h = se(grid.unsqueeze(0), meta.unsqueeze(0))ƒwe
#print(h)

policy_head = PolicyHead().to(device)
value_head = ValueHead().to(device)

actor_critic = WordleActorCritic(observation_encoder, shared_encoder, policy_head, value_head)

shared_params = list(observation_encoder.parameters()) + list(shared_encoder.parameters()) + list(letter_encoder.parameters())
policy_params = shared_params + list(policy_head.parameters())  # Include policy head
value_params = shared_params + list(value_head.parameters())  # Include value head

optimizer_policy = torch.optim.Adam(params=policy_params, lr=2e-4)
optimizer_value = torch.optim.Adam(params=value_params, lr=2e-4)

# Restore models

checkpoint = torch.load("checkpoints/baseline_extra_200/checkpoint_epoch_290.pth") # 100 8 40 # 1000 10 20 # full 60 (3.71) # extra 200 290 (3.68)
letter_encoder.load_state_dict(checkpoint["letter_encoder"])
observation_encoder.load_state_dict(checkpoint["observation_encoder"])
shared_encoder.load_state_dict(checkpoint["shared_encoder"])
policy_head.load_state_dict(checkpoint["policy_head"])
value_head.load_state_dict(checkpoint["value_head"])

# Restore optimizers
#optimizer_policy.load_state_dict(checkpoint["optimizer_policy"])
#optimizer_value.load_state_dict(checkpoint["optimizer_value"])


# Get last completed epoch
start_epoch = checkpoint["epoch"] + 1


# LR schedules for more fine grained descent
scheduler_policy = CosineAnnealingLR(optimizer_policy, T_max=400, eta_min=1e-4)
scheduler_value = CosineAnnealingLR(optimizer_value, T_max=400, eta_min=1e-4)

#scheduler_policy.load_state_dict(checkpoint["scheduler_policy"])
#scheduler_value.load_state_dict(checkpoint["scheduler_value"])

training_loop(BatchedWordleEnv(word_list, answer_list, batch_size=512), actor_critic, optimizer_policy, optimizer_value, word_list, answer_list, word_matrix, save_dir="checkpoints/baseline_extra_600",
              log_dir="runs/baseline_extra_600", start_epoch = 0, num_epochs=400, minibatch_size=256, clip_epsilon=0.2, ppo_epochs=4, entropy_coef=0.01, entropy_decay=0.99, eval_and_save_per=10, fifo_queue=deque(maxlen=200), fifo_threshold=5, device=device, scheduler_policy=scheduler_policy, scheduler_value=scheduler_value)
#evaluate_policy_on_all_answers(BatchedWordleEnv, word_list, answer_list, oe, se, ph, device=device)