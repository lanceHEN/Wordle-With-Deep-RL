from envs.WordleGame import WordleGame
from envs.wordleenv import WordleEnv
from envs.batchedenv import BatchedWordleEnv
from utils.LoadList import load_word_list
from utils.wordtoonehot import word_to_onehot
from models.letterencoder import LetterEncoder
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

# Set the device
device = torch.device("mps")

# Load the answer list and word list
# We find only allowing possible answers as guesses dramatically speeds up training
answer_list = load_word_list('data/5letteranswersshuffled.txt')[:]
word_list = answer_list

# Load word matrix
word_matrix = torch.stack([word_to_onehot(w) for w in word_list]).to(device)  # shape: [vocab_size, 130]

# Load model
actor_critic = WordleActorCritic().to(device)

# optimizer
optimizer = torch.optim.Adam(params=actor_critic.parameters(), lr=2e-4)

# LR scheduler for more fine grained descent
scheduler = CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-4)

# which epoch to start at
start_epoch = 0


# Uncomment the below lines if you want to restore a checkpoint
'''
checkpoint_dir = "ENTER CHECKPOINT DIR"

checkpoint = torch.load(checkpoint_dir)

actor_critic.load_state_dict(checkpoint["model"])

# Restore optimizer
optimizer.load_state_dict(checkpoint["optimizer"])

# Get last completed epoch
start_epoch = checkpoint["epoch"] + 1


# Restore scheduler
scheduler.load_state_dict(checkpoint["scheduler"])
'''

training_checkpoint_dir = "ENTER CHECKPOINT DIR FOR SAVING DURING TRAINING"
training_logging_dir = "ENTER LOGGING DIR FOR TRAINING"

# Train the model
training_loop(BatchedWordleEnv(word_list, answer_list, batch_size=512), actor_critic, optimizer, word_list, answer_list, word_matrix, save_dir=training_checkpoint_dir,
              log_dir=training_logging_dir, start_epoch = start_epoch, num_epochs=400, minibatch_size=256, clip_epsilon=0.2, value_loss_coef=1.0, ppo_epochs=4, eval_and_save_per=10,
              fifo_queue=deque(maxlen=200), fifo_threshold=5, device=device, scheduler=scheduler)
