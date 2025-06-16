from envs.batched_env import BatchedWordleEnv
from utils.load_list import load_word_list
from utils.word_to_onehot import word_to_onehot
from models.wordle_actor_critic import WordleActorCritic
from training.train_loop import training_loop
import torch
from collections import deque

# Set the device
device = torch.device("mps")

# Load the answer list and word list
# We find only allowing possible answers as guesses dramatically speeds up training
answer_list = load_word_list('data/5letteranswersshuffled.txt')[:]

word_list = answer_list
# word_list = load_word_list('data/5letterwords.txt')[:] # uncomment to use all 14,855 guess words

# Load word matrix
word_matrix = torch.stack([word_to_onehot(w) for w in word_list]).to(device)  # shape: [vocab_size, 130]

# Load model
actor_critic = WordleActorCritic().to(device)

# optimizer
optimizer = torch.optim.Adam(params=actor_critic.parameters(), lr=3e-4)

# which epoch to start at
start_epoch = 0

# where to log checkpoints and tensorboard runs
training_checkpoint_dir = "ENTER CHECKPOINT DIR FOR SAVING DURING TRAINING"
training_logging_dir = "ENTER LOGGING DIR FOR TRAINING"

# Uncomment the below lines if you want to restore a checkpoint
'''
checkpoint_dir = "ENTER CHECKPOINT DIR"

checkpoint = torch.load(checkpoint_dir)

actor_critic.load_state_dict(checkpoint["model"])

# Restore optimizer
optimizer.load_state_dict(checkpoint["optimizer"])

# Get last completed epoch
start_epoch = checkpoint["epoch"] + 1
'''


# Train the model
training_loop(BatchedWordleEnv(word_list, answer_list, batch_size=512), actor_critic, optimizer, word_list, answer_list, word_matrix, save_dir=training_checkpoint_dir,
              log_dir=training_logging_dir, start_epoch = start_epoch, num_epochs=300, minibatch_size=256, clip_epsilon=0.2, value_loss_coef=1.0, ppo_epochs=4, eval_and_save_per=10,
              fifo_queue=deque(maxlen=200), fifo_threshold=5, device=device)
