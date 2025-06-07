from envs.WordleGame import WordleGame
from envs.wordleenv import WordleEnv
from envs.batchedenv import BatchedWordleEnv
from models.letterencoder import LetterEncoder
from models.wordencoder import WordEncoder
from models.observationencoder import ObservationEncoder
from models.sharedencoder import SharedEncoder
from models.policynet import PolicyHead
from models.valuenet import ValueHead
from training.trajectorycollector import generate_trajectory
from training.ppotrainer import ppo_update
from training.trainloop import training_loop
from training.eval import evaluate_policy_on_all_answers
import torch
import random

#torch.autograd.set_detect_anomaly(True)

device = torch.device("mps")

def load_word_list(path):
    with open(path, "r") as f:
        words = [line.strip() for line in f if line.strip()]
    return words

#word_list = random.sample(load_word_list('../data/5letteranswers.txt'), 1000)
answer_list = load_word_list('../data/5letteranswers.txt')
#answer_list = word_list
env = WordleEnv(word_list=word_list, answer_list=answer_list)

# Shape: [num_words, 5 * 26]
def word_to_onehot(word):
    onehot = torch.zeros(5, 26)
    for i, c in enumerate(word):
        onehot[i, ord(c) - ord('a')] = 1.0
    return onehot.flatten()

word_matrix = torch.stack([word_to_onehot(w) for w in word_list]).to(device)  # shape: [vocab_size, 130]


#env.reset()
#obs, reward, done = env.step("slate")

#valid_indices = obs["valid_indices"]
#print(len(valid_indices))

#obs, reward, done = env.step("droid")

#valid_indices = obs["valid_indices"]
#print(len(valid_indices))

le = LetterEncoder().to(device)
oe = ObservationEncoder(le).to(device)
#grid, meta = oe(obs)
#print(grid)
#print(meta)
se = SharedEncoder().to(device)
#h = se(grid.unsqueeze(0), meta.unsqueeze(0))ƒwe
#print(h)

ph = PolicyHead().to(device)
vh = ValueHead().to(device)

#logits = ph(h, [valid_indices], word_embeddings)
#value = vh(h)

#print(value)

shared_params = list(oe.parameters()) + list(se.parameters()) + list(le.parameters())
policy_params = shared_params + list(ph.parameters())  # Include policy head
value_params = shared_params + list(vh.parameters())  # Include value head

optimizer_policy = torch.optim.Adam(params=policy_params, lr=1e-4)
optimizer_value = torch.optim.Adam(params=value_params, lr=1e-4)

# Restore models
"""
checkpoint = torch.load("checkpoints/baseline/checkpoint_epoch_60.pth")
le.load_state_dict(checkpoint["letter_encoder"])
oe.load_state_dict(checkpoint["observation_encoder"])
se.load_state_dict(checkpoint["shared_encoder"])
ph.load_state_dict(checkpoint["policy_head"])
vh.load_state_dict(checkpoint["value_head"])

# Restore optimizers
optimizer_policy.load_state_dict(checkpoint["optimizer_policy"])
optimizer_value.load_state_dict(checkpoint["optimizer_value"])

# Get last completed epoch
start_epoch = checkpoint["epoch"] + 1
"""
training_loop(BatchedWordleEnv(word_list, answer_list, batch_size=384), le, oe, se, ph, vh, optimizer_policy, optimizer_value, word_list, answer_list, word_matrix, save_dir="checkpoints/baseline_1000_2", log_dir="runs/baseline_1000_2", start_epoch = 0, num_epochs=400, eval_and_save_per=10, device=device)
#evaluate_policy_on_all_answers(BatchedWordleEnv, word_list, answer_list, oe, se, ph, device=device)