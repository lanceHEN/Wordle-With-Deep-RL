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

#torch.autograd.set_detect_anomaly(True)

device = torch.device("mps")

def load_word_list(path):
    with open(path, "r") as f:
        words = [line.strip() for line in f if line.strip()]
    return words

word_list = load_word_list('../data/5letterwords.txt')
answer_list = load_word_list('../data/5letterwords.txt')

#env = WordleEnv(word_list=word_list, answer_list=answer_list)

#env.reset()
#obs, reward, done = env.step("slate")

#valid_indices = obs["valid_indices"]
#print(valid_indices)

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

optimizer_policy = torch.optim.Adam(params=policy_params, lr=1e-3)
optimizer_value = torch.optim.Adam(params=value_params, lr=1e-3)

training_loop(BatchedWordleEnv(word_list, answer_list, batch_size=16), le, oe, se, ph, vh, optimizer_policy, optimizer_value, word_list, answer_list, save_dir="checkpoints/baseline", device=device)
#evaluate_policy_on_all_answers(BatchedWordleEnv, word_list, answer_list, oe, se, ph, device=device)