import torch
from models.wordle_actor_critic import WordleActorCritic

# A simple user-friendly model wrapper around a WordleActorCritic, (optionally) taking in WordleActorCritic,
# guess list and word encodings and producing a guess word whenever given a particular
# observation. This is useful for actually playing the game in an interpretable manner (e.g. interacting with WordleView).
class ModelWrapper:
    
    # A model does not need to be specified on construction - if not specified, it is created with default params.
    def __init__(self, guess_list, word_encodings, model=None, device='cpu'):
        if model is None:
            self.model = WordleActorCritic().to(device)
        else:
            self.model = model.to(device)
        self.model.eval()
        self.guess_list = guess_list
        self.word_encodings = word_encodings.to(device)
        self.device = device

    # Given an observation, produces the model's 5-letter guess for it.
    def get_guess(self, obs):
        # Prepare observation in batch form
        obs_batch = [obs]

        # get logits
        with torch.no_grad():
            logits, _ = self.model(obs_batch, self.word_encodings)

        # get the guess - largest logit <-> largest probability
        best_idx = torch.argmax(logits[0]).item()
        return self.guess_list[best_idx]
