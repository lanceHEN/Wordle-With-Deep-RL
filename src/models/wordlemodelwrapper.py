import torch

# A simple user-friendly model wrapper that takes in a WordleActorCritic, word list, and word encodings and produces a guess word whenever given a particular
# observation. This is useful for actually playing the game in an interpretable manner (e.g. interacting with WordleView)
class ModelWrapper:
    def __init__(self, model, word_list, word_matrix, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.word_list = word_list
        self.word_matrix = word_matrix.to(device)
        self.device = device

    def get_guess(self, obs):
        # Prepare observation in batch form
        obs_batch = [obs]

        with torch.no_grad():
            logits, _ = self.model(obs_batch, self.word_matrix)

        best_idx = torch.argmax(logits[0]).item()
        return self.word_list[best_idx]