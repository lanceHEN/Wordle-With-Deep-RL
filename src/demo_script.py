import torch
import numpy as np

from models.letter_encoder import LetterEncoder
from models.observation_encoder import ObservationEncoder
from models.shared_encoder import SharedEncoder
from models.policy_head import PolicyHead
from models.value_head import ValueHead
from envs.wordle_game import WordleGame
from envs.wordle_view import WordleView
from envs.wordle_env import WordleEnv
from utils.load_list import load_word_list
from utils.word_to_onehot import word_to_onehot

answer_list = load_word_list('../data/5letteranswersshuffled.txt')
word_list = answer_list

# To demo the model playing a game of Wordle
def demo_wordle_game(word: str):
    '''Demo a game of Wordle, where the model plays against a fixed word.'''
    
    # Initialize the game with a fixed word
    game = WordleGame(word_list=word_list, answer_list=word_list, word='apple')

    # Load the trained model from pth file
    device = torch.device('cpu')
    model_path = 'checkpoint_epoch_790.pth'
    model = torch.load(model_path)

    word_matrix = torch.stack([word_to_onehot(w) for w in word_list]).to(device)  # shape: [vocab_size, 130]

    letter_encoder = LetterEncoder().to(device)
    observation_encoder = ObservationEncoder(letter_encoder, vocab_size=len(word_list)).to(device)
    shared_encoder = SharedEncoder().to(device)
    policy_head = PolicyHead().to(device)
    value_head = ValueHead().to(device)

    letter_encoder.load_state_dict(model["letter_encoder"])
    observation_encoder.load_state_dict(model["observation_encoder"])
    shared_encoder.load_state_dict(model["shared_encoder"])
    policy_head.load_state_dict(model["policy_head"])
    value_head.load_state_dict(model["value_head"])

    # Get the model's guess
    with torch.no_grad():
        env = WordleEnv(word_list, answer_list)
        obs_list = env.reset()

        # Compute logits
        logits, _ = model(obs_list, word_matrix)
        actions = torch.argmax(logits, dim=-1).tolist()

        # Get the guessed words
        guess_words = [word_list[a] for a in actions]

    # Play the game with the model's guesses
    for guess in guess_words:
        # Check if the game is over
        if game.is_game_over():
            break
        
        # Play the guess in the game
        print(f'Guessing: {guess}')
        game.play_guess(guess)
        game.render()
    
    # Print the final result
    if game.is_won:
        print(f'You guessed the word \'{game.word.upper()}\' in {game.num_guesses} guesses!')
    else:
        print(f'Sorry, you lost! The word was: \'{game.word.upper()}\'')

if __name__ == '__main__':
    # Choose a random word
    word_list = load_word_list('../data/5letteranswersshuffled.txt')
    random_word = word_list[np.random.randint(len(word_list))]

    demo_wordle_game('apple') # Use random word or chosen word