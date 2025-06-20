import torch
import numpy as np
import pygame
import sys

from models.wordle_actor_critic import WordleActorCritic
from models.wordle_model_wrapper import ModelWrapper
from envs.wordle_game import WordleGame
from view.wordle_view import WordleView
from envs.wordle_env import WordleEnv
from utils.load_list import load_word_list
from utils.word_to_encoding import word_to_encoding

# Initialize the answer list and guess list.
answer_list = load_word_list('data/5_letter_answers_shuffled.txt')
guess_list = answer_list # We set the guess list to the answer list for faster training.
# guess_list = load_word_list('data/5_letter_words.txt')[:] # uncomment to use all 14,855 guess words

def demo_wordle_game(word, device, model_path):
    '''Demo a visual game of Wordle, where the model plays against a fixed word.'''
    
    # Initialize the environment with a fixed word.
    env = WordleEnv(guess_list=guess_list, answer_list=answer_list)
    obs = env.reset(word=word) # We need to record the observation for the model to know what to do.
    done = False
    
    # Initialize the view and relevant variables.
    view = WordleView(game = env.game)
    message_printed = False  # To ensure message is printed once
    clock = pygame.time.Clock()
    
    view.draw_game()

    # Initialize the word encoding matrix.
    word_encodings = torch.stack([word_to_encoding(w) for w in guess_list]).to(device)  # shape: [vocab_size, 130]
    
    # Load the trained model from pth file.
    checkpoint = torch.load(model_path)

    actor_critic = WordleActorCritic().to(device)
    actor_critic.load_state_dict(checkpoint['model'])
    
    # Note we could actually construct this without having to specify a WordleActorCritic (it would make one automatically with default params, in that case)
    model = ModelWrapper(guess_list, word_encodings, model=actor_critic, device=device)
    
    # Play the game to completion.
    while not done:
        
        guess = model.get_guess(obs)
        # update current guess in view
        env.game.current_guess = guess
        view.draw_game()
        pygame.time.wait(1000)  # Wait 1s for visual clarity
        obs, _, done = env.step(guess)
        view.draw_game()
        
        # Print game end messages
        if done and not message_printed:
            if env.game.is_won:
                print(f'Guessed the word \'{env.game.word.upper()}\' in {env.game.num_guesses} tries!')
            else:
                print(f'Loss! The word was: \'{env.game.word.upper()}\'')
            message_printed = True

        clock.tick(30)
        
    # close the game
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    # Choose a random word.
    answer_list = load_word_list('data/5_letter_answers_shuffled.txt')
    random_word = answer_list[np.random.randint(len(answer_list))]
    
    # Set the device.
    device = torch.device("cpu") # Change as needed.
    
    # Set the model path.
    model_path = "checkpoints/best_model.pth" # Change as needed.

    demo_wordle_game(random_word, device, model_path) # Use random word or chosen word.