import torch
import numpy as np
import pygame
import sys

from models.wordle_actor_critic import WordleActorCritic
from models.wordle_model_wrapper import ModelWrapper
from envs.wordle_game import WordleGame
from envs.wordle_view import WordleView
from envs.wordle_env import WordleEnv
from utils.load_list import load_word_list
from utils.word_to_onehot import word_to_onehot


answer_list = load_word_list('data/5_letter_answers_shuffled.txt')
word_list = answer_list

def demo_wordle_game(word: str):
    '''Demo a visual game of Wordle, where the model plays against a fixed word.'''
    
    # Initialize the env with a fixed word
    env = WordleEnv(word_list=word_list, answer_list=word_list)
    obs = env.reset(word=word)
    done = False
    
    # Initialize the view and relevant variables
    view = WordleView(game = env.game)
    message_printed = False  # To ensure message is printed once
    clock = pygame.time.Clock()
    
    view.draw_game()

    # Load the trained model from pth file
    device = torch.device('cpu')
    model_path = 'checkpoints/best_model.pth'
    checkpoint = torch.load(model_path)

    word_matrix = torch.stack([word_to_onehot(w) for w in word_list]).to(device)  # shape: [vocab_size, 130]

    actor_critic = WordleActorCritic().to(device)
    actor_critic.load_state_dict(checkpoint['model'])
    
    model = ModelWrapper(word_list, word_matrix, model=actor_critic, device=device)
    
    # play the game to completion
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
    # Choose a random word
    word_list = load_word_list('data/5_letter_answers_shuffled.txt')
    random_word = word_list[np.random.randint(len(word_list))]

    demo_wordle_game(random_word) # Use random word or chosen word