from envs.wordle_env import WordleEnv
from envs.wordle_game import WordleGame
from models.wordle_actor_critic import WordleActorCritic
from models.wordle_model_wrapper import ModelWrapper
import pygame
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.load_list import load_word_list
from utils.word_to_onehot import word_to_onehot

class WordleView:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 150, 0)
    YELLOW = (200, 200, 0)
    GRAY = (100, 100, 100)
    SCREEN_WIDTH = 400
    SCREEN_HEIGHT = 500
    FONT_SIZE = 48
    TILE_SIZE = 60
    PADDING = 10

    def __init__(self):
        pygame.init()
        self.FONT = pygame.font.Font(None, self.FONT_SIZE)
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption('Wordle')
        self.clock = pygame.time.Clock()
        self.game = self.initialize_game()
        self.message_printed = False

    def color_to_rgb(self, colors, j):
        if colors[j] == 'green':
            return self.GREEN
        elif colors[j] == 'yellow':
            return self.YELLOW
        else:
            return self.GRAY

    def calc_tile_posn(self, z, z_2) -> int:
        return z + z_2 * (self.TILE_SIZE + self.PADDING)

    def draw_game(self):
        self.screen.fill(self.WHITE)
        start_x = 30
        start_y = 30

        for i, (guess, colors) in enumerate(self.game.get_feedback()):
            for j, letter in enumerate(guess):
                color = self.color_to_rgb(colors, j)
                x = self.calc_tile_posn(start_x, j)
                y = self.calc_tile_posn(start_y, i)
                pygame.draw.rect(self.screen, color, (x, y, self.TILE_SIZE, self.TILE_SIZE))
                text_surface = self.FONT.render(letter.upper(), True, self.WHITE)
                text_rect = text_surface.get_rect(center=(x + self.TILE_SIZE // 2, y + self.TILE_SIZE // 2))
                self.screen.blit(text_surface, text_rect)

        if not self.game.is_game_over():
            current_row = len(self.game.get_feedback())
            for j, letter in enumerate(self.game.current_guess):
                x = self.calc_tile_posn(start_x, j)
                y = self.calc_tile_posn(start_y, current_row)
                pygame.draw.rect(self.screen, self.BLACK, (x, y, self.TILE_SIZE, self.TILE_SIZE), width=2)
                text_surface = self.FONT.render(letter.upper(), True, self.BLACK)
                text_rect = text_surface.get_rect(center=(x + self.TILE_SIZE // 2, y + self.TILE_SIZE // 2))
                self.screen.blit(text_surface, text_rect)

        pygame.display.flip()

    def initialize_game(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        word_list_path = os.path.join(base_dir, 'data', '5letterwords.txt')
        answer_list_path = os.path.join(base_dir, 'data', '5letteranswers.txt')
        word_list = load_word_list(word_list_path)
        answer_list = load_word_list(answer_list_path)
        return WordleGame(word_list=word_list, answer_list=answer_list, word='musty')

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN and not self.game.is_game_over():
            if event.key == pygame.K_RETURN:
                if len(self.game.current_guess) == 5:
                    try:
                        self.game.play_guess(self.game.current_guess.lower())
                        self.game.current_guess = ''
                    except ValueError as e:
                        print(e)
            elif event.key == pygame.K_BACKSPACE:
                self.game.current_guess = self.game.current_guess[:-1]
            elif pygame.K_a <= event.key <= pygame.K_z:
                if len(self.game.current_guess) < 5:
                    self.game.current_guess += event.unicode

def initialize_env(word_list: list, answer_list: list):
    '''Initialize the Wordle env with the given word and answer lists.'''
    env = WordleEnv(word_list=word_list, answer_list=answer_list)
    return env

    # if self.game.is_game_over() and not self.message_printed:
    #     if self.game.is_won:
    #         print(f'You guessed the word \'{self.game.word.upper()}\' in {self.game.num_guesses} tries!')
    #     else:
    #         print(f'Sorry, you lost! The word was: \'{self.game.word.upper()}\'')
    #     self.message_printed = True

def main(word_list, answer_list, model=None):
    # Load word lists and initialize the env
    env = initialize_env(word_list, answer_list)
    obs = env.reset(word=None)
    done = False
    running = True
    message_printed = False  # To ensure message is printed once
    clock = pygame.time.Clock()
    view = WordleView()

    while running:
        view.draw_game(env.game) # Draw current game state
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif model is None:
                view.handle_input(env.game, event)  # Only process keyboard input if model is None

            
        if model is not None and not done:
            guess = model.get_guess(obs)
            env.game.current_guess = guess
            view.draw_game(env.game)
            obs, _, done = env.step(guess)
            pygame.time.wait(1000)  # Wait 1s for visual clarity

        # Print game end messages
        if done and not message_printed:
            if env.game.is_won:
                print(f'You guessed the word \'{env.game.word.upper()}\' in {env.game.num_guesses} tries!')
            else:
                print(f'Sorry, you lost! The word was: \'{env.game.word.upper()}\'')
            message_printed = True

        clock.tick(30)
    pygame.quit()
    sys.exit()

# set device
device = torch.device("mps")

# set answer list and word list (here we set word list to answer list)
answer_list = load_word_list('data/5letteranswersshuffled.txt')[:]
word_list = answer_list

# get word embeddings (one hot for simplicity)
word_matrix = torch.stack([word_to_onehot(w) for w in word_list]).to(device)  # shape: [vocab_size, 130]
    
# initialize model
actor_critic = WordleActorCritic().to(device)

# load weights
checkpoint = torch.load("checkpoints/test_comprehensive_save/checkpoint_epoch_0.pth")
actor_critic.load_state_dict(checkpoint["model"])

# get model wrapper for easy interfacing
model = ModelWrapper(word_list, word_matrix, model=actor_critic device=device)

if __name__ == '__main__':
    main(word_list, answer_list, model=model) # set model to none for human play
