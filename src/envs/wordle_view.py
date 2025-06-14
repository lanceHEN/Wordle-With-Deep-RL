import pygame
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.load_list import load_word_list
from envs.wordle_game import WordleGame
from envs.wordle_env import WordleEnv
from models.wordle_actor_critic import WordleActorCritic
from models.wordle_model_wrapper import ModelWrapper
from utils.word_to_onehot import word_to_onehot

pygame.init()

# Constants
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
FONT = pygame.font.Font(None, FONT_SIZE)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Wordle')

def color_to_rgb(colors, j):
    '''Convert color names to RGB tuples for drawing.'''
    if colors[j] == 'green':
        return GREEN
    elif colors[j] == 'yellow':
        return YELLOW
    else:
        return GRAY
    
def calc_tile_posn(z, z_2) -> int:
    '''Calculate position of a tile.
    Args:
        z (int): Starting position.
        z_2 (int): The index of the row.
    Returns:
        int: The calculated position of the tile.
    '''
    return z + z_2 * (TILE_SIZE + PADDING)

def draw_game(game: WordleGame):
    '''Draw the current state of the Wordle game on the screen.'''
    screen.fill(WHITE) # White background

    start_x = 30
    start_y = 30

    for i, (guess, colors) in enumerate(game.get_feedback()):
        for j, letter in enumerate(guess):
            color = color_to_rgb(colors, j)
            
            # Calculate position for each tile
            x = calc_tile_posn(start_x, j)
            y = calc_tile_posn(start_y, i)
            # Draw colored tile
            pygame.draw.rect(screen, color, (x, y, TILE_SIZE, TILE_SIZE))
            
            # Draw letter in white
            text_surface = FONT.render(letter.upper(), True, WHITE)
            text_rect = text_surface.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
            screen.blit(text_surface, text_rect)

    # Draw current guess tiles
    if not game.is_game_over():
        current_row = len(game.get_feedback())
        for j, letter in enumerate(game.current_guess):
            x = calc_tile_posn(start_x, j)
            y = calc_tile_posn(start_y, current_row)

            # Draw box outline
            pygame.draw.rect(screen, BLACK, (x, y, TILE_SIZE, TILE_SIZE), width=2)
            
            # Draw letter in black
            text_surface = FONT.render(letter.upper(), True, BLACK)
            text_rect = text_surface.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
            screen.blit(text_surface, text_rect)

    pygame.display.flip()

def initialize_env(word_list: list, answer_list: list):
    '''Initialize the Wordle env with the given word and answer lists.'''
    env = WordleEnv(word_list=word_list, answer_list=answer_list)
    return env

def handle_input(game: WordleGame, event):
    '''Handle keyboard input for the game.
    Args:
        game (WordleGame): The current game instance.
        event (pygame.event.Event): The keyboard event.
    '''
    if event.type == pygame.KEYDOWN and not game.is_game_over():
        # Handle enter key to submit guess
        if event.key == pygame.K_RETURN:
            if len(game.current_guess) == 5:
                try:
                    game.play_guess(game.current_guess.lower())
                    game.current_guess = ''
                except ValueError as e:
                    print(e)
        # Handle backspace to remove last letter
        elif event.key == pygame.K_BACKSPACE:
            game.current_guess = game.current_guess[:-1]
        # Handle letter input (A-Z)
        elif pygame.K_a <= event.key <= pygame.K_z:
            if len(game.current_guess) < 5:
                game.current_guess += event.unicode

def main(word_list, answer_list, model=None):
    # Load word lists and initialize the env
    env = initialize_env(word_list, answer_list)
    obs = env.reset(word=None)
    done = False
    running = True
    message_printed = False  # To ensure message is printed once
    clock = pygame.time.Clock()

    while running:
        draw_game(env.game) # Draw current game state
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif model is None:
                handle_input(env.game, event)  # Only process keyboard input if model is None

            
        if model is not None and not done:
            guess = model.get_guess(obs)
            env.game.current_guess = guess
            draw_game(env.game)
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
model = ModelWrapper(actor_critic, word_list, word_matrix, device)

if __name__ == '__main__':
    main(word_list, answer_list, model=model) # set model to none for human play
