import pygame
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # needed to access dirs at the same level, when running the file
from utils.load_list import load_word_list
from envs.wordle_game import WordleGame

# The WordleView class provides a view of the wordle game, which may be played with manually via keyboard inputs. It can either build on top of an existing game
# or initialize one on its own.
class WordleView:
    # define RGB tuples for different colors
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

    # Initializes a WordleView, either with the a supplied game if given or with a new random one.
    def __init__(self, game=None):
        pygame.init()
        self.FONT = pygame.font.Font(None, self.FONT_SIZE)
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption('Wordle')
        self.clock = pygame.time.Clock()
        self.game = game if game is not None else self.initialize_game()
        self.message_printed = False

    def color_to_rgb(self, colors, j):
        '''Convert color names to RGB tuples for drawing.'''
        if colors[j] == 'green':
            return self.GREEN
        elif colors[j] == 'yellow':
            return self.YELLOW
        else:
            return self.GRAY

    def calc_tile_posn(self, z, z_2) -> int:
        '''Calculate position of a tile.
        Args:
            z (int): Starting position.
            z_2 (int): The index of the row.
        Returns:
            int: The calculated position of the tile.
        '''
        return z + z_2 * (self.TILE_SIZE + self.PADDING)

    def draw_game(self):
        '''Draw the current state of the stored Wordle game on the screen.'''
        self.screen.fill(self.WHITE)
        start_x = 30
        start_y = 30

        for i, (guess, colors) in enumerate(self.game.get_feedback()):
            for j, letter in enumerate(guess):
                color = self.color_to_rgb(colors, j)
                
                # Calculate position for each tile
                x = self.calc_tile_posn(start_x, j)
                y = self.calc_tile_posn(start_y, i)
                
                # Draw colored tile
                pygame.draw.rect(self.screen, color, (x, y, self.TILE_SIZE, self.TILE_SIZE))
                
                # Draw letter in white
                text_surface = self.FONT.render(letter.upper(), True, self.WHITE)
                text_rect = text_surface.get_rect(center=(x + self.TILE_SIZE // 2, y + self.TILE_SIZE // 2))
                self.screen.blit(text_surface, text_rect)

        # Draw current guess tiles
        if not self.game.is_game_over():
            current_row = len(self.game.get_feedback())
            for j, letter in enumerate(self.game.current_guess):
                x = self.calc_tile_posn(start_x, j)
                y = self.calc_tile_posn(start_y, current_row)
                
                # Draw box outline
                pygame.draw.rect(self.screen, self.BLACK, (x, y, self.TILE_SIZE, self.TILE_SIZE), width=2)
                
                # Draw letter in black
                text_surface = self.FONT.render(letter.upper(), True, self.BLACK)
                text_rect = text_surface.get_rect(center=(x + self.TILE_SIZE // 2, y + self.TILE_SIZE // 2))
                self.screen.blit(text_surface, text_rect)

        pygame.display.flip()

    def initialize_game(self):
        '''Initialize a Wordle game for the view with a random word'''
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        word_list_path = os.path.join(base_dir, 'data', '5_letter_words.txt')
        answer_list_path = os.path.join(base_dir, 'data', '5_letter_answers.txt')
        word_list = load_word_list(word_list_path)
        answer_list = load_word_list(answer_list_path)
        return WordleGame(word_list=word_list, answer_list=answer_list)

    def handle_input(self, event):
        '''Handle keyboard input for the stored game, allowing guesses to be made.
        Args:
            event (pygame.event.Event): The keyboard event.
        '''
        if event.type == pygame.KEYDOWN and not self.game.is_game_over():
            # Handle enter key to submit guess
            if event.key == pygame.K_RETURN:
                if len(self.game.current_guess) == 5:
                    try:
                        self.game.play_guess(self.game.current_guess.lower())
                        self.game.current_guess = ''
                    except ValueError as e:
                        print(e)
                        
            # Handle backspace to remove last letter
            elif event.key == pygame.K_BACKSPACE:
                self.game.current_guess = self.game.current_guess[:-1]
            
            # Handle letter input (A-Z)
            elif pygame.K_a <= event.key <= pygame.K_z:
                if len(self.game.current_guess) < 5:
                    self.game.current_guess += event.unicode

# Allows manual play of Wordle, updating the view with new user inputs and stopping when the game is over.
def main():
    # Load word lists and initialize the game
    running = True
    message_printed = False  # To ensure message is printed once
    clock = pygame.time.Clock()
    view = WordleView()

    while running:
        view.draw_game() # Draw current game state
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            view.handle_input(event)

        # Print game end messages
        if view.game.is_game_over() and not message_printed:
            if view.game.is_won:
                print(f'You guessed the word \'{view.game.word.upper()}\' in {view.game.num_guesses} tries!')
            else:
                print(f'Sorry, you lost! The word was: \'{view.game.word.upper()}\'')
            message_printed = True
            

        clock.tick(30)
    pygame.quit()
    sys.exit()

# Manually play a game of Wordle.
if __name__ == '__main__':
    main()