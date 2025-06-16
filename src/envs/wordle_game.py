# Libraries
import numpy as np

# To represent a playable game of Wordle
class WordleGame:

    def __init__(self, word_list, answer_list, word=None):
        '''Initialize the WordleGame with a list of words and answers.
        Args:
            word_list (list): List of words.
            answer_list (list): List of possible answers.
            word (str, optional): The word to guess (for testing).
        '''
        self.word_list = word_list
        self.answer_list = answer_list
        self.word = word if word else self.answer_list[np.random.randint(len(self.answer_list))]
        self.feedback = [] # List of tuples (guess, colors)
        self.max_guesses = 6
        self.current_guess = ''
        self.num_guesses = 0
        self.is_won = False

    def is_game_over(self) -> bool:
        '''Is the game over (reached 6 guesses or guessed the correct word)?'''
        return self.num_guesses >= self.max_guesses or self.is_won

    def get_feedback(self) -> list:
        '''Get a copy of the feedback for the guesses so far.
        Returns:
            list: List of feedback for each guess.
        '''
        return self.feedback.copy()
    
    def play_guess(self, guess: str):
        '''Guess a 5 letter word (string of length 5).
        Args:
            guess (str): The word to guess.
        Returns: None, updates the game state.
        Raises:
            ValueError: If the guess is not a valid word or not 5 letters long.
        '''
        if len(guess) != 5 or guess not in self.word_list:
            raise ValueError('Invalid guess: must be a 5-letter word from the word list.')
        self.current_guess = guess

        if not self.is_game_over():
            self.num_guesses += 1
            colors = ['gray'] * 5
            word_letters = list(self.word)
            guess_letters = list(guess)
            # Correct letters in correct position
            for i in range(5):
                if guess_letters[i] == word_letters[i]:
                    colors[i] = 'green'
                    word_letters[i] = None # to keep track of used letters
                    guess_letters[i] = None
            #  Correct letters in wrong position
            for i in range(5):
                if guess_letters[i] is not None and guess_letters[i] in word_letters:
                    colors[i] = 'yellow'
                    word_letters[word_letters.index(guess_letters[i])] = None
            self.feedback.append((guess, colors))
            if self.current_guess == self.word:
                self.is_won = True

    def reset_game(self):
        '''Reset the game to the initial state.
        Returns: None, resets the game state.
        '''
        self.current_guess = ''
        self.num_guesses = 0
        self.is_won = False
        self.feedback = []
        self.word = self.word
