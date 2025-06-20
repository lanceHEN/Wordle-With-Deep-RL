# For testing WorldleGame class
import unittest
from envs.wordle_game import WordleGame
from utils.load_list import load_word_list

GUESS_LIST = load_word_list('../data/5letterwords.txt')
ANSWER_LIST = load_word_list('../data/5letteranswers.txt')

class TestWordleGame(unittest.TestCase):

    def test_initialization(self):
        '''Test the initialization of the WordleGame class.'''
        self.game = WordleGame(GUESS_LIST, ANSWER_LIST, word='apple')
        self.assertEqual(len(self.game.guess_list), len(GUESS_LIST))
        self.assertEqual(len(self.game.answer_list), len(ANSWER_LIST))
        self.assertEqual(self.game.word, 'apple')
        self.assertEqual(self.game.num_guesses, 0)
        self.assertFalse(self.game.is_won)

    def test_play_guess_valid_and_win(self):
        '''Test playing a valid guess.'''
        self.game = WordleGame(GUESS_LIST, ANSWER_LIST, word='apple')
        self.game.play_guess('apple')
        self.assertEqual(self.game.current_guess, 'apple')
        self.assertTrue(self.game.is_won)
        self.assertEqual(self.game.get_feedback(), [('apple', ['green', 'green', 'green', 'green', 'green'])])
        self.assertEqual(self.game.num_guesses, 1)

    def test_play_guess_valid_yellows(self):
        '''Test playing a valid guess with yellows.'''
        self.game = WordleGame(GUESS_LIST, ANSWER_LIST, word='apple')
        self.game.play_guess('peach')
        self.assertEqual(self.game.current_guess, 'peach')
        self.assertEqual(self.game.num_guesses, 1)
        self.assertFalse(self.game.is_won)
        feedback = self.game.get_feedback()
        guess, colors = feedback[0]
        self.assertEqual(guess, 'peach')
        self.assertEqual(colors, ['yellow', 'yellow', 'yellow', 'gray', 'gray'])

    def test_play_guess_valid_double_letter(self):
        '''Test playing a valid guess with double letters.'''
        self.game = WordleGame(GUESS_LIST, ANSWER_LIST, word='plead')
        self.game.play_guess('apple')
        self.assertEqual(self.game.current_guess, 'apple')
        self.assertEqual(self.game.num_guesses, 1)
        self.assertFalse(self.game.is_won)
        feedback = self.game.get_feedback()
        guess, colors = feedback[0]
        self.assertEqual(guess, 'apple')
        self.assertEqual(colors, ['yellow', 'yellow', 'gray', 'yellow', 'yellow'])

    def test_play_guess_valid_grays(self):
        '''Test playing a valid guess with grays.'''
        self.game = WordleGame(GUESS_LIST, ANSWER_LIST, word='apple')
        self.game.play_guess('botch')
        self.assertEqual(self.game.current_guess, 'botch')
        self.assertEqual(self.game.num_guesses, 1)
        self.assertFalse(self.game.is_won)
        feedback = self.game.get_feedback()
        guess, colors = feedback[0]
        self.assertEqual(guess, 'botch')
        self.assertEqual(colors, ['gray', 'gray', 'gray', 'gray', 'gray'])

    def test_play_guess_not_in_guess_list(self):
        '''Test playing a guess that is not in the guess list.'''
        self.game = WordleGame(GUESS_LIST, ANSWER_LIST, word='apple')
        with self.assertRaises(ValueError):
            self.game.play_guess('xyzzy')

    def test_is_game_over_won(self):
        '''Test if the game is over when won.'''
        self.game = WordleGame(GUESS_LIST, ANSWER_LIST, word='apple')
        self.game.play_guess('apple')
        self.assertTrue(self.game.is_game_over())

    def test_is_game_over_max_guesses_reached(self):
        '''Test if the game is over when max guesses are reached.'''
        self.game = WordleGame(GUESS_LIST, ANSWER_LIST, word='grape')
        for _ in range(6):
            self.game.play_guess('apple')  # This will not change the state after the first guess
        self.assertTrue(self.game.is_game_over())
        self.assertFalse(self.game.is_won)
        self.assertEqual(self.game.num_guesses, 6)

    def test_reset_game(self):
        '''Test resetting the game.'''
        self.game = WordleGame(GUESS_LIST, ANSWER_LIST, word='apple')
        self.game.play_guess('apple')
        self.assertTrue(self.game.is_won)
        self.game.reset_game()
        self.assertEqual(self.game.current_guess, '')
        self.assertEqual(self.game.num_guesses, 0)
        self.assertFalse(self.game.is_won)
        self.assertEqual(len(self.game.get_feedback()), 0)
        self.assertEqual(self.game.word, 'apple')

if __name__ == '__main__':
    unittest.main()
