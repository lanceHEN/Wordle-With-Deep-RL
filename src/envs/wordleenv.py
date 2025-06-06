import random
import copy
from typing import List, Tuple, Optional
from envs.WordleGame import WordleGame # referring WordleGame class

class WordleEnv:
    ''' wraps around WordleGame '''
    
    def __init__(self, word_list: Optional[List[str]] = None, answer_list: Optional[List[str]] = None):
        '''
        initializes wordle environment
        Args:
            word_list: list of valid guesses (None by default)
            answer_list: list of possible secret words (None by default)
        '''
        self.word_list = word_list
        self.answer_list = answer_list
        self.game = None
        self.candidate_words = []
        self.word_to_idx = {word: i for i, word in enumerate(word_list)}
    
    def reset(self, word: Optional[str] = None):
        '''
        resets the environment w/ a brand new game
        args:
            word: optional secret word for testing
        returns:
            initial obs
        '''
        self.game = WordleGame(self.word_list, self.answer_list, word=word)
        self.candidate_words = list(self.word_list)
        return self._get_obs()
    
    def step(self, guess: str):
        '''
        take a step in the environment to make an action by trying a guess
        args:
            guess: 5-letter word guess
        returns:
            tuple of (obs, reward, done)
        '''
        # plays given guess
        self.game.play_guess(guess)

        # gets feedback
        feedback = self.game.get_feedback()
        latest_guess, latest_result = feedback[-1]
        
        # filter candidate_words by keeping candidates consistent w/ that feedback
        self._filter_cands(latest_guess, latest_result)
        
        # reward
        if self.game.is_game_over():
            if self.game.is_won:
                reward = 15  # guess is correst
            else:
                reward = -10  # game is lost
        else:
            reward = -1  # guess is incorrect
        
        # return obs and done
        obs = self._get_obs()
        done = self.game.is_game_over()
        
        return obs, reward, done
    
    # could create a mask instead (?)
    def _get_obs(self):
        '''
        get observation of the current env state
        returns:
            dict:
            - feedback: list of tuples
            - turn_number: current turn num
            - candidate_indices: indices of candidate words in original word_list
        '''
        if self.game is None:
            return {
                'feedback': [],
                'turn_number': 0,
                'candidate_indices': [] }
        
        # getting the indices of candidate words in the original word_list
        candidate_indices = [self.word_to_idx[word] for word in self.word_list ]
        
        return {
            'feedback': self.game.get_feedback(),
            'turn_number': self.game.num_guesses,
            'valid_indices': candidate_indices }
    
    # addt'l to methods, called prev
    def _filter_cands(self, guess: str, result: List[str]):
        ''' 
        filters candidate words based on feedback from a guess 
        '''
        candidates_new = []
        
        for candidate in self.candidate_words:
            if self._is_consistent(candidate, guess, result):
                candidates_new.append(candidate)
        self.candidate_words = candidates_new
    
    def _is_consistent(self, candidate: str, guess: str, result: List[str]) -> bool:
        '''
        check if a candidate word is consistent w/ result and guess
        '''
        # simulates the result if candidate word was the secret word
        simulated_result = self._simulate_fb(candidate, guess)
        return simulated_result == result
    
    # not completely sure ab this, double check
    def _simulate_fb(self, secret: str, guess: str) -> list[str]:
        '''
        simulates the feedback that would be given for a guess against secret word
        '''
        result = ['gray'] * 5
        secret_chars = list(secret)
        guess_chars = list(guess)
        
        # mark exact matches in green
        for i in range(5):
            if guess_chars[i] == secret_chars[i]:
                result[i] = 'green'
                secret_chars[i] = None
                guess_chars[i] = None
        
        # mark incorrect place matches in yellow
        for i in range(5):
            if guess_chars[i] is not None:
                if guess_chars[i] in secret_chars:
                    result[i] = 'yellow'
                    secret_chars[secret_chars.index(guess_chars[i])] = None
        
        return result
    
# random agent training loop implementation? would implement it very similar to gymnasium lab