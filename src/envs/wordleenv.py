from typing import List, Tuple, Optional
from envs.WordleGame import WordleGame # referring WordleGame class
import math

class WordleEnv:
    '''
    The WordleEnv class wraps around WordleGame, for use by an RL agent. Its reset method initializes a new game, and
    its step method, when given a 5 letter word, plays that guess to the internal game and produces the observation, reward, and
    whether or not the game ended. The observation is a dict with three keys:
    1. 'feedback', which maps to a list of (guess, results) tuples, where guess is the guessed word and results is a
        list of lists of "green", "yellow" or "gray" feedback results
    2. 'turn_number', mapping to the current turn in the game
    3. 'valid_indices', which maps to a list of indices of words in the given word_list that do not contradict existing feedback
    The agent earns a specified win reward if they win, a lose reward if they lose, and an intermediate reward of -1 + info_gain_coef * normalized_info_gain,
    where normalized_info_gain is between 0 and 1, and represents how much information was found via the guess, rewarding more informative incorrect guesses.
    '''
    
    def __init__(self, word_list, answer_list, win_reward=20, lose_reward=-10, info_gain_coef=0.1):
        '''
        Initializes a wordle environment, with the given word list, answer list, win reward, and (normalized) information gain
        reward coefficient
        Args:
            word_list: list of valid guesses
            answer_list: list of possible secret words
            win_reward: reward for winning
            lose_reward: reward for losing
            info_gain_coef: coefficient to multiply by normalized info gain for incorrect guesses
        '''
        self.word_list = word_list
        self.answer_list = answer_list
        self.game = None
        self.candidate_words = []
        self.word_to_idx = {word: i for i, word in enumerate(word_list)}
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.info_gain_coef = info_gain_coef
    
    def reset(self, word: Optional[str] = None):
        '''
        resets the environment w/ a brand new game
        args:
            word: optional secret word for testing
        returns:
            the initial observation
        '''
        self.game = WordleGame(self.word_list, self.answer_list, word=word)
        self.candidate_words = list(self.word_list)
        return self._get_obs()
    
    def step(self, guess: str):
        '''
        Take a step in the environment to make an action by trying a guess. Rewards are +20 for winning, -10 for losing
        and -1 + 0.1*(normalized_info_gain)
        args:
            guess: 5-letter word guess
        returns:
            tuple of (obs, reward, done), where obs is a dict with three keys:
            1. 'feedback', which maps to a list of (guess, results) tuples, where guess is the guessed word and results is a
                list of lists of "green", "yellow" or "gray" feedback results
            2. 'turn_number', mapping to the current turn in the game
            3. 'valid_indices', which maps to a list of indices of words in the given word_list that do not contradict existing feedback 
        '''
        # plays given guess
        self.game.play_guess(guess)

        # gets feedback
        feedback = self.game.get_feedback()
        latest_guess, latest_result = feedback[-1]
        
        prior = len(self.candidate_words) # how many words before the guess
        
        # filter candidate_words by keeping candidates consistent w/ that feedback
        self._filter_cands(latest_guess, latest_result)
        
        posterior = len(self.candidate_words) # how many words after the guess
        
        # reward
        if self.game.is_game_over():
            if self.game.is_won:
                reward = self.win_reward  # guess is correst
            else:
                reward = self.lose_reward  # game is lost
        else:
            info_gain = math.log(prior) - math.log(posterior)
            normalized_info_gain = info_gain / math.log(prior) # between 0 and 1
            reward = normalized_info_gain*self.info_gain_coef - 1 # guess is incorrect
        
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
        candidate_indices = [self.word_to_idx[word] for word in self.candidate_words]
        
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
    
    def _simulate_fb(self, secret: str, guess: str) -> list[str]:
        '''
        simulates the feedback that would be given for a guess against secret word.
        this is useful to determine whether the given secret word has feedback matching up with actual feedback,
        and is useful for filtering candidate words.
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