from typing import List, Optional
from envs.wordle_game import WordleGame # referring WordleGame class

class WordleEnv:
    '''
    The WordleEnv class wraps around WordleGame, for use by an RL agent. Its reset method initializes a new game, and
    its step method, when given a 5-letter word, plays that guess to the internal game and produces the observation, reward, and
    whether or not the game ended. The observation is a dict with three keys:
    1. 'feedback', which maps to a list of (guess, results) tuples, where guess is the guessed word and results is a
        list of lists of "green", "yellow", or "gray" feedback results
    2. 'turn_number', mapping to the current turn in the game
    3. 'valid_indices', which maps to a list of indices of words in the given guess_list that do not contradict existing feedback
    The agent earns a specified win reward if they win, a lose reward if they lose, and an intermediate reward of -1 otherwise
    Essentially, the goal is for the learning agent to maximize return by solving Wordle in as few guesses as possible.
    '''
    
    def __init__(self, guess_list, answer_list, win_reward=20, lose_reward=-10):
        '''
        Initializes a Wordle RL environment, with the given guess list, answer list, and win reward
        Args:
            guess_list: list of valid guesses
            answer_list: list of possible secret words/a smaller pool of words
            win_reward: positive reward delivered for winning
            lose_reward: negative reward delivered for losing
        '''
        self.guess_list = guess_list
        self.answer_list = answer_list
        self.game = None
        self.candidate_words = [] # list of answer words consistent with feedback
        self.word_to_idx = {word: i for i, word in enumerate(guess_list)} # translates a word to an integer index, useful for getting candidate indices
        self.win_reward = win_reward
        self.lose_reward = lose_reward

    def reset(self, word: Optional[str] = None):
        '''
        Resets the environment with a brand new game by starting a new episode
        Args:
            word: optional secret word
        Returns:
            The initial observation
        '''
        self.game = WordleGame(self.guess_list, self.answer_list, word=word)
        self.candidate_words = list(self.guess_list) # at t = 0 every word is a viable candidate
        return self._get_obs()
    
    def step(self, guess: str):
        '''
        Takes a step in the environment to make an action by trying a guess. 
        Rewards are the given positive win reward for winning, the given negative lose reward for losing, and -1 otherwise.
        Args:
            guess: 5-letter word guess
        Returns:
            A transition tuple of (obs, reward, done), where obs is a dict with three keys:
            1. 'feedback', which maps to a list of (guess, results) tuples, where guess is the guessed word and results is a
                list of lists of "green", "yellow", or "gray" feedback results
            2. 'turn_number', mapping to the current turn in the game
            3. 'valid_indices', which maps to a list of indices of words in the given guess_list that do not contradict existing feedback 
        '''
        # Plays given guess
        self.game.play_guess(guess)

        # Fetches feedback
        feedback = self.game.get_feedback()
        latest_guess, latest_result = feedback[-1] # The newest feedback tuple corresponds to a guess
        
        # Filters candidate_words by keeping candidates consistent with the feedback
        self._filter_cands(latest_guess, latest_result)
        
        # Computes reward signal
        if self.game.is_game_over():
            if self.game.is_won:
                reward = self.win_reward  # If guess is correct
            else:
                reward = self.lose_reward  # Game is lost
        else:
            reward = -1 # If guess is incorrect --> a time penalty pushes the agent to shorter solutions
        
        # Returns the observation and packages the output
        obs = self._get_obs()
        done = self.game.is_game_over()
        
        return obs, reward, done
    
    def _get_obs(self):
        '''
        Gets an observation of the current env state, compiled into a model-friendly structure
        Returns:
            dict:
            - feedback: list of tuples
            - turn_number: current turn number
            - candidate_indices: indices of candidate words in the original guess_list
        '''
        # If the environment hasn't been set yet, it provides an empty placeholder to do so
        if self.game is None: 
            return {
                'feedback': [],
                'turn_number': 0,
                'candidate_indices': [] }
        
        # Gets the indices of candidate_words in the original guess_list
        candidate_indices = [self.word_to_idx[word] for word in self.candidate_words]
        
        return {
            'feedback': self.game.get_feedback(),
            'turn_number': self.game.num_guesses,
            'valid_indices': candidate_indices }
    
   
    def _filter_cands(self, guess: str, result: List[str]):
        ''' 
        Filters candidate words to only those that produce a result when a guess is made
        '''
        candidates_new = []
        
        for candidate in self.candidate_words:
            if self._is_consistent(candidate, guess, result):
                candidates_new.append(candidate)
        self.candidate_words = candidates_new
    
    def _is_consistent(self, candidate: str, guess: str, result: List[str]) -> bool:
        '''
        Check if a candidate word is consistent with the result and guess (did it yield the result given guess)
        '''
        # Simulates the result if candidate_word was the secret word
        simulated_result = self._simulate_fb(candidate, guess)
        return simulated_result == result
    
    def _simulate_fb(self, secret: str, guess: str) -> list[str]:
        '''
        Simulates the feedback that would be given for a guess under an assumption that the secret word is the answer.
        This is useful to determine whether the given secret word has feedback matching up with actual feedback,
        and is also useful for filtering candidate words.
        '''
        result = ['gray'] * 5
        secret_chars = list(secret)
        guess_chars = list(guess)
        
        # Marks the exact matches in green
        for i in range(5):
            if guess_chars[i] == secret_chars[i]:
                result[i] = 'green'
                secret_chars[i] = None
                guess_chars[i] = None
        
        # Marks incorrect place matches in yellow
        for i in range(5):
            if guess_chars[i] is not None:
                if guess_chars[i] in secret_chars:
                    result[i] = 'yellow'
                    # Removes the first occurrence to handle duplicates
                    secret_chars[secret_chars.index(guess_chars[i])] = None
        
        return result
