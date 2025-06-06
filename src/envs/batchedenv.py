import random
from envs.wordleenv import WordleEnv

# this class allows for a batch of wordle environments together, enabling quicker trajectory collection and training time
# made in part with generative AI
class BatchedWordleEnv:
    
    # given an environment class (e.g. WordleEnv), word list, answer list and batch size, produced a BatchedWordleEnv, a batched set of environments
    # with the given number of batches
    def __init__(self, word_list, answer_list, batch_size, env_class=WordleEnv):
        self.envs = [env_class(word_list, answer_list) for _ in range(batch_size)]
        self.batch_size = batch_size
        self.word_list = word_list
        self.answer_list = answer_list

        # store current states
        self.current_obs = [env.reset() for env in self.envs]
        self.dones = [False] * batch_size

    # resets across each env in the batch, and returns the observations for each
    # if starting words specified, starts each env with the given starting words
    def reset(self, starting_words):
        if self.starting_words:
            self.current_obs = [env.reset(starting_words[i]) for i, env in enumerate(self.envs)]
        else:
            self.current_obs = [env.reset() for env in self.envs]
        self.dones = [False] * self.batch_size
        return self.current_obs

    # for each action in the list, applies that on the corresponding environment, returning the corresponding observations, rewards, and dones
    def step(self, actions):
        """
        Args:
            actions: list of str (words) to play for each env
        Returns:
            next_obs_list, reward_list, done_list
        """
        next_obs = []
        rewards = []
        new_dones = []

        for i, (env, action, done) in enumerate(zip(self.envs, actions, self.dones)):
            if done:
                next_obs.append(self.current_obs[i])
                rewards.append(0.0)
                new_dones.append(True)
            else:
                obs, reward, done = env.step(action)
                self.current_obs[i] = obs
                next_obs.append(obs)
                rewards.append(reward)
                new_dones.append(done)

        self.dones = new_dones
        return next_obs, rewards, new_dones

    # determines if all the envs are done
    def all_done(self):
        return all(self.dones)
