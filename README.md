# Solving Wordle with Deep RL

This repository contains the codebase for a deep reinforcement learning (RL) agent that learns to solve Wordle in as few guesses as possible using Proximal Policy Optimization (PPO). The project treats Wordle as a Markov Decision Process (MDP) with different rewards for winning, losing, or making incorrect guesses, and experiments with various architectures to boost performance.

## Methodology Overview

We model Wordle as a Markov Decision Process (MDP) with:
- **States**: Feedback from previous guesses, current turn number, and valid guess indices.
- **Actions**: Valid word guesses that do not contradict known feedback.
- **Rewards**: Large positive reward for winning (we use +20), negative for losing (-10), and a small penalty for incorrect guesses (-1) to encourage efficiency.
- **Discount factor**: $\gamma = 1$, since there are no more than 6 steps.
- **Probabilities** of ending up in different states, given an initial state and action (these are uncertain to reflect the stochasticity of the game from the agent's perspective). Notably, probabilities are never needed or computed anywhere in the implementation, requiring only theoretical description. Particularly, we define the probability to be the ratio of the number of answers matching the feedback of the destination state, to the total number of answers matching the feedback of the source state.

Training is conducted using PPO, which involves a neural network to pick actions from a probability distribution over them, consisting of:
- **Observation Encoder**: Given an observed state dictionary, produces a numerical representation for neural networks consisting of:
  - A grid tensor, which stores learned encodings for letters and one-hot encodings for feedback at each position on the board.
  - A meta vector, which stores additional information including turn number and number of remaining valid guesses (normalized).
- **Shared Encoder**: Given a grid tensor and meta vector from the Observation Encoder, produces a latent vector representation for use by both the Policy and Value heads.
- **Policy Head**: Given the latent vector representation, produces a query vector which can then be multiplied with a matrix of word encodings to produce logits over each word (mask those for valid indices to $- \infty$), from which a probability distribution may be derived via softmax.
- **Value Head**: Given the latent vector representation, produces a scalar prediction of the value of the original state.

The actual PPO training scheme involves first collecting a number of episodes or trajectories from the environment, where each trajectory consists of states, actions, rewards, and other relevant values from full games. For each step in these episodes, we compute:

- **Returns**: The sum of current and future rewards from that step (undiscounted in our case, since $\gamma = 1$).
- **Advantages**: The difference between actual return and predicted state value, used to measure how much better or worse the action performed than expected.
  
Training proceeds by updating the policy and value networks using mini-batches from these trajectories across several PPO epochs. The policy network is updated by maximizing a clipped objective that prevents overly large updates, resulting in more stable learning. Meanwhile, the value network is trained to capture the actual returns as accurately as possible.

To focus learning on challenging words, we maintain a first-in, first-out (FIFO) queue storing words that took more than a fixed number of guesses to answer. Rather than generating solely random episodes, a fixed percentage of them are reserved to use words popped from the queue.

Our implementation supports training with either a feedforward network (FFN) or a convolutional neural network (CNN) (treating guesses as channels) in the Shared Encoder.

## Code Layout
<pre>
├── README.md # You are here!
├── requirements.txt # Use to install all the necessary requirements
├── checkpoints # Directory to store model checkpoints
│   └── best_model.pth # .pth file storing our best model (3.63 guesses per word, 99.5% win rate)
├── data # Directory to store different word lists
│   ├── 5_letter_answers.txt # All 2,315 Wordle answers, in alphabetical order
│   ├── 5_letter_answers_shuffled.txt # All 2,315 Wordle answers, in random order
│   └── 5_letter_words.txt # All 14,855 Wordle guesses
└── src # Directory containing the implementation of the agent, game, and environment
    ├── demo_script.py # Demonstrates a model on a visual Wordle game
    ├── envs # Directory for the wordle game and environment
    │   ├── __init__.py
    │   ├── batched_env.py # Wraps multiple environments together in a batch for faster training
    │   ├── test_wordle_game.py # Tests for WordleGame
    │   ├── wordle_env.py # Contains an environment wrapper for Wordle, for agent interaction
    │   └── wordle_game.py # Contains the implementation of Wordle, as class WordleGame
    ├── eval # Directory containing code to evaluate the model
    │   ├── __init__.py
    │   └── evaluate_policy.py # Provides code to evaluate a model's average guesses and win rate
    ├── models # Directory containing the agent's architectural implementation
    │   ├── __init__.py
    │   ├── cnn_shared_encoder.py # A CNN implementation of the Shared Encoder
    │   ├── ffn_shared_encoder.py # A FFN implementation of the Shared Encoder
    │   ├── letter_encoder.py # Learnable letter embeddings
    │   ├── obs_shared_wrapper.py # A wrapper around the Observation and Shared Encoder
    │   ├── observation_encoder.py # The Observation Encoder
    │   ├── policy_head.py # The Policy Head
    │   ├── value_head.py # The Value Head
    │   ├── wordle_actor_critic.py # A wrapper around each component of the model architecture, which produces logits and predicted values when given batched observations
    │   └── wordle_model_wrapper.py # A wrapper around a WordleActorCritic, which simply produces a human-readable guess any time it's given an observation
    ├── train_script.py # A script to train the model
    ├── training # A directory for PPO training implementation
    │   ├── __init__.py
    │   ├── ppo_trainer.py # Code for the PPO update algorithm
    │   ├── train_loop.py # Code for the overall training loop
    │   └── trajectory_collector.py Code for collecting trajectories for PPO training
    ├── utils # A directory for random utility functions
    │   ├── __init__.py
    │   ├── load_list.py # Provides a function to load a word file into a list of strings
    │   └── word_to_encoding.py # Provides a function to produce an encoding for a word as a concatenation of its one-hot letter encodings
    └── view # A directory containing the playable view of the Wordle game
        ├── __init__.py
        └── wordle_view.py # A visual, playable Wordle game, which a human can interact with by running the main function
</pre>

## Setup Instructions

### Requirements (found in `requirements.txt`)
- Python 3.8+
- PyTorch
- NumPy
- Pygame (for visualization)
- tqdm
- tensorboard

We recommend using [conda](https://docs.conda.io/en/latest/) to manage dependencies:
```bash
# Step 1: Create a new conda environment
conda create -n wordle-env python=3.11 -y

# Step 2: Activate the environment
conda activate wordle-env

# Step 3: Install dependencies from the requirements.txt file
pip install -r requirements.txt
```

## Quick Start

First, we need to initialize the answer list and guess list:
```python
answer_list = load_word_list('data/5_letter_answers_shuffled.txt')
guess_list = answer_list # For much faster training, just set to answer list
# guess_list = load_word_list('data/5_letter_words.txt')[:] # Or uncomment to use all 14,855 guess words.
```

Next, we need to initialize an environment to work with:
```python
env = WordleEnv(guess_list=guess_list, answer_list=answer_list)
# Reset the environment, optionally specifying the answer word
answer = None # If set to None, a random word is chosen by the environment. Otherwise, the environment will use the given word.
obs = env.reset(word=answer) # We need to record the observation for the model to know what to do.
```

Next, we need to set the device:
```python
device = torch.device("cpu") # Change as needed.
```

Next, we need to initialize the word encoding matrix:
```python
word_encodings = torch.stack([word_to_encoding(w) for w in guess_list]).to(device)  # shape: [vocab_size, 130]
```

Next, we neet to initialize the model. You could initialize each component individually and pass them into the ```WordleActorCritic``` constructor, but for simplicity here we can make the object without specifying them:
```python
actor_critic = WordleActorCritic().to(device) # Create the underlying model.
    
# Make a wrapper for the model to take in an observation and output a guess (rather than logits and a value prediction).
# Note we could actually construct this without having to specify a WordleActorCritic (it would make one automatically with default params, in that case).
# Here, we simply construct a WordleActorCritic for clarity on making an underlying model - which is necessary for training.
model = ModelWrapper(guess_list, word_encodings, model=actor_critic, device=device)
```

Optionally, we can load a checkpoint for the model:
```python
model_path = "checkpoints/best_model.pth" # Change as needed.
checkpoint = torch.load(model_path)

actor_critic.load_state_dict(checkpoint['model']) # Load the saved weights.
```

Then, we can get a guess from the model, whenever given an observation, and step through the environment:
```python
guess = model.get_guess(obs) # Get the model guess.
obs, reward, done = env.step(guess) # Step through the environment with the guess, getting the new observation, reward, and whether the game ended.
```

## Helpful Pre-made Code

### Demo the Agent!
To watch the agent play, navigate to `src/demo_script.py`, adjust the word list, answer list, device, and model checkpoint as needed, and click run!

### Train the Agent!
To begin training, navigate to `src/train_script.py` and configure your device, answer and word list, and optimizer as needed (we already have default options set up). For convenience, we simply created a WordleActorCritic wrapper with default parameters, but you may explore different architecture choices by creating individual components like an ```ObservationEncoder```, ```FFNSharedEncoder```, ```CNNSharedEncoder```, ```PolicyHead```, or ```ValueHead``` and passing them into the ```WordleActorCritic``` constructor.

You can then list directories for model checkpoints and tensorboard logging, and uncomment the following lines as needed to load checkpoints later on. The final line will run the training loop, with plenty of options to play with like number of epochs, batch size, FIFO threshold, etc.

### Evaluate the Agent!
You can then evaluate the trained agent across all possible answers by running `evaluate_policy_on_all_answers` from ` src/eval/evaluate_policy.py`, printing and returning average number of guesses and win rate.

## Future Work
- Not limiting the model to strictly valid words.
- Exploring RNNs like Transformers for sequential modeling in the Shared Encoder.
- Exploring Wordle variants like [Absurdle](https://qntm.org/files/absurdle/absurdle.html) or [Sixdle](https://word.rodeo/sixdle/).
