# Solving Wordle with Deep RL

This repository contains the codebase for a deep reinforcement learning (RL) agent that learns to solve Wordle in as few guesses as possible using Proximal Policy Optimization (PPO). The project treats Wordle as an MDP with different rewards for winning, losing, or making incorrect guesses, and experiments with various architectures to boost performance.

## Methodology Overview

We model Wordle as a Markov Decision Process (MDP) with:
- **States**: Feedback from previous guesses, current turn number, and valid guess indices
- **Actions**: Valid word guesses that do not contradict known feedback
- **Rewards**: Positive reward for winning, negative for losing, and a small penalty for incorrect guesses to encourage efficiency
- **Discount factor**: $\gamma = 1$, since there's no more than 6 steps

Training is conducted using PPO, which involves a neural network to pick actions from a probability distribution over them, consisting of:
- **Observation Encoder**: Given an observed state dictionary, produces a numerical representation for neural networks consisting of:
  - A grid tensor, which stores learned encodings for letters and one-hot encodings for feedback at each position on the board
  - A meta vector, which stores additional information including turn number and number of remaining valid guesses (normalized)
- **Shared Encoder**: Given a grid tensor and meta vector from the Observation Encoder, produces a latent vector representation for use by both the Policy and Value heads
- **Policy Head**: Given the latent vector representation, produces a query vector which can then be multiplied with a matrix of word encodings to produce logits over each word (mask those for valid indices to $- \infty$), from which a probability distribution may be derived via softmax
- **Value Head**: Given the latent vector representation, produces a scalar prediction of the value of the original state

The actual PPO training scheme involves first collecting a number of episodes or trajectories from the environment, where each trajectory consists of states, actions, rewards, and other relevant values from full games. For each step in these episodes, we compute:

- **Returns**: The sum of current and future rewards from that step (undiscounted in our case, since $\gamma = 1$).
- **Advantages**: The difference between actual return and predicted state value, used to measure how much better or worse the action performed than expected.
  
Training proceeds by updating the policy and value networks using mini-batches from these trajectories across several PPO epochs. The policy is updated by maximizing a clipped objective that prevents overly large updates, stabilizing learning. Meanwhile, the value network is trained to regress toward the actual returns.

To focus learning on challenging words, we maintain a FIFO queue storing words that took more than a fixed number of guesses to answer. Rather than generating solely random episodes, a fixed percentage of them are reserved to use words popped from the queue.

Our implementation supports training with either a feedforward network (FFN) or a convolutional neural network (CNN) in the shared encoder.

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
└── src # Directory containing the implementation of the agent and environment
    ├── demo_script.py # Demonstrates a model on a visual Wordle game
    ├── envs
    │   ├── __init__.py
    │   ├── batched_env.py # Wraps multiple environments together in a batch for faster training
    │   ├── test_wordle_game.py # Tests for WordleGame
    │   ├── wordle_env.py # Contains an environment wrapper for Wordle, for agent interaction
    │   └── wordle_game.py # Contains the implementation of Wordle, as class WordleGame
    ├── eval
    │   ├── __init__.py
    │   └── eval.py # Provides code to evaluate a model's average guesses and win rate
    ├── models
    │   ├── __init__.py
    │   ├── cnn_encoder.py # A CNN implementation of the Shared Encoder
    │   ├── ffn_shared_encoder.py # A FFN implementation of the Shared Encoder
    │   ├── letter_encoder.py # Learnable letter embeddings
    │   ├── obs_shared_wrapper.py # A wrapper around the Observation and Shared Encoder
    │   ├── observation_encoder.py # The Observation Encoder
    │   ├── policy_head.py # The Policy Head
    │   ├── value_head.py # The Value Head
    │   ├── wordle_actor_critic.py # A wrapper around each component of the model architecture
    │   └── wordle_model_wrapper.py # A wrapper around a WordleActorCritic, which simply produces a human-readable guess any time it's given an observation
    ├── train_script.py # A script to train the model
    ├── training # A directory for PPO training implementation
    │   ├── __init__.py
    │   ├── ppo_trainer.py
    │   ├── train_loop.py
    │   └── trajectory_collector.py
    ├── utils # A directory for random utility functions
    │   ├── __init__.py
    │   ├── load_list.py # Provides a function to load a word file into a list of strings
    │   └── word_to_onehot.py # Provides a function to produce an encoding for a word as a concatenation of its one-hot letter encodings
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

## Demo the Agent!
To watch the agent play, navigate to `src/demo_script.py`, adjust the word list, answer list, device, and model checkpoints as needed, and click run!

## Train the Agent!
To begin training, navigate to `src/train_script.py` and configure your device, answer and word list, and optimizer as needed (we already have default options set up). For convenience, we simply created a WordleActorCritic wrapper with default parameters, but you may explore different architecture choices by creating individual components like an ObservationEncoder, FFNSharedEncoder, CNNSharedEncoder, PolicyHead, or ValueHead and passing them into the WordleActorCritic constructor.


You can then list directories for model checkpoints and tensorboard logging, and uncomment the following lines as needed to load checkpoints later on. The final line will run the training loop, with plenty of options to play with like number of epochs, batch size, FIFO threshold, etc.

## Evaluate the Agent!
You can then evaluate the trained agent across all possible answers by running `evaluate_policy_on_all_answers` from ` src/eval/eval.py`, printing and returning average number of guesses and win rate.

## Future Work
- Not limiting the model to strictly valid words
- Exploring RNNs like Transformers for sequential modeling in the Shared Encoder
- Exploring Wordle variants like [Absurdle](https://qntm.org/files/absurdle/absurdle.html) or [Sixdle](https://word.rodeo/sixdle/)
