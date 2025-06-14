import torch
import numpy as np
from envs.WordleView import WordleView
from envs.wordleenv import WordleEnv
from utils.wordtoonehot import word_to_onehot
from utils import LoadList

# To demo the model playing a game of Wordle
def demo_wordle_game(word: str):
    '''Demo a game of Wordle, where the model plays against a fixed word.'''
    
    # Initialize the game with a fixed word
    game = WordleView(word=word)
    game.reset()
    game.render()

    # Load the trained model from pth file
    model_path = 'checkpoint_epoch_790.pth'
    model = torch.load(model_path)
    device = torch.device('cpu')

    answer_list = LoadList.load_word_list('../data/5letteranswersshuffled.txt')[:]
    word_list = list(set(answer_list + LoadList.load_word_list('../data/5letterwords.txt')[:200]))
    word_matrix = torch.stack([word_to_onehot(w) for w in word_list]).to(device)  # shape: [vocab_size, 130]

    # Get the model's guess
    with torch.no_grad():
        batch_answers = answer_list[1]
        env = WordleEnv(word_list, answer_list, batch_size=len(batch_answers))
        obs_list = env.reset(starting_words=batch_answers)

        # Compute logits
        logits, _ = model(obs_list, word_matrix)
        actions = torch.argmax(logits, dim=-1).tolist()

        # Get the guessed words
        guess_words = [word_list[a] for a in actions]

    # Play the game with the model's guesses
    for guess in guess_words:
        # Check if the game is over
        if game.is_game_over():
            break
        
        # Play the guess in the game
        print(f'Guessing: {guess}')
        game.play_guess(guess)
        game.render()
    
    # Print the final result
    if game.is_won:
        print(f'You guessed the word \'{game.word.upper()}\' in {game.num_guesses} guesses!')
    else:
        print(f'Sorry, you lost! The word was: \'{game.word.upper()}\'')

if __name__ == '__main__':
    # Choose a random word
    word_list = LoadList.load_word_list('../data/5letteranswersshuffled.txt')
    random_word = word_list[np.random.randint(len(word_list))]

    demo_wordle_game('apple') # Use random word or chosen word