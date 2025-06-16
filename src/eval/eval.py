import torch
from envs.wordle_env import WordleEnv

# given an environment class, word list, answer list, word embeddings, model, and batch_size,
# evaluates model performance on every answer word, including win rate (% of time the answer is found in time)
# and average guesses used
# does this in batches with given batch_size to speed up computation
# returns win rate and avg_guesses, in addition to printing them
# made with help of generative AI
def evaluate_policy_on_all_answers(env_class, word_list, answer_list, word_matrix, actor_critic, batch_size=512):
    total_games = len(answer_list)
    total_wins = 0
    total_guesses = []

    with torch.no_grad():
        
        for batch_start in range(0, total_games, batch_size):
            batch_answers = answer_list[batch_start:batch_start + batch_size]

            env = env_class(word_list, answer_list, batch_size=len(batch_answers)) # batch size should always beb exact
            obs_list = env.reset(starting_words=batch_answers)

            guesses = [0 for _ in range(len(batch_answers))]
            done_flags = [False] * len(batch_answers)
            won_flags = [False] * len(batch_answers)
            rewards = [0.0] * len(batch_answers)

            for _ in range(6):
                if all(done_flags):
                    break

                # Compute logits
                logits, _ = actor_critic(obs_list, word_matrix)
                actions = torch.argmax(logits, dim=-1).tolist()

                # Get the guessed words
                guess_words = [word_list[a] for a in actions]

                # Step the environment
                obs_list, reward_list, done_list = env.step(guess_words)

                for i in range(len(batch_answers)):
                    if not done_flags[i]:
                        guesses[i] += 1
                        rewards[i] = reward_list[i]
                        if done_list[i] and reward_list[i] == 20:
                            won_flags[i] = True
                        done_flags[i] = done_list[i]

            # count number of wins
            for won, g in zip(won_flags, guesses):
                if won:
                    total_wins += 1
                total_guesses.append(g)
                    
            #break

    win_rate = total_wins / total_games
    avg_guesses = sum(total_guesses) / len(total_guesses) if total_guesses else float("inf")

    print(f"[Evaluation] Win rate: {win_rate:.3f} ({total_wins}/{total_games})")
    print(f"[Evaluation] Avg guesses: {avg_guesses:.2f}")
    #print(total_guesses)

    return win_rate, avg_guesses
