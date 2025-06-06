import torch
from envs.wordleenv import WordleEnv

# given an environment class, word list, answer list, model architecture, and device, evaluates model performance on every answer
# word, including win rate (% of time the answer is found in time) and average guesses
# does this in batches to speed up computation
# made with help of generative AI
def evaluate_policy_on_all_answers(env_class, word_list, answer_list, observation_encoder, shared_encoder, policy_head, batch_size=32, device="cpu"):
    total_games = len(answer_list)
    total_wins = 0
    total_guesses = []

    with torch.no_grad():
        # Dummy input for fallback use
        dummy_obs = WordleEnv(word_list, answer_list).reset("crane")
        g_dummy, m_dummy = observation_encoder(dummy_obs)
        for batch_start in range(0, total_games, batch_size):
            batch_answers = answer_list[batch_start:batch_start + batch_size]
            env = env_class(word_list, answer_list, starting_words=batch_answers, batch_size=batch_size)
            obs_list = env.reset()

            guesses = [0 for _ in range(len(batch_answers))]
            done_flags = [False] * len(batch_answers)
            rewards = [0.0] * len(batch_answers)

            for _ in range(6):
                if all(done_flags):
                    break

                grid_batch, meta_batch, valid_indices_batch = [], [], []
                for i, obs in enumerate(obs_list):
                    if not done_flags[i]:
                        g, m = observation_encoder(obs)
                        grid_batch.append(g)
                        meta_batch.append(m)
                        valid_indices_batch.append(obs["valid_indices"])
                    else:
                        # dummy input for done games
                        grid_batch.append(torch.zeros_like(g_dummy))
                        meta_batch.append(torch.zeros_like(m_dummy))
                        valid_indices_batch.append([0])  # dummy

                grid_tensor = torch.stack(grid_batch).to(device)
                meta_tensor = torch.stack(meta_batch).to(device)
                logits = policy_head(shared_encoder(grid_tensor, meta_tensor), valid_indices_batch)
                actions = torch.argmax(logits, dim=-1).tolist()

                guess_words = [word_list[a] for a in actions]
                obs_list, reward_list, done_list = env.step(guess_words)

                for i in range(len(batch_answers)):
                    if not done_flags[i]:
                        guesses[i] += 1
                        rewards[i] = reward_list[i]
                        done_flags[i] = done_list[i]

            for r, g in zip(rewards, guesses):
                if r == 1:
                    total_wins += 1
                    total_guesses.append(g)

    win_rate = total_wins / total_games
    avg_guesses = sum(total_guesses) / len(total_guesses) if total_guesses else float("inf")

    print(f"[Evaluation] Win rate: {win_rate:.3f} ({total_wins}/{total_games})")
    print(f"[Evaluation] Avg guesses: {avg_guesses:.2f}")

    return win_rate, avg_guesses
