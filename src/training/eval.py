import torch
from envs.wordleenv import WordleEnv

# given an environment class, word list, answer list, model architecture, and device, evaluates model performance on every answer
# word, including win rate (% of time the answer is found in time) and average guesses
# does this in batches to speed up computation
# made with help of generative AI
def evaluate_policy_on_all_answers(env_class, word_list, answer_list, word_matrix, observation_encoder, shared_encoder, policy_head, batch_size=512, device="cpu"):
    total_games = len(answer_list)
    total_wins = 0
    total_guesses = []

    with torch.no_grad():
        # Dummy input for fallback use
        dummy_obs = WordleEnv(word_list, answer_list).reset("crane")
        g_dummy, m_dummy = observation_encoder([dummy_obs])
        g_dummy = g_dummy.squeeze(0)
        m_dummy = m_dummy.squeeze(0)
        
        for batch_start in range(0, total_games, batch_size):
            #print(batch_start)
            batch_answers = answer_list[batch_start:batch_start + batch_size]
            #print("before reset")
            #print("before construction")
            env = env_class(word_list, answer_list, batch_size=len(batch_answers)) # batch size should always beb exact
            #print("constructed")
            obs_list = env.reset(starting_words=batch_answers)
            #print("after reset")

            guesses = [0 for _ in range(len(batch_answers))]
            done_flags = [False] * len(batch_answers)
            won_flags = [False] * len(batch_answers)
            rewards = [0.0] * len(batch_answers)

            #print("right before starting guesses")
            for _ in range(6):
                #print("guess", _)
                if all(done_flags):
                    break

                # Filter obs_list into ones that are still active
                active_obs_indices = [i for i, done in enumerate(done_flags) if not done]
                active_obs = [obs_list[i] for i in active_obs_indices]

                # Batched encoding only for active observations
                if active_obs:
                    grids_active, metas_active = observation_encoder(active_obs)
                    grids_active = list(grids_active)  # unstack [B, 6, 5, D] -> list of [6, 5, D]
                    metas_active = list(metas_active)  # unstack [B, 2]       -> list of [2]
                else:
                    grids_active = metas_active = None

                # Prepare full batches by inserting dummy values for completed games
                grid_batch, meta_batch, valid_indices_batch = [], [], []

                active_idx = 0
                for i in range(len(obs_list)):
                    if not done_flags[i]:
                        grid_batch.append(grids_active[active_idx])
                        meta_batch.append(metas_active[active_idx])
                        valid_indices_batch.append(obs_list[i]["valid_indices"])
                        active_idx += 1
                    else:
                        grid_batch.append(g_dummy)
                        meta_batch.append(m_dummy)
                        valid_indices_batch.append([0])  # dummy
                        
                # Stack and move to device
                grid_tensor = torch.stack(grid_batch).to(device)
                meta_tensor = torch.stack(meta_batch).to(device)
                
                logits = policy_head(shared_encoder(grid_tensor, meta_tensor), valid_indices_batch, word_matrix)
                actions = torch.argmax(logits, dim=-1).tolist()

                guess_words = [word_list[a] for a in actions]
                obs_list, reward_list, done_list = env.step(guess_words)
                
                #print("reward_list:", reward_list)

                for i in range(len(batch_answers)):
                    if not done_flags[i]:
                        guesses[i] += 1
                        rewards[i] = reward_list[i]
                        if done_list[i] and reward_list[i] == 30: # did they win
                            won_flags[i] = True
                        done_flags[i] = done_list[i]

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
