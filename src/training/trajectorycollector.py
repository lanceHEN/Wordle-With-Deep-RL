import torch
from torch.distributions import Categorical

# made with help of generative AI
def generate_trajectory(env, word_list, observation_encoder, shared_encoder, policy_head, value_head, device="cpu", gamma=1):
    """
    Simulates one episode of Wordle using the current policy.

    At each step:
    - Encodes the observation.
    - Computes logits for all words, masking invalid ones.
    - Samples an action after taking softmax over the logits, logs probability, and records reward and value.
    
    Returns:
        A dictionary with everything needed for PPO training - observations, actions, log probs, returns, advantages, and indices of valid words.
    """
    obs = env.reset()
    done = False

    observations = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    valid_indices_all = []

    with torch.no_grad(): # not training during collection
        while not done:
            # Encode observation
            grid_tensor, meta_tensor = observation_encoder(obs)
            grid_tensor = grid_tensor.to(device)
            meta_tensor = meta_tensor.to(device)

            # get latent state
            h_policy = shared_encoder(grid_tensor.unsqueeze(0), meta_tensor.unsqueeze(0))
            h_value = shared_encoder(grid_tensor.unsqueeze(0), meta_tensor.unsqueeze(0))

            # Get valid action logits (from dot product )
        
            valid_indices = obs["valid_indices"]
            scores = policy_head(h_policy, [valid_indices]) # logits for all guessses - [1, vocab_size]
            dist = Categorical(logits=scores)
            # Choose an action
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx) # get log probability - taking the log provides numerical stability

            # Get actual word from index
            action_word = word_list[action_idx.item()]

            # Step the environment
            next_obs, reward, done = env.step(action_word)

            value = value_head(h_value).squeeze() # get predicted value for state

            # Store trajectory
            observations.append(obs)
            actions.append(action_idx.item())
            log_probs.append(log_prob.squeeze(0))
            rewards.append(torch.tensor(reward, dtype=torch.float, device=device))
            values.append(value)
            valid_indices_all.append(valid_indices)

            obs = next_obs
            #print(obs["valid_indices"])

        # final value - always 0 because of terminal state
        last_value = 0

        # Advantage and return calculations
        values.append(torch.tensor(last_value, device=device))
        advantages, returns = compute_advantages(rewards, values, gamma=gamma, device=device)


        return {
            "observations": observations,
            "actions": actions,
            "log_probs": log_probs,
            "returns": returns,
            "advantages": advantages,
            "valid_indices": valid_indices_all
        }

# made with help of generative AI
def compute_advantages(rewards, values, gamma=1, device="cpu"):
    """
    Computes simple advantages and returns.
    Advantages are simply the difference of returns and predicted values.
    
    Args:
        rewards: list of individual rewards [r_0, r_1, ..., r_T-1]
        values: list of value estimates [v_0, v_1, ..., v_T] (note: T+1 entries)
        gamma: discount factor
    
    Returns:
        advantages: A_t = G_t - V(s_t)
    """
    returns = []
    G = values[-1].item()  # start with bootstrapped value 0 
    for r in reversed(rewards): # work backwards from end, accumulating actual rewards
        G = r.item() + gamma * G
        returns.insert(0, G)

    # Convert to tensors
    returns = torch.tensor(returns, dtype=torch.float, device=device)
    values = torch.stack(values[:-1])
    advantages = returns - values

    return advantages, returns

# made with help of generative AI
def generate_batched_trajectories(
    batched_env,
    word_list,
    observation_encoder,
    shared_encoder,
    policy_head,
    value_head,
    gamma=1.0,
    device="cpu"
):
    """
    Simulates each episode of Wordle in the given batched_env simultaneously using the current policy.

    At each step:
    - Encodes the observations.
    - Computes logits for all words, for each episode, masking invalid ones.
    - Samples actions after taking softmax over the scores, logs probabilities, and records rewards and values.
    
    Returns:
        A dictionary with everything needed for PPO training - observations, actions, log probs, returns, advantages, and indices of valid words.
    """
    batch_size = batched_env.batch_size
    obs_list = batched_env.reset()

    all_obs = [[] for _ in range(batch_size)]
    all_actions = [[] for _ in range(batch_size)]
    all_log_probs = [[] for _ in range(batch_size)]
    all_rewards = [[] for _ in range(batch_size)]
    all_values = [[] for _ in range(batch_size)]

    with torch.no_grad():
        while not batched_env.all_done():
            # Vectorized observation encoding
            grids, metas, valid_indices_batch = zip(*[observation_encoder(obs) + (obs["valid_indices"],) for obs in obs_list])
            grids = torch.stack(grids).to(device)
            metas = torch.stack(metas).to(device)

            h = shared_encoder(grids, metas)
            logits = policy_head(h, valid_indices_batch)
            values = value_head(h).squeeze(-1)

            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            words = [word_list[a.item()] for a in actions]

            next_obs_list, rewards, dones = batched_env.step(words)

            for i in range(batch_size):
                if not dones[i]:
                    all_obs[i].append(obs_list[i])
                    all_actions[i].append(actions[i].item())
                    all_log_probs[i].append(log_probs[i])
                    all_rewards[i].append(torch.tensor(rewards[i], dtype=torch.float))
                    all_values[i].append(values[i])

            obs_list = next_obs_list

    # Compute returns and advantages
    trajectories = {"observations": [], "actions": [], "log_probs": [], "returns": [], "advantages": []}

    for i in range(batch_size):
        rewards = all_rewards[i]
        values = all_values[i]
        values.append(torch.tensor(0.0, device=device))  # final value = 0
        returns, advantages = compute_advantages(rewards, values, gamma, device=device)
        trajectories["observations"].extend(all_obs[i])
        trajectories["actions"].extend(all_actions[i])
        trajectories["log_probs"].extend(all_log_probs[i])
        trajectories["returns"].extend(returns)
        trajectories["advantages"].extend(advantages)

    return trajectories
