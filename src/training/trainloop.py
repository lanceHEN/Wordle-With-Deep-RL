import torch
from tqdm import trange
from training.trajectorycollector import generate_trajectory, generate_batched_trajectories
from training.ppotrainer import ppo_update
import random
from training.eval import evaluate_policy_on_all_answers
from envs.wordleenv import WordleEnv
import os


# Main training loop for PPO applied to Wordle
# given batched environments, the model architecutre, word list, answer list, optimizers, number of epochs, number of ppo epochs, minibatch size, gamma, clip_epsilon, and device,
# trains the model for the specified number of epochs, at each epoch collecting trajectories from the batched environments, and applying PPO updates
# also prints out avergae reward, success rate, and average number of guesses
# periodically also saves the model
# made in part with generative AI
def training_loop(
    batched_env,
    letter_encoder,
    observation_encoder,
    shared_encoder,
    policy_head,
    value_head,
    optimizer_policy,
    optimizer_value,
    word_list,
    answer_list,
    save_dir,
    num_epochs=1000,
    ppo_epochs=4,
    minibatch_size=32,
    gamma=1.0,
    clip_epsilon=0.2,
    device=torch.device("cpu"),
):
    
    os.makedirs(save_dir, exist_ok=True) # make sure checkpoint dir exists
    
    for epoch in trange(num_epochs, desc="Training"):
        # Collect one full batch of trajectories from the batched environment
        traj = generate_batched_trajectories(
            batched_env,
            word_list,
            observation_encoder,
            shared_encoder,
            policy_head,
            value_head,
            gamma=gamma,
            device=device
        )

        # Normalize advantages
        advantages_tensor = torch.tensor(traj["advantages"], dtype=torch.float32)
        mean, std = advantages_tensor.mean(), advantages_tensor.std()
        traj["advantages"] = [(a - mean) / (std + 1e-8) for a in traj["advantages"]]

        # Create dataset
        dataset = list(zip(
            traj["observations"],
            traj["actions"],
            traj["log_probs"],
            traj["returns"],
            traj["advantages"]
        ))

        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        # PPO updates
        for _ in range(ppo_epochs):
            random.shuffle(indices)

            for start_idx in range(0, dataset_size, minibatch_size):
                batch_indices = indices[start_idx:start_idx + minibatch_size]

                obs_batch = [dataset[i][0] for i in batch_indices]
                act_batch = [dataset[i][1] for i in batch_indices]
                logp_batch = [dataset[i][2] for i in batch_indices]
                ret_batch = [dataset[i][3] for i in batch_indices]
                adv_batch = [dataset[i][4] for i in batch_indices]

                ppo_update(
                    observation_encoder,
                    shared_encoder,
                    policy_head,
                    value_head,
                    optimizer_policy,
                    optimizer_value,
                    obs_batch,
                    act_batch,
                    adv_batch,
                    ret_batch,
                    logp_batch,
                    clip_epsilon=clip_epsilon,
                    device=device
                )
          
        continue     
        if epoch % 50 == 0: # evaluate and save state
            evaluate_policy_on_all_answers(
            env_class=WordleEnv,
            word_list=word_list,
            answer_list=answer_list,
            observation_encoder=observation_encoder,
            shared_encoder=shared_encoder,
            policy_head=policy_head,
            device=device)
            
            # save model state
            checkpoint = {
                "letter_encoder": letter_encoder.state_dict(),
                "observation_encoder": observation_encoder.state_dict(),
                "shared_encoder": shared_encoder.state_dict(),
                "policy_head": policy_head.state_dict(),
                "value_head": value_head.state_dict(),
                "optimizer_policy": optimizer_policy.state_dict(),
                "optimizer_value": optimizer_value.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"))