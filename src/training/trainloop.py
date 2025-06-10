import torch
from tqdm import trange
from training.trajectorycollector import generate_batched_trajectories
from training.ppotrainer import ppo_update
import random
from training.eval import evaluate_policy_on_all_answers
from envs.batchedenv import BatchedWordleEnv
import os
from torch.utils.tensorboard import SummaryWriter
from collections import deque



# Main training loop for PPO applied to Wordle
# trains the model for the specified number of epochs, at each epoch collecting trajectories from the batched environments, and applying PPO updates
# also periodically (every eval_and_save_per epochs) prints out and saves to tensorboard the success rate, and average number of guesses, and also saves the model.
# This also includes a FIFO queue, used for concentrating the model on challenging words - words which take fifo_threshold or more guesses.
# No more than fifo_percentage of the total number of guesses is set aside for words in the FIFO queue.
# made in part with generative AI
def training_loop(
    batched_env, # the batched wordle environments, as a BatchedWordleEnv
    actor_critic, # the wrapped model architecture, as a WordleActorCritic
    optimizer_policy, # optimizer for the policy network
    optimizer_value, # optimizer for the value network
    word_list, # list of all words that can be used as guesses
    answer_list, # list of all words that can be used as answers
    word_matrix, # one hot embeddings for every word in word_list, as a [len(word_list), 130] torch tensor
    save_dir, # directory to save the model parameters
    log_dir, # directory to save tensorboard logs
    num_epochs=1000, # number of overall epochs
    start_epoch=0, # epoch to start (useful for resuming training)
    ppo_epochs=4, # how many ppo epochs to run on a given set of observations
    eval_and_save_per=20, # how often, in # epochs, to evaluate model performance on all words in answer_list and save the state
    minibatch_size=256, # how many steps to consider at once in the PPO update
    gamma=1.0, # discount factor 
    clip_epsilon=0.2, # how tightly to clip the probability ratio for PPO
    entropy_coef=0.01, # coefficient for entropy in model loss - the higher, the more it is encouraged to have more balanced probabilites, encouraging exploration
    entropy_decay=0.95, # the rate at which to decay entropy_coef, applied at the end of each overall epoch
    device=torch.device("cpu"), # device
    fifo_queue=deque(maxlen=20), # size of the FIFO queue to store challenging words
    fifo_threshold=5, # minimum number of required guesses to add a word to the FIFO queue
    fifo_percentage=0.2, # percentage of words in the environment that should be devoted to FIFO words
    scheduler_policy=None, # optional scheduler for policy optimizer
    scheduler_value=None# optional scheduler for value optimizer
):
    
    os.makedirs(save_dir, exist_ok=True) # make sure checkpoint dir exists
    writer = SummaryWriter(log_dir)
    
    for epoch in trange(start_epoch, num_epochs, desc="Training"):
        # Collect one full batch of trajectories from the batched environment
        traj = generate_batched_trajectories(
            batched_env,
            word_list,
            answer_list,
            word_matrix,
            actor_critic,
            gamma=gamma,
            device=device,
            fifo_queue=fifo_queue,
            fifo_percentage=fifo_percentage
        )

        # Add hard solutions to FIFO queue
        # Meaning: words that took at least fifo_threshold guesses, whether they were found or not
        for answer, guesses in zip(traj["env_answers"], traj["num_guesses"]):
            if guesses >= fifo_threshold: # at least fifo_threshold guesses
                fifo_queue.append(answer)
                
        #print("FIFO queue:", fifo_queue) # helpful to print to monitor progress on reducing guesses

        # Normalize advantages for more stable PPO updates
        advantages_tensor = torch.tensor(traj["advantages"], dtype=torch.float32)
        mean, std = advantages_tensor.mean(), advantages_tensor.std()
        traj["advantages"] = [(a - mean) / (std + 1e-8) for a in traj["advantages"]]

        # Create dataset for PPO
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
        for ppo_epoch in range(ppo_epochs):
            random.shuffle(indices)

            for start_idx in range(0, dataset_size, minibatch_size):
                batch_indices = indices[start_idx:start_idx + minibatch_size]

                obs_batch = [dataset[i][0] for i in batch_indices]
                act_batch = [dataset[i][1] for i in batch_indices]
                logp_batch = [dataset[i][2] for i in batch_indices]
                ret_batch = [dataset[i][3] for i in batch_indices]
                adv_batch = [dataset[i][4] for i in batch_indices]

                ppo_update(
                    actor_critic,
                    optimizer_policy,
                    optimizer_value,
                    obs_batch,
                    act_batch,
                    adv_batch,
                    ret_batch,
                    logp_batch,
                    word_matrix,
                    clip_epsilon=clip_epsilon,
                    entropy_coef=entropy_coef,
                    device=device,
                    writer=writer,
                    global_step=epoch * ppo_epochs + ppo_epoch
                )
                
        entropy_coef *= entropy_decay # update entropy
          
        #continue   
        # update schedulers
        if scheduler_policy:
            scheduler_policy.step()
        if scheduler_value:
            scheduler_value.step()
        
        if epoch % eval_and_save_per == 0: # evaluate and save state
            print("Evaluating policy on all answers...")
            win_rate, avg_guesses = evaluate_policy_on_all_answers(
            env_class=BatchedWordleEnv,
            word_list=word_list,
            answer_list=answer_list,
            word_matrix=word_matrix,
            actor_critic=actor_critic,
            device=device)
            writer.add_scalar("Eval/win_rate", win_rate, epoch)
            writer.add_scalar("Eval/avg_guesses", avg_guesses, epoch)
            
            print(f"Saving model state for epoch {epoch}...")
            
            # save model state
            checkpoint = {
                "letter_encoder": actor_critic.obs_shared.observation_encoder.letter_encoder.state_dict(),
                "observation_encoder": actor_critic.obs_shared.observation_encoder.state_dict(),
                "shared_encoder": actor_critic.obs_shared.shared_encoder.state_dict(),
                "policy_head": actor_critic.policy_head.state_dict(),
                "value_head": actor_critic.value_head.state_dict(),
                "optimizer_policy": optimizer_policy.state_dict(),
                "optimizer_value": optimizer_value.state_dict(),
                "epoch": epoch,
            }
            if scheduler_policy:
                checkpoint["scheduler_policy"] = scheduler_policy.state_dict()
            if scheduler_value:
                checkpoint["scheduler_value"] = scheduler_value.state_dict()
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"))
            print("Model state saved!")
            
    # close writer
    writer.close()