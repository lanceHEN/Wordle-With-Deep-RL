import torch
from tqdm import trange
from training.trajectorycollector import generate_trajectory, generate_batched_trajectories
from training.ppotrainer import ppo_update
import random
from training.eval import evaluate_policy_on_all_answers
from envs.batchedenv import BatchedWordleEnv
import os
from torch.utils.tensorboard import SummaryWriter
from collections import deque



# Main training loop for PPO applied to Wordle
# given batched environments, the model architecutre, word list, answer list, optimizers, number of epochs, number of ppo epochs, minibatch size, gamma, clip_epsilon, and device,
# trains the model for the specified number of epochs, at each epoch collecting trajectories from the batched environments, and applying PPO updates
# also periodically (every eval_and_save_per epochs) prints out and saves to tensorboard avergae success rate, and average number of guesses, and also saves the model
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
    word_matrix,
    save_dir,
    log_dir,
    num_epochs=1000,
    start_epoch=0,
    ppo_epochs=4,
    eval_and_save_per=20,
    minibatch_size=32,
    gamma=1.0,
    clip_epsilon=0.2,
    device=torch.device("cpu"),
    fifo_queue=deque(maxlen=20),
    fifo_threshold=5,
    fifo_percentage=0.2,
    scheduler_policy=None,
    scheduler_value=None
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
            observation_encoder,
            shared_encoder,
            policy_head,
            value_head,
            gamma=gamma,
            device=device,
            fifo_queue=fifo_queue,
            fifo_percentage=fifo_percentage
        )

        # Add hard solutions to FIFO queue
        # Meaning: words that took at least fifo_threshold guesses, whether they were found or not
        for answer, guesses in zip(traj["envAnswers"], traj["numGuesses"]):
            if guesses >= fifo_threshold: # at least fifo_threshold guesses
                fifo_queue.append(answer)
                
        print(fifo_queue)

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
                    word_matrix,
                    clip_epsilon=clip_epsilon,
                    device=device,
                    writer=writer,
                    global_step=epoch * ppo_epochs + ppo_epoch
                )
          
        #continue   
        # update schedulers  
        if scheduler_policy:
            scheduler_policy.step()
        if scheduler_value:
            scheduler_value.step()
        
        if epoch % eval_and_save_per == 0: # evaluate and save state
            print("evaluating policy on all answers...")
            win_rate, avg_guesses = evaluate_policy_on_all_answers(
            env_class=BatchedWordleEnv,
            word_list=word_list,
            answer_list=answer_list,
            word_matrix=word_matrix,
            observation_encoder=observation_encoder,
            shared_encoder=shared_encoder,
            policy_head=policy_head,
            device=device)
            writer.add_scalar("Eval/win_rate", win_rate, epoch)
            writer.add_scalar("Eval/avg_guesses", avg_guesses, epoch)
            
            print("saving model state...")
            # save model state
            checkpoint = {
                "letter_encoder": letter_encoder.state_dict(),
                "observation_encoder": observation_encoder.state_dict(),
                "shared_encoder": shared_encoder.state_dict(),
                "policy_head": policy_head.state_dict(),
                "value_head": value_head.state_dict(),
                "optimizer_policy": optimizer_policy.state_dict(),
                "optimizer_value": optimizer_value.state_dict(),
                "scheduler_policy": scheduler_policy.state_dict(),
                "scheduler_value": scheduler_value.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"))
            
    # close writer
    writer.close()