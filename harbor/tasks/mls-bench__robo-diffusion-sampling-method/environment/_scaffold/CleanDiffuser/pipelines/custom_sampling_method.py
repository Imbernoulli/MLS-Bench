import os
from copy import deepcopy

import d4rl
import gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DQLMlp
from cleandiffuser.utils import report_parameters, DQLCritic, FreezeModules
from utils import set_seed


@hydra.main(config_path="../configs/custom/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}_s{args.seed}/'  # per-seed dir: parallel seeds of one label must not share ckpts
    if args.mode == "train": import shutil; shutil.rmtree(save_path, ignore_errors=True)  # fresh train dir: never silently score a stale exact-step ckpt
    os.makedirs(save_path, exist_ok=True)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), args.normalize_reward)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # ============================================================================
    # FIXED: Policy, Critic and Training
    # ============================================================================
    # Diffusion Q-Learning (DQL): diffusion actor + twin Q critic with BC + Q
    # loss. The trained actor/critic, dataset, environment list, seeds and
    # evaluation loop are fixed — this task is about the sampler, so the thing
    # you edit is the reverse process at inference time, further below.

    # --------------- Network Architecture -----------------
    nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # --------------- Diffusion Model Actor --------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
        x_max=+1. * torch.ones((1, act_dim), device=args.device),
        x_min=-1. * torch.ones((1, act_dim), device=args.device),
        diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)

    # ------------------ Critic ---------------------
    critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
    critic_target = deepcopy(critic).requires_grad_(False).eval()
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

    # ---------------------- Training ----------------------
    if args.mode == "train":

        actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
        critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)

        actor.train()
        critic.train()

        n_gradient_step = 0
        log = {"bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0.}

        prior = torch.zeros((args.batch_size, act_dim), device=args.device)

        for batch in loop_dataloader(dataloader):

            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            # Critic Training
            current_q1, current_q2 = critic(obs, act)

            next_act, _ = actor.sample(
                prior, solver=args.solver,
                n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=True,
                temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=False)

            target_q = torch.min(*critic_target(next_obs, next_act))
            target_q = (rew + (1 - tml) * args.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # Policy Training
            bc_loss = actor.loss(act, obs)
            new_act, _ = actor.sample(
                prior, solver=args.solver,
                n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=False,
                temperature=1.0, condition_cfg=obs, w_cfg=1.0, requires_grad=True)

            with FreezeModules([critic, ]):
                q1_new_action, q2_new_action = critic(obs, new_act)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + args.task.eta * q_loss

            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()

            actor_lr_scheduler.step()
            critic_lr_scheduler.step()

            # ema
            if n_gradient_step % args.ema_update_interval == 0:
                if n_gradient_step >= 1000:
                    actor.ema_update()
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

            log["bc_loss"] += bc_loss.item()
            log["q_loss"] += q_loss.item()
            log["critic_loss"] += critic_loss.item()
            log["target_q_mean"] += target_q.mean().item()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["bc_loss"] /= args.log_interval
                log["q_loss"] /= args.log_interval
                log["critic_loss"] /= args.log_interval
                log["target_q_mean"] /= args.log_interval
                print(f"TRAIN_METRICS gradient_steps={log['gradient_steps']} "
                      f"bc_loss={log['bc_loss']:.4f} q_loss={log['q_loss']:.4f} "
                      f"critic_loss={log['critic_loss']:.4f} target_q_mean={log['target_q_mean']:.4f}")
                log = {"bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0.}

            if (n_gradient_step + 1) % args.save_interval == 0:
                actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                actor.save(save_path + f"diffusion_ckpt_latest.pt")
                torch.save({
                    "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                }, save_path + f"critic_ckpt_{n_gradient_step + 1}.pt")
                torch.save({
                    "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                }, save_path + f"critic_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.gradient_steps:
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":

        actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.ckpt}.pt")
        critic.load_state_dict(critic_ckpt["critic"])
        critic_target.load_state_dict(critic_ckpt["critic_target"])

        actor.eval()
        critic.eval()
        critic_target.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        # ============================================================================
        # FIXED: NFE accounting — do not modify
        # ============================================================================
        # Counts real denoiser evaluations with a forward hook, so the reported
        # NFE is what your sampler actually spends rather than a number declared
        # in a config file. One hook call == one network evaluation, whatever
        # batch it carries.
        _nfe = {"calls": 0, "samples": 0}

        def _count_nfe(_module, _inputs, _output):
            _nfe["calls"] += 1

        for _m in (actor.model, actor.model_ema):
            try:
                _m["diffusion"].register_forward_hook(_count_nfe)
            except (KeyError, TypeError, AttributeError):
                pass

        prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
        for i in range(args.num_episodes):

            env_eval.seed(args.seed + i * args.num_envs) if hasattr(env_eval, "seed") else None; obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)

                _nfe["samples"] += 1

                # ====================================================================
                # EDITABLE REGION: Sampling Algorithm
                # ====================================================================
                # Produce `act` of shape [num_envs * num_candidates, act_dim] by
                # running a reverse diffusion process conditioned on `obs`.
                #
                # The default below delegates to CleanDiffuser's built-in solvers,
                # driven by `solver` / `sampling_steps` in the YAML. You are not
                # limited to that: implement the reverse process yourself and call
                # the denoiser directly —
                #
                #   net = actor.model_ema["diffusion"] if args.use_ema else actor.model["diffusion"]
                #   pred = net(x_t, t, cond)        # eps or x0, per actor.predict_noise
                #
                # with the schedule available from actor.alphas / actor.sigmas (see
                # cleandiffuser/diffusion/diffusionsde.py). Fewer evaluations at the
                # same return is the point: every call to the denoiser is counted and
                # reported as the NFE the score penalizes, so the cost you pay is the
                # cost you are scored on.
                act, log = actor.sample(
                    prior,
                    solver=args.solver,
                    n_samples=args.num_envs * args.num_candidates,
                    sample_steps=args.sampling_steps,
                    condition_cfg=obs, w_cfg=1.0,
                    use_ema=args.use_ema, temperature=args.temperature)
                # ====================================================================
                # FIXED: Candidate Selection and Environment Step
                # ====================================================================

                with torch.no_grad():
                    q = critic_target.q_min(obs, act)
                    q = q.view(-1, args.num_candidates, 1)
                    w = torch.softmax(q * args.task.weight_temperature, 1)
                    act = act.view(-1, args.num_candidates, act_dim)

                    indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
                    sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

                obs, rew, done, info = env_eval.step(sampled_act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew

                if np.all(cum_done):
                    break

            episode_rewards.append(ep_reward)

        raw_episode_rewards = episode_rewards
        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        mean_score = float(np.mean(episode_rewards))
        std_score = float(np.std(episode_rewards))
        mean_ep_reward = float(np.mean(raw_episode_rewards))
        print(f"EVAL_METRICS normalized_score={mean_score:.4f} normalized_score_std={std_score:.4f} episode_reward={mean_ep_reward:.2f}")

        # Ceil, not round: an adaptive sampler averaging 10.5 evaluations must
        # not report 10 and collect the full no-penalty credit reserved for a
        # 10-step budget. Round-half-to-even would also floor an average of 0.5
        # to zero. Constant-call samplers are unaffected.
        measured_nfe = -(-_nfe["calls"] // max(_nfe["samples"], 1))
        print(f"NFE_METRICS sampling_steps={measured_nfe}", flush=True)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
