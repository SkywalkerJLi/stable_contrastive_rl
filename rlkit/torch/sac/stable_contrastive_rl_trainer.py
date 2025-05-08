from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix

from rlkit.utils.data_augmentation import AUG_TO_FUNC


class StableContrastiveRLTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf,
            target_qf=None,
            discount=0.99,
            lr=3e-4,
            gradient_clipping=None,  # TODO
            optimizer_class=optim.Adam,
            update_period=1,
            soft_target_tau=1e-2,
            target_update_period=1,
            use_td=False,
            entropy_coefficient=None,
            target_entropy=None,
            bc_coef=0.05,
            augment_order=[],
            augment_probability=0.0,

            * args,
            **kwargs
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf = qf
        self.target_qf = target_qf
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.gradient_clipping = gradient_clipping
        self.use_td = use_td
        self.entropy_coefficient = entropy_coefficient
        self.adaptive_entropy_coefficient = entropy_coefficient is None
        self.target_entropy = target_entropy
        self.bc_coef = bc_coef
        self.discount = discount
        self.update_period = update_period
        self.augment_probability = augment_probability

        if self.adaptive_entropy_coefficient:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=lr,
            )

        self.qf_criterion = nn.BCEWithLogitsLoss(reduction='none')

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=lr,
        )

        ptu.copy_model_params_from_to(self.qf, self.target_qf)

        self.eval_statistics = OrderedDict()
        self.n_train_steps_total = 0

        self.need_to_update_eval_statistics = {
            'train/': True,
            'eval/': True,
        }

        self.augment_stack = None
        self.augment_funcs = {}
        if augment_probability > 0:
            self.augment_funcs = {}
            for aug_name in augment_order:
                assert aug_name in AUG_TO_FUNC, 'invalid data aug string'
                self.augment_funcs[aug_name] = AUG_TO_FUNC[aug_name]

    def augment(self, batch, train=True):
        augmented_batch = dict()
        for key, value in batch.items():
            augmented_batch[key] = value

        if (train and self.augment_probability > 0 and
                torch.rand(1) < self.augment_probability and
                batch['observations'].shape[0] > 0):
            width = self.policy.input_width
            height = self.policy.input_height
            channel = self.policy.input_channels // 2

            img_obs = batch['observations'].reshape(
                -1, channel, width, height)
            next_img_obs = batch['next_observations'].reshape(
                -1, channel, width, height)
            img_goal = batch['contexts'].reshape(
                -1, channel, width, height)

            # transpose to (B, C, H, W)
            aug_img_obs = img_obs.permute(0, 1, 3, 2)
            aug_img_goal = img_goal.permute(0, 1, 3, 2)
            aug_next_img_obs = next_img_obs.permute(0, 1, 3, 2)

            for aug, func in self.augment_funcs.items():
                # apply same augmentation
                aug_img_obs_goal = func(torch.cat([aug_img_obs, aug_img_goal], dim=1))
                aug_img_obs, aug_img_goal = aug_img_obs_goal[:, :channel], aug_img_obs_goal[:, channel:]
                aug_next_img_obs = func(aug_next_img_obs)

            # transpose to (B, C, W, H)
            aug_img_obs = aug_img_obs.permute(0, 1, 3, 2)
            aug_img_goal = aug_img_goal.permute(0, 1, 3, 2)
            aug_next_img_obs = aug_next_img_obs.permute(0, 1, 3, 2)

            augmented_batch['augmented_observations'] = aug_img_obs.flatten(1)
            augmented_batch['augmented_next_observations'] = aug_next_img_obs.flatten(1)
            augmented_batch['augmented_contexts'] = aug_img_goal.flatten(1)
        else:
            augmented_batch['augmented_observations'] = augmented_batch['observations']
            augmented_batch['augmented_next_observations'] = augmented_batch['next_observations']
            augmented_batch['augmented_contexts'] = augmented_batch['contexts']

        return augmented_batch

    def train_from_torch(self, batch, train=True):
        if train:
            for net in self.networks:
                net.train(True)
        else:
            for net in self.networks:
                net.train(False)

        batch['observations'] = batch['observations'] / 255.0
        batch['next_observations'] = batch['next_observations'] / 255.0
        batch['contexts'] = batch['contexts'] / 255.0

        batch = self.augment(batch, train=train)

        reward = batch['rewards'] 
        terminal = batch['terminals'] 
        action = batch['actions'] # List of actions from replay buffer batch

        obs = batch['observations'] # List of observations from replay buffer batch (images) 
        print("obs shape: ", obs.shape)
        next_obs = batch['next_observations'] # List of observations from replay buffer (images) 
        print("next_obs shape: ", next_obs.shape)
        goal = batch['contexts']
        print("goal shape: ", goal.shape)

        aug_obs = batch['augmented_observations'] # Size (B, C * W * H)
        aug_goal = batch['augmented_contexts'] # Size (B, C * W * H)

        batch_size = obs.shape[0]

        ### OUR IMPLEMENTATION ###

        # Use the next state as the goal state
        new_goal = next_obs

        # logits are the batch multiplication of the state action representation and future state representation
        # it encodes the similarity between a state action rep and the future rep for all batches
        # Shape (B, B ,2)
        logits = self.qf(torch.cat([obs, new_goal], -1), action)
        print("logits shape: ", logits.shape)

        # we evaluate the next-state Q function using random goals
        # shifts indicies one to the left to find next state targets
        # Shape (B,)
        random_state_indices = torch.roll(torch.arange(batch_size, dtype=torch.int64), -1)
        print("random_state_indices shape: ", random_state_indices.shape)

        # Each future state is shuffled 
        random_future_state = new_goal[random_state_indices]
        print("random_future_state shape: ", random_future_state.shape)

        # Append the next observation image with a different future state image 
        next_observation_random_future_state = torch.cat([next_obs, random_future_state], -1)
        print("next_observation_random_future_state shape: ", next_observation_random_future_state.shape)

        # Given the next state and random future state, output distribution of actions
        next_dist = self.policy(next_observation_random_future_state)

        # Sample action
        next_action = next_dist.rsample()

        # Target Q network to get the likelihood of reaching the random future state, given the next state and next action
        # (B, B, 2)
        next_logits = self.target_qf(next_observation_random_future_state, next_action)
        print("next_logits 1 shape: ", next_logits.shape)

        next_logits = torch.sigmoid(next_logits)
        print("next_logits 2 shape: ", next_logits.shape)

        # take the minimum of the two target Q functions
        #(B, B)
        next_logits = torch.min(next_logits, dim=-1)[0].detach()
        print("next_logits 3 shape: ", next_logits.shape)

        # get the similiarity measure between the state action and its corresponding future state
        # Shape (B,)
        next_v = torch.diag(next_logits)
        print("next_v shape: ", next_v.shape)

        
        # Calculating the W value as in the paper
        # Shape (B,)
        w = next_v / (1 - next_v)
        w_clipping = 20.0
        w = torch.clamp(w, min=0.0, max=w_clipping)
        print("w shape: ", w.shape)

        # (B, B, 2) --> (B, 2), computes diagonal of each twin Q.
        # The diagonal entries of the logits represent the similiarity measure between a state action and its corresponding future state
        pos_logits = torch.diagonal(logits).permute(1, 0) # permuting creates two rows where each row contains the diagonal entries of a Q function
        print("pos_logits shape: ", pos_logits.shape)

        # want the pos_logits to be as close to 1 as possible
        # Shape (B, 2)
        loss_pos = self.qf_criterion(pos_logits, ptu.ones_like(pos_logits))
        print("loss_pos shape: ", loss_pos.shape)
        
        # selects from 0 to B - 1 a specific row from logits, say row i
        # then selects column i+1 due to goal_indicies being the roll of torch.arange(batch_size)
        # so final shape is (B, 2) but each element is a mismatched dot product between the state action and future(goal) state
        # selects a state and action representation from one observation and the future state from another observation
        neg_logits = logits[torch.arange(batch_size), random_state_indices]
        print("neg_logits shape: ", neg_logits.shape)


        # given the next state and action from the current state and action, 
        # if the random future state is very likely to occur from the next state and action, 
        # then we want the neg logits to be close to 1
        # (B, 1) * (B, 2) broadcasts to (B, 2)
        loss_neg_future_weighted = w[:, None] * self.qf_criterion(neg_logits, ptu.ones_like(neg_logits))
        print("loss_neg_future_weighted shape: ", loss_neg_future_weighted.shape)
        
        # want the neg logits which come from the random future states to be close to 0
        # Shape (B, 2)
        loss_neg = self.qf_criterion(neg_logits, ptu.zeros_like(neg_logits))
        print("loss_neg shape: ", loss_neg.shape)


        # loss_pos is log σ(f (s, a, s′))
        # loss_neg_future_weighted is ⌊w(s′, a′, sf )⌋sg log σ(f (s, a, sf ))
        # loss_neg is log(1 − σ(f (s, a, sf )))
        # Shape (B, 2)
        qf_loss = (1 - self.discount) * loss_pos + self.discount * loss_neg_future_weighted + loss_neg
        print("qf_loss 1 shape: ", qf_loss.shape)
        
        # Average loss over batch
        qf_loss = torch.mean(qf_loss)
        print("qf_loss 2 shape: ", qf_loss.shape)

        """
        Policy and Alpha Loss
        """

        # Goal conditioned policy
        obs_goal = torch.cat([obs, goal], -1)

        # Use data augmented goal and observations for policy objective
        aug_obs_goal = torch.cat([aug_obs, aug_goal], -1)

        dist = self.policy(obs_goal)
        dist_aug = self.policy(aug_obs_goal)

        # Get sampled action and log probs
        sampled_action, log_prob = dist.rsample_and_logprob()

        alpha = self.entropy_coefficient

        q_action = self.qf(obs_goal, sampled_action)
        q_action = torch.min(q_action, dim = -1)[0]

        orig_action = action

        actor_q_loss = alpha * log_prob - torch.diag(q_action)

        # Taken from their implementation
        train_mask = ((orig_action * 1E8 % 10)[:, 0] != 4).float()


        actor_aug_log_loss =  - train_mask * dist_aug.log_prob(orig_action)

        actor_loss = (1 - self.bc_coef) * actor_q_loss + self.bc_coef * actor_aug_log_loss

        actor_loss = torch.mean(actor_q_loss)

        if train:
            """
            Optimization.
            """
            if self.n_train_steps_total % self.update_period == 0:
                # if self.adaptive_entropy_coefficient:
                #     self.alpha_optimizer.zero_grad()
                #     alpha_loss.backward()
                #     if (self.gradient_clipping is not None and
                #             self.gradient_clipping > 0):
                #         torch.nn.utils.clip_grad_norm(
                #             [self.log_alpha], self.gradient_clipping)
                #     self.alpha_optimizer.step()

                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                # if (self.gradient_clipping is not None and
                #         self.gradient_clipping > 0):
                #     torch.nn.utils.clip_grad_norm(
                #         self.policy.parameters(), self.gradient_clipping)
                self.policy_optimizer.step()

                self.qf_optimizer.zero_grad()
                qf_loss.backward()
                # if (self.gradient_clipping is not None and
                #         self.gradient_clipping > 0):
                #     torch.nn.utils.clip_grad_norm(
                #         self.qf.parameters(), self.gradient_clipping)
                self.qf_optimizer.step()

            """
            Soft Updates
            """
            if self.n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    self.qf, self.target_qf, self.soft_target_tau
                )

        """
        Save some statistics for eval
        """
        if train:
            prefix = 'train/'
        else:
            prefix = 'eval/'

        if self.need_to_update_eval_statistics[prefix]:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics[prefix + 'QF Loss'] = np.mean(
                ptu.get_numpy(qf_loss))
            self.eval_statistics[prefix + 'Policy Loss'] = np.mean(
                ptu.get_numpy(actor_loss))
            self.eval_statistics[prefix + 'Policy Loss/Actor Q Loss'] = np.mean(
                ptu.get_numpy(actor_q_loss))
            # self.eval_statistics[prefix + 'Policy Loss/GCBC Loss'] = np.mean(
            #     ptu.get_numpy(gcbc_loss_log))
            # self.eval_statistics[prefix + 'Policy Loss/GCBC Val Loss'] = np.mean(
            #     ptu.get_numpy(gcbc_val_loss_log))
            # self.eval_statistics[prefix + 'Policy Loss/Augmented GCBC Loss'] = np.mean(
            #     ptu.get_numpy(aug_gcbc_loss_log))
            # self.eval_statistics[prefix + 'Policy Loss/Augmented GCBC Val Loss'] = np.mean(
            #     ptu.get_numpy(aug_gcbc_val_loss_log))

            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Policy Mean',
                ptu.get_numpy(dist.mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Policy STD',
                ptu.get_numpy(dist.stddev),
            ))

            # # critic statistics
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'qf/sa_repr_norm',
            #     ptu.get_numpy(sa_repr_norm),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'qf/g_repr_norm',
            #     ptu.get_numpy(g_repr_norm),
            # ))

            # if self.qf.repr_norm:
            #     self.eval_statistics[prefix + 'qf/repr_log_scale'] = np.mean(
            #         ptu.get_numpy(self.qf.repr_log_scale))

            # self.eval_statistics[prefix + 'qf/logits_pos'] = np.mean(
            #     ptu.get_numpy(logits_pos))
            # self.eval_statistics[prefix + 'qf/logits_neg'] = np.mean(
            #     ptu.get_numpy(logits_neg))
            # self.eval_statistics[prefix + 'qf/q_pos_ratio'] = np.mean(
            #     ptu.get_numpy(q_pos_ratio))
            # self.eval_statistics[prefix + 'qf/q_neg_ratio'] = np.mean(
            #     ptu.get_numpy(q_neg_ratio))
            # self.eval_statistics[prefix + 'qf/binary_accuracy'] = np.mean(
            #     ptu.get_numpy(binary_accuracy))
            # self.eval_statistics[prefix + 'qf/categorical_accuracy'] = np.mean(
            #     ptu.get_numpy(categorical_accuracy))

            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'logits',
                ptu.get_numpy(logits),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'w',
                ptu.get_numpy(w),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'reward',
                ptu.get_numpy(reward),
            ))
            
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'terminal',
                ptu.get_numpy(terminal),
            ))

            actor_statistics = add_prefix(
                dist.get_diagnostics(), prefix + 'policy/')
            self.eval_statistics.update(actor_statistics)

            # if self.entropy_coefficient is not None:
            self.eval_statistics[prefix + 'alpha'] = alpha
            # else:
            #     self.eval_statistics[prefix + 'alpha'] = np.mean(
            #         ptu.get_numpy(alpha))
            # if self.adaptive_entropy_coefficient:
            #     self.eval_statistics[prefix + 'Alpha Loss'] = np.mean(
            #         ptu.get_numpy(alpha_loss))

        if train:
            self.n_train_steps_total += 1

        self.need_to_update_eval_statistics[prefix] = False

        for net in self.networks:
            net.train(False)

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        for key in self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics[key] = True

    @property
    def networks(self):
        nets = [
            self.policy,
            self.qf,
            self.target_qf,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf=self.qf,
            target_qf=self.target_qf,
        )
