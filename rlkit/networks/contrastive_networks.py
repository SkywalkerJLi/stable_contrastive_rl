import torch
from torch import nn

from rlkit.torch.networks import Mlp, CNN
from rlkit.torch import pytorch_util as ptu


class ContrastiveQf(nn.Module):
    def __init__(self,
                 hidden_sizes,
                 representation_dim,
                 action_dim,
                 obs_dim=None,
                 use_image_obs=False,
                 imsize=None,
                 img_encoder_arch='cnn',
                 repr_norm=False,
                 repr_norm_temp=True,
                 repr_log_scale=None,
                 twin_q=False,
                 **kwargs,
                 ):
        super().__init__()

        self._use_image_obs = use_image_obs
        self._imsize = imsize
        self._img_encoder_arch = img_encoder_arch
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._representation_dim = representation_dim
        self._repr_norm = repr_norm
        self._repr_norm_temp = repr_norm_temp
        self._twin_q = twin_q

        self._img_encoder = None
        if self._use_image_obs:
            assert isinstance(imsize, int)
            cnn_kwargs = kwargs.copy()
            layer_norm = cnn_kwargs.pop('layer_norm')

            if self._img_encoder_arch == 'cnn':
                cnn_kwargs['input_width'] = imsize
                cnn_kwargs['input_height'] = imsize
                cnn_kwargs['input_channels'] = 3
                cnn_kwargs['output_size'] = None
                cnn_kwargs['kernel_sizes'] = [8, 4, 3]
                cnn_kwargs['n_channels'] = [32, 64, 64]
                cnn_kwargs['strides'] = [4, 2, 1]
                cnn_kwargs['paddings'] = [2, 1, 1]
                cnn_kwargs['conv_normalization_type'] = 'layer' if layer_norm else 'none'
                cnn_kwargs['fc_normalization_type'] = 'layer' if layer_norm else 'none'
                cnn_kwargs['output_conv_channels'] = True
                self._img_encoder = CNN(**cnn_kwargs)
            else:
                raise RuntimeError("Unknown image encoder architecture: {}".format(
                    self._img_encoder_arch))

        state_dim = self._img_encoder.conv_output_flat_size if self._use_image_obs else self._obs_dim

        self.state_action_encoder = Mlp(
            hidden_sizes, representation_dim, state_dim + self._action_dim,
            **kwargs,
        ) # output is size 16
        self.future_state_encoder = Mlp(
            hidden_sizes, representation_dim, state_dim,
            **kwargs,
        ) # output is size 16
        self.state_action_encoder_2 = Mlp(
            hidden_sizes, representation_dim, state_dim + self._action_dim,
            **kwargs,
        ) # output is size 16
        self.future_state_encoder_2 = Mlp(
            hidden_sizes, representation_dim, state_dim,
            **kwargs,
        ) # output is size 16

    ### OUR IMPLEMENTATION ###
    def _compute_representation(self, obs, action, encoded_images=None):
        # We first convert the observation images into a flattened state and future state representation
        # Rather than converting the images twice, we can compute it for the first Q function and then pass it 
        # into the second
        if encoded_images is None:
            imlen = self._imsize * self._imsize * 3
            state = self._img_encoder(obs[:, :imlen]).flatten(1) # passes the first image in the observation into the CNN which outputs a state representation 
            future_state = self._img_encoder(obs[:, imlen:]).flatten(1) # passes the second image in the observation into the CNN which outputs a future state representation
        else:
            state, future_state = encoded_images

        if encoded_images is None:
            state_action_repr = self.state_action_encoder(torch.cat([state, action], dim=-1)) # concats the state rep with the action rep to get a sa rep
            future_state_repr = self.future_state_encoder(future_state) # converts the goal representation into a size 16 representation
        else:
            state_action_repr = self.state_action_encoder_2(torch.cat([state, action], dim=-1))
            future_state_repr = self.future_state_encoder_2(future_state)

        return state_action_repr, future_state_repr, (state, future_state)

    ### OUR IMPLEMENTATION ###
    def forward(self, obs, action): # The forward pass returns both Q functions 

        # First Q function 
        state_action_repr, future_state_repr, encoded_images = self._compute_representation(obs, action)

        # computes the similarity metric between different state action and goal representations within a batch
        # iterates over all state action reps and goal reps. The diagonal represents the state action rep and goal rep from 
        # THE SAME OBSERVATION --> so this ideally should be high (close to 1)
        outer = torch.bmm(state_action_repr.unsqueeze(0), future_state_repr.permute(1, 0).unsqueeze(0))[0] 

        # Second Q function reusing the encoded images
        state_action_repr_2, future_state_repr_2, _ = self._compute_representation(obs, action, encoded_images)
        outer2 = torch.bmm(state_action_repr_2.unsqueeze(0), future_state_repr_2.permute(1, 0).unsqueeze(0))[0]

        # Stack the two Q function values together
        outer = torch.stack([outer, outer2], dim=-1)

        return outer
