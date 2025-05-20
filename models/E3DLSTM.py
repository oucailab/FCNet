import torch
import torch.nn as nn
from layers.E3DLSTM.Eidetic3DLSTMCell import Eidetic3DLSTMCell


class E3DLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        img_size,
        patch_size,
        kernel_size_3D,
        layer_norm,
    ):
        super(E3DLSTM, self).__init__()

        H, W = img_size
        height = H // patch_size[0]
        width = W // patch_size[1]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)  # num_layers: LSTM的层数，需要与len(hidden_dim)相等

        self.window_length = 2
        self.window_stride = 1

        self.MSE_criterion = nn.MSELoss()
        self.L1_criterion = nn.L1Loss()

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(
                Eidetic3DLSTMCell(
                    cur_input_dim,
                    self.hidden_dim[i],
                    self.window_length,
                    height,
                    width,
                    kernel_size_3D,
                    layer_norm,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_output = nn.Conv3d(
            self.hidden_dim[-1],
            output_dim,
            kernel_size=(self.window_length, 1, 1),
            stride=(self.window_length, 1, 1),
            bias=False,
        )

    def forward(self, input_x, targets):
        """
        Args:
            input_x: 5D张量，[b, l, c, h, w]
            land_mask: 陆地掩码
        Returns:
            output: 5D张量，[b, 1, c, h, w]
        """
        input_x = input_x.contiguous()
        targets = targets.contiguous()
        device = input_x.device

        assert len(input_x.shape) == 5
        seq_len = input_x.size(1)

        batch_size = input_x.size(0)
        height = input_x.size(3)
        width = input_x.size(4)

        next_frames = []
        h_t = []
        c_t = []
        c_history = []
        input_list = []

        for t in range(self.window_length - 1):
            input_list.append(torch.zeros_like(input_x[:, 0]))

        for i in range(self.num_layers):
            zeros = torch.zeros(
                batch_size,
                self.hidden_dim[i],
                self.window_length,
                height,
                width,
                device=device,
            )
            h_t.append(zeros)
            c_t.append(zeros)
            c_history.append(zeros)

        memory = torch.zeros(
            batch_size,
            self.hidden_dim[0],
            self.window_length,
            height,
            width,
            device=device,
        )

        for t in range(seq_len):
            input_ = input_x[:, t]

            input_list.append(input_)

            if t % (self.window_length - self.window_stride) == 0:
                input_ = torch.stack(input_list[t:], dim=0)
                input_ = input_.permute(1, 2, 0, 3, 4).contiguous()

            for i in range(self.num_layers):
                if t == 0:
                    c_history[i] = c_t[i]
                else:
                    c_history[i] = torch.cat((c_history[i], c_t[i]), 1)
                if i == 0:
                    h_t[i], c_t[i], memory = self.cell_list[i](
                        input_, h_t[i], c_t[i], memory, c_history[i]
                    )
                else:
                    h_t[i], c_t[i], memory = self.cell_list[i](
                        h_t[i - 1], h_t[i], c_t[i], memory, c_history[i]
                    )

            x_gen = self.conv_output(h_t[self.num_layers - 1]).squeeze(2)
            x_gen = torch.clamp(x_gen, 0, 1)
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1).contiguous()
        loss = self.MSE_criterion(next_frames, targets) + self.L1_criterion(
            next_frames, targets
        )

        return next_frames, loss
