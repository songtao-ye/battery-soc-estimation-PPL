import torch
import torch.nn as nn

from models.pos_encoding import positional_encoding

class PMTN(nn.Module):
    def __init__(self, d_model, patch_len, context_length, SOCOCV = False):
        super(PMTN, self).__init__()

        # self.ignored = 2**num_layers-1
        self.d_model = d_model
        self.nvars = 3
        self.SOCOCV = SOCOCV

        # patching
        self.patch_len = patch_len
        self.stride = patch_len
        patch_num = int((context_length - patch_len)/self.stride + 1)
        self.padding_patch = True if patch_len!=self.stride else False
        if self.padding_patch: # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            patch_num += 1

        # embedding
        # self.vars_linear = nn.Linear(self.nvars, 1)
        # self.value_embedding = nn.Linear(self.patch_len, self.d_model)
        self.value_embedding = nn.Conv2d(in_channels=self.patch_len, out_channels=self.d_model, kernel_size=(1, self.nvars))

        # local_encoder
        self.encoder = nn.Linear(self.d_model, self.d_model)
        # self.encoder = nn.Sequential(
        #     nn.Linear(self.d_model, self.d_model*4),
        #     nn.Dropout(0.),
        #     nn.Linear(self.d_model*4, self.d_model)
        # )
        # soc
        # self.conv_soc = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(self.d_model)

        self.lstm_embedding = nn.Linear(1, self.d_model)

        self.lstm = nn.LSTM(self.d_model, self.d_model, 1, batch_first=True)

        self.fc = nn.Linear(self.d_model, 1)
        self.sigmoid = nn.Sigmoid()
        # self.fc2 = nn.Linear(3, 1)


    def forward(self, x, SOCocv):  # x: (bs, seq_len, n_vars)
        # patching + embedding
        x = x.permute(0, 2, 1) # x: (bs, n_vars, seq_len)
        if self.padding_patch:
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # x: (bs, n_vars, patch_num, patch_len)
        x = x.permute(0, 3, 2, 1) # x: (bs, patch_len, patch_num, n_vars)
        u = self.value_embedding(x).squeeze() # x: (bs, d_model, patch_num)
        u = u.permute(0, 2, 1) # u: (bs, patch_num, d_model)
        # encoder
        f = self.encoder(u) # f: (bs, patch_num, d_model)

        # soc
        if self.SOCOCV:
            # with torch.no_grad():
            SOCocv = self.lstm_embedding(SOCocv)
            h0 = SOCocv.unsqueeze(0)
            c0 = torch.zeros_like(h0)

            z, _ = self.lstm(f,(h0,c0)) # out_SOC: (bs, patch_num, d_model)
        else:
            z, _ = self.lstm(f) # out_SOC: (bs, patch_num, d_model)

        # z = self.norm(z)
        pres = self.fc(z).squeeze() # out_SOC: (bs, patch_num, 1)

        pres = self.sigmoid(pres)

        return pres


if __name__ == '__main__':
    input = torch.randn(4, 300, 3) # x: (batch_size, seq_len, channel)
    model = PMTN(d_model=64, patch_len=12, context_length=300)

    SOCocv = torch.randn(4, 1)
    output = model(input, SOCocv)