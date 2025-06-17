import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, x):
        batch, seq, d_model = x.size()

        q = self.q_linear(x).view(batch, self.nhead, seq, self.d_k)               # [b,h,s,d_k]
        k = self.k_linear(x).view(batch, seq, self.nhead, self.d_k).transpose(1,2)  # 效果同上，不改变内存数据
        v = self.v_linear(x).view(batch, seq, self.nhead, self.d_k).transpose(1,2)

        attn = self.attention(q, k, v)  # [b,h,s,d_k]
        attn = attn.transpose(1,2).contiguous().view(batch, seq, d_model)

        x = self.ln1(x + self.out_linear(attn))
        x = self.ln2(x + self.ff(x))
        return x


# ===== Simplified UNet =====
class SimpleUNet(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.bot = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.dec2 = nn.Conv2d(128 + text_dim, 64, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, text_emb):
        # x: [b,3,h,w], text_emb: [b,text_dim]
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))

        b = F.relu(self.bot(e2))

        # Expand text embedding to image shape # [batch, d_model] → [batch, d_model, 1, 1] -> [batch, d_model, H, W]
        text_feat = text_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, b.size(2), b.size(3))

        d2 = F.relu(self.dec2(torch.cat([b, text_feat], dim=1)))
        out = self.dec1(d2)
        return out  # predicted noise


# ===== Full Model Wrapper =====
class DiffusionText2Image(nn.Module):
    def __init__(self, vocab_size, text_dim, image_size, max_len):
        super().__init__()
        self.text_encoder = SimpleTransformerEncoder(vocab_size, text_dim, nhead=4, num_layers=2, max_len=max_len)
        self.unet = SimpleUNet(text_dim)

    def forward(self, image, text):
        text_features = self.text_encoder(text)  # [b,seq,d_model]
        pooled_text = text_features.mean(dim=1)  # [b,d_model]
        out = self.unet(image, pooled_text)
        return out


# ===== Simplified Transformer Encoder (self-implemented) =====
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
        super().__init__()
        # 用来把 离散整数 id（比如单词 id）→ 连续向量 (embedding vector)。
        self.token_emb = nn.Embedding(vocab_size, d_model)  # 输入(batch_size, seq_len)输出(batch_size, seq_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead) for _ in range(num_layers)
        ])

    def forward(self, x):  # x: [batch, seq]
        x = self.token_emb(x) + self.pos_emb[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return x  # [batch, seq, d_model]


# ===== Example Usage =====
vocab_size = 1000
text_dim = 128
image_size = 32
max_len = 16

model = DiffusionText2Image(vocab_size, text_dim, image_size, max_len)
image = torch.randn(4, 3, image_size, image_size)
text = torch.randint(0, vocab_size, (4, max_len))

predicted_noise = model(image, text)
print(predicted_noise.shape)  # [4, 3, 32, 32]
