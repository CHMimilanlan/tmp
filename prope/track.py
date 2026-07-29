

class TrackPRoPEAttentionAdapter(nn.Module):
    """
    Parallel Track-PRoPE attention adapter for the LTX audio branch.

    Pipeline:
        audio hidden [B,T,D_model]
          ├─ absolute trajectory embedding
          ├─ q/k/v projection to smaller adapter dimension
          ├─ Track-PRoPE transforms
          ├─ scaled dot-product attention
          ├─ inverse/output transform
          └─ zero-initialized output projection

    该模块输出与输入相同形状：
        [B,T,D_model]
    """

    def __init__(self, config: TrackPRoPEConfig):
        super().__init__()
        self.config = config

        if config.adapter_dim % config.num_heads != 0:
            raise ValueError(
                "adapter_dim must be divisible by num_heads"
            )

        self.head_dim = config.adapter_dim // config.num_heads

        if self.head_dim % 4 != 0:
            raise ValueError(
                "Track-PRoPE head_dim must be divisible by 4. "
                f"Got adapter_dim={config.adapter_dim}, "
                f"num_heads={config.num_heads}, "
                f"head_dim={self.head_dim}."
            )

        # 绝对轨迹特征：
        # x, y, dx, dy, speed, valid
        self.track_encoder = nn.Sequential(
            nn.Linear(6, config.adapter_dim),
            nn.SiLU(),
            nn.Linear(config.adapter_dim, config.adapter_dim),
        )

        self.q_proj = nn.Linear(
            config.model_dim,
            config.adapter_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            config.model_dim,
            config.adapter_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            config.model_dim,
            config.adapter_dim,
            bias=True,
        )

        self.q_norm = nn.RMSNorm(
            config.adapter_dim,
            eps=config.norm_eps,
        )
        self.k_norm = nn.RMSNorm(
            config.adapter_dim,
            eps=config.norm_eps,
        )

        self.out_proj = nn.Linear(
            config.adapter_dim,
            config.model_dim,
            bias=True,
        )

        # 保证加入 Adapter 时，模型初始行为与原始 LTX 完全一致
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
