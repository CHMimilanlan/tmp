

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


    def _split_heads(self, hidden: Tensor) -> Tensor:
        batch_size, seq_len, _ = hidden.shape

        return hidden.view(
            batch_size,
            seq_len,
            self.config.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _merge_heads(self, hidden: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = hidden.shape

        return hidden.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            self.config.adapter_dim,
        )

    def forward(
        self,
        audio_hidden: Tensor,
        track_xy: Tensor,
        track_valid: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            audio_hidden:
                [B,T_audio,D_model]
                建议传入 LTX 中已经经过 AdaLN 的 norm_ax。

            track_xy:
                [B,T_audio,2]，必须已经与 Audio token 对齐。

            track_valid:
                [B,T_audio]

            attention_mask:
                可选的 additive SDPA mask，
                例如 [B,1,T,T]。
        """
        batch_size, seq_len, model_dim = audio_hidden.shape

        if model_dim != self.config.model_dim:
            raise ValueError(
                f"Expected model_dim={self.config.model_dim}, "
                f"got {model_dim}"
            )

        if track_xy.shape != (batch_size, seq_len, 2):
            raise ValueError(
                f"track_xy must be {(batch_size, seq_len, 2)}, "
                f"got {tuple(track_xy.shape)}"
            )

        if track_valid is None:
            track_valid = torch.isfinite(track_xy).all(dim=-1)
        else:
            track_valid = track_valid.bool()

        track_xy = torch.nan_to_num(
            track_xy,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        )

        track_features, centered_xy = build_track_features(
            track_xy,
            track_valid,
        )

        track_condition = self.track_encoder(
            track_features.to(audio_hidden.dtype)
        )

        # 将绝对轨迹信息加入 Q/K/V 内容
        q = self.q_proj(audio_hidden) + track_condition
        k = self.k_proj(audio_hidden) + track_condition
        v = self.v_proj(audio_hidden) + track_condition

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        matrices, inverse_matrices = (
            build_track_projective_matrices(
                centered_xy=centered_xy,
                track_valid=track_valid,
                matrix_scale=self.config.matrix_scale,
            )
        )

        # 与原始 PRoPE/GTA 相同：
        # Q 使用 P^T，K/V 使用 P^-1
        q = apply_per_token_4x4(
            q,
            matrices.transpose(-1, -2),
        )
        k = apply_per_token_4x4(
            k,
            inverse_matrices,
        )
        v = apply_per_token_4x4(
            v,
            inverse_matrices,
        )

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # 输出使用当前 query token 对应的 P 恢复
        output = apply_per_token_4x4(
            output,
            matrices,
        )

        output = self._merge_heads(output)

        # 无轨迹的 query 位置不注入控制
        output = output * track_valid.unsqueeze(-1).to(output.dtype)

        return self.out_proj(output)
    
    


