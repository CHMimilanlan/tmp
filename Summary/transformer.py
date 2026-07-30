from dataclasses import dataclass, field, replace

import torch
from torch import nn

from ltx_core.model.transformer.adaln import adaln_embedding_coefficient
from ltx_core.model.transformer.attention import (
    Attention,
    AttentionCallable,
    AttentionFunction,
    AttentionOps,
    MaskedAttentionCallable,
    MaskedAttentionFunction,
)
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.ops import (
    AdaZeroCallable,
    GatedAttentionCallable,
    PostSACallable,
    PreAttentionCallable,
    PytorchAdaZeroFunction,
    PytorchGatedAttention,
    PytorchPostSAFunction,
    PytorchPreAttention,
)
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgs

from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.model.transformer.summary_tokens import (
    AudioToSummaryAttention,
    AudioToSummaryConfig,
    ICLoraSpatialSummary,
    ICLoraSummaryConfig,
)



@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


@dataclass(frozen=True)
class TransformerOpsConfig:
    """Pluggable ops for :class:`BasicAVTransformerBlock`.
    Use :meth:`from_functions` to construct from enum values or partial overrides
    without spelling out a full :class:`AttentionOps`.
    """

    attention_ops: AttentionOps = field(default_factory=AttentionOps)
    ada_zero_function: AdaZeroCallable = field(default_factory=PytorchAdaZeroFunction)
    post_sa_function: PostSACallable = field(default_factory=PytorchPostSAFunction)

    @classmethod
    def from_functions(
        cls,
        attention: AttentionFunction | AttentionCallable = AttentionFunction.AUTOMATIC,
        masked_attention: MaskedAttentionFunction | MaskedAttentionCallable = MaskedAttentionFunction.AUTOMATIC,
        preattention: PreAttentionCallable | None = None,
        gated_attention: GatedAttentionCallable | None = None,
        ada_zero: AdaZeroCallable | None = None,
        post_sa: PostSACallable | None = None,
    ) -> "TransformerOpsConfig":
        """Build a config from individual functions or enums. Each *None* slot
        falls back to the standard PyTorch implementation."""
        attention_callable = attention.to_callable() if isinstance(attention, AttentionFunction) else attention
        masked_callable = (
            masked_attention.to_callable()
            if isinstance(masked_attention, MaskedAttentionFunction)
            else masked_attention
        )
        attention_ops = AttentionOps(
            attention_function=attention_callable,
            masked_attention_function=masked_callable,
            preattention_function=preattention if preattention is not None else PytorchPreAttention(),
            gated_attention_function=(gated_attention if gated_attention is not None else PytorchGatedAttention()),
        )
        return cls(
            attention_ops=attention_ops,
            ada_zero_function=ada_zero if ada_zero is not None else PytorchAdaZeroFunction(),
            post_sa_function=post_sa if post_sa is not None else PytorchPostSAFunction(),
        )


# Frozen, so safe to share as a default argument across callers that want the
# stock PyTorch ops without explicit construction.
DEFAULT_TRANSFORMER_OPS = TransformerOpsConfig()


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        ops: TransformerOpsConfig | None = None,
        transformer_args: dict = None,
    ):
        super().__init__()

        if ops is None:
            ops = TransformerOpsConfig()
        self.ada_zero_function = ops.ada_zero_function
        self.post_sa_function = ops.post_sa_function
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            video_sst_size = adaln_embedding_coefficient(video.cross_attention_adaln)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(video_sst_size, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            audio_sst_size = adaln_embedding_coefficient(audio.cross_attention_adaln)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(audio_sst_size, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )

        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps

        # =======================
        self.use_track_prope = transformer_args.get("use_track_prope", False)
        self.audio_track_prope = None
        self.audio_model_dim = audio.dim if audio is not None else None
        # =======================

    def initialize_track_prope(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        adapter_dim: int = 512,
        num_heads: int = 8,
        matrix_scale: float = 0.5,
    ) -> None:
        """
        Initialize Track-PRoPE after the base LTX checkpoint has been loaded.

        This method must not be called while the parent transformer is still
        on the meta device.
        """
        if self.audio_model_dim is None:
            # Video-only block.
            return

        if self.audio_track_prope is not None:
            # Avoid accidental reinitialization.
            return

        if not self.use_track_prope:
            return 

        device = torch.device(device)

        if device.type == "meta":
            raise RuntimeError(
                "Track-PRoPE cannot be initialized on the meta device. "
                "Call initialize_track_prope() after load_transformer() returns."
            )

        from ltx_core.model.transformer.track_prope import (
            TrackPRoPEAttentionAdapter,
            TrackPRoPEConfig,
        )

        module = TrackPRoPEAttentionAdapter(
            TrackPRoPEConfig(
                model_dim=self.audio_model_dim,
                adapter_dim=adapter_dim,
                num_heads=num_heads,
                matrix_scale=matrix_scale,
            )
        )

        self.audio_track_prope = module.to(
            device=device,
            dtype=dtype,
        )
        self.use_track_prope = True


    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, scale_shift_indices
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale, shift = (t.squeeze(2) for t in scale_shift_ada_values)
        (gate,) = (t.squeeze(2) for t in gate_ada_values)

        return scale, shift, gate

    def _apply_text_cross_attention(
        self,
        x_normed: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation.
        ``x_normed`` is the RMS-normalized self-attention output produced by
        ``post_sa_function`` -- this method does not normalize again.
        """
        if cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x_normed.shape[0], timestep, slice(6, 9))
            return apply_cross_attention_adaln(
                x_normed,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
            )
        return attn(x_normed, context=context, mask=context_mask)

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)


        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            norm_vx = self.ada_zero_function(vx, self.norm_eps, vscale_msa, vshift_msa)
            del vshift_msa, vscale_msa

            vx_msa_out = self.attn1(
                norm_vx,
                pe=video.positional_embeddings,
                mask=video.self_attention_mask,
                perturbation_mask=video.self_attn_perturbation_mask,
                all_perturbed=video.self_attn_all_perturbed,
            )
            vx, vx_normed = self.post_sa_function(vx, vx_msa_out, None, self.norm_eps, vgate_msa)
            del vgate_msa, norm_vx, vx_msa_out
            vx = vx + self._apply_text_cross_attention(
                vx_normed,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            del vx_normed

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            norm_ax = self.ada_zero_function(ax, self.norm_eps, ascale_msa, ashift_msa)
            del ashift_msa, ascale_msa
            ax_msa_out = self.audio_attn1(
                norm_ax,
                pe=audio.positional_embeddings,
                mask=audio.self_attention_mask,
                perturbation_mask=audio.self_attn_perturbation_mask,
                all_perturbed=audio.self_attn_all_perturbed,
            )

            # ================================================
            # breakpoint()
            if self.use_track_prope:
                track_delta = self.audio_track_prope(
                    audio_hidden=norm_ax,
                    track_xy=audio.track_xy,
                    track_valid=audio.track_valid,
                    attention_mask=audio.self_attention_mask,
                )
                ax_msa_out = ax_msa_out + track_delta
            # ================================================

            
            ax, ax_normed = self.post_sa_function(ax, ax_msa_out, None, self.norm_eps, agate_msa)
            del agate_msa, norm_ax, ax_msa_out
            ax = ax + self._apply_text_cross_attention(
                ax_normed,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            del ax_normed

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            # Snapshot vx/ax before A2V mutates vx; V2A's video keys/values must
            # use the pre-A2V state so direction order doesn't bias the result.
            vx_pre_av = vx
            ax_pre_av = ax
            if run_a2v and not video.cross_attn_skip_all:
                scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_vx_scaled = self.ada_zero_function(vx_pre_av, self.norm_eps, scale_ca_video_a2v, shift_ca_video_a2v)
                del scale_ca_video_a2v, shift_ca_video_a2v

                scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_ax_scaled = self.ada_zero_function(ax_pre_av, self.norm_eps, scale_ca_audio_a2v, shift_ca_audio_a2v)
                del scale_ca_audio_a2v, shift_ca_audio_a2v
                vx = vx + (
                    self.audio_to_video_attn(
                        a2v_vx_scaled,
                        context=a2v_ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                    * video.cross_attn_perturbation_mask
                )
                del gate_out_a2v, a2v_vx_scaled, a2v_ax_scaled

            if run_v2a and not audio.cross_attn_skip_all:
                scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_ax_scaled = self.ada_zero_function(ax_pre_av, self.norm_eps, scale_ca_audio_v2a, shift_ca_audio_v2a)
                del scale_ca_audio_v2a, shift_ca_audio_v2a
                scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_vx_scaled = self.ada_zero_function(vx_pre_av, self.norm_eps, scale_ca_video_v2a, shift_ca_video_v2a)
                del scale_ca_video_v2a, shift_ca_video_v2a
                ax = ax + (
                    self.video_to_audio_attn(
                        v2a_ax_scaled,
                        context=v2a_vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                    * audio.cross_attn_perturbation_mask
                )
                del gate_out_v2a, v2a_vx_scaled, v2a_ax_scaled
            del vx_pre_av, ax_pre_av

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = self.ada_zero_function(vx, self.norm_eps, vscale_mlp, vshift_mlp)
            vx = vx + self.ff(vx_scaled) * vgate_mlp

            del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = self.ada_zero_function(ax, self.norm_eps, ascale_mlp, ashift_mlp)
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def apply_cross_attention_adaln(
    x_normed: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor,
    context_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply query/key AdaLN modulation then cross-attention.
    ``x_normed`` is already RMS-normalized by ``post_sa_function``; this only
    applies the affine (scale/shift) modulation, so the normalization is not
    repeated here.
    """
    batch_size = x_normed.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x_normed.device, dtype=x_normed.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = x_normed * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate



class BasicAVTransformerBlockWithTrackSummary(BasicAVTransformerBlock):
    """BasicAVTransformerBlock extended with IC-LoRA-aware Summary Tokens.

    The parent class still owns every original LTX attention, FFN and AdaLN
    parameter.  This subclass only adds a parallel spatial-summary residual to
    the existing Video-to-Audio branch.

    The complete IC-LoRA video sequence is expected in the official order:

        [target tokens, appended reference tokens]
        [6144 target, 1536 reference] -> 7680 total

    ``ICLoraSummaryConfig.source_mode`` chooses whether the Summary module uses:

    - explicit ``audio.track_xy``;
    - appended reference tokens without ``track_xy``;
    - reference tokens to guide a second read from target tokens (recommended);
    - both explicit coordinates and reference tokens.

    ``use_audio_to_summary_attention`` independently controls the optional
    second layer where Audio hidden states are Query and Summary Tokens are
    Key/Value.  When disabled, Summary Tokens are projected directly to the
    audio hidden dimension.
    """

    def __init__(
        self,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        ops: TransformerOpsConfig | None = None,
        transformer_args: dict = None,
        *,
        ic_lora_summary_config: ICLoraSummaryConfig | None = None,
        audio_to_summary_config: AudioToSummaryConfig | None = None,
    ) -> None:
        super().__init__(
            video=video,
            audio=audio,
            rope_type=rope_type,
            norm_eps=norm_eps,
            ops=ops,
            transformer_args=transformer_args,
        )
        self.use_ic_lora_summary = transformer_args.get("use_ic_lora_summary", False)
        self.use_audio_to_summary_attention = transformer_args.get("use_audio_summary", False)

        self.last_reference_summary_attention: torch.Tensor | None = None
        self.last_target_summary_attention: torch.Tensor | None = None
        self.last_inferred_track_xy: torch.Tensor | None = None
        self.last_audio_to_summary_attention: torch.Tensor | None = None

        if self.use_audio_to_summary_attention and not self.use_ic_lora_summary:
            raise ValueError(
                "Audio-to-Summary attention requires use_ic_lora_summary=True"
            )

        # =======================
        # The extra Track-Summary submodules are NOT created here. Like
        # Track-PRoPE, the base block is built without them so the pretrained
        # LTX checkpoint can be loaded cleanly. They are created later by
        # calling initialize_track_summary() once the checkpoint is loaded and
        # the block is off the meta device.
        #
        # We only stash the dimensions and optional config overrides needed for
        # deferred construction.
        # =======================
        self._summary_video_dim = video.dim if video is not None else None
        self._summary_audio_dim = audio.dim if audio is not None else None
        self._summary_audio_heads = audio.heads if audio is not None else None
        self._ic_lora_summary_config_arg = ic_lora_summary_config
        self._audio_to_summary_config_arg = audio_to_summary_config

        self.ic_lora_summarizer = None
        self.summary_to_audio = None
        self.audio_to_summary_attn = None
        self.summary_scale = None
        self.ic_lora_summary_config = None
        self.audio_to_summary_config = None

    def initialize_track_summary(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """
        Initialize IC-LoRA Track-Summary modules after the base LTX checkpoint
        has been loaded.

        This method must not be called while the parent transformer is still on
        the meta device. It is controlled by ``use_ic_lora_summary`` and
        ``use_audio_summary`` (stored as ``use_audio_to_summary_attention``).
        """
        if not self.use_ic_lora_summary:
            return

        if self.ic_lora_summarizer is not None:
            # Avoid accidental reinitialization.
            return

        device = torch.device(device)

        if device.type == "meta":
            raise RuntimeError(
                "Track-Summary cannot be initialized on the meta device. "
                "Call initialize_track_summary() after load_transformer() returns."
            )

        if self._summary_video_dim is None or self._summary_audio_dim is None:
            raise ValueError(
                "IC-LoRA Summary requires both video and audio TransformerConfig"
            )

        ic_lora_summary_config = self._ic_lora_summary_config_arg
        if ic_lora_summary_config is None:
            ic_lora_summary_config = ICLoraSummaryConfig(
                video_dim=self._summary_video_dim
            )
        elif ic_lora_summary_config.video_dim != self._summary_video_dim:
            raise ValueError(
                "ic_lora_summary_config.video_dim must match video.dim, got "
                f"{ic_lora_summary_config.video_dim} vs {self._summary_video_dim}"
            )

        self.ic_lora_summary_config = ic_lora_summary_config
        self.ic_lora_summarizer = ICLoraSpatialSummary(ic_lora_summary_config)

        if self.use_audio_to_summary_attention:
            audio_to_summary_config = self._audio_to_summary_config_arg
            if audio_to_summary_config is None:
                audio_to_summary_config = AudioToSummaryConfig(
                    audio_dim=self._summary_audio_dim,
                    summary_dim=ic_lora_summary_config.summary_dim,
                    num_heads=self._summary_audio_heads,
                )
            elif audio_to_summary_config.audio_dim != self._summary_audio_dim:
                raise ValueError(
                    "audio_to_summary_config.audio_dim must match audio.dim, got "
                    f"{audio_to_summary_config.audio_dim} vs {self._summary_audio_dim}"
                )
            elif (
                audio_to_summary_config.summary_dim
                != ic_lora_summary_config.summary_dim
            ):
                raise ValueError(
                    "audio_to_summary_config.summary_dim must match "
                    "ic_lora_summary_config.summary_dim"
                )
            self.audio_to_summary_config = audio_to_summary_config
            self.audio_to_summary_attn = AudioToSummaryAttention(
                audio_to_summary_config
            )
            self.summary_to_audio = None
        else:
            self.audio_to_summary_config = None
            self.audio_to_summary_attn = None
            self.summary_to_audio = nn.Linear(
                ic_lora_summary_config.summary_dim,
                self._summary_audio_dim,
            )
            # Preserve the pretrained block exactly at initialization.
            nn.init.zeros_(self.summary_to_audio.weight)
            nn.init.zeros_(self.summary_to_audio.bias)

        # Keep this at one: the zero-initialized output projection then receives
        # gradients immediately, while the new residual initially remains zero.
        self.summary_scale = nn.Parameter(torch.tensor(1.0))

        # Move the newly created modules onto the requested device/dtype so they
        # match the already-loaded base block.
        self.ic_lora_summarizer = self.ic_lora_summarizer.to(
            device=device, dtype=dtype
        )
        if self.audio_to_summary_attn is not None:
            self.audio_to_summary_attn = self.audio_to_summary_attn.to(
                device=device, dtype=dtype
            )
        if self.summary_to_audio is not None:
            self.summary_to_audio = self.summary_to_audio.to(
                device=device, dtype=dtype
            )
        self.summary_scale = nn.Parameter(
            self.summary_scale.data.to(device=device, dtype=dtype)
        )

    def set_summary_source_mode(self, source_mode: str) -> None:
        """
        Update ``source_mode`` on the IC-LoRA Summary config after the block has
        been initialized.

        ``source_mode`` controls where Summary Tokens come from. Supported
        values: ``track_xy``, ``reference_tokens``, ``reference_guided_target``
        and ``hybrid``.

        This must be called after ``initialize_track_summary()`` so the
        ``ic_lora_summarizer`` and its config already exist.
        """
        valid_modes = {
            "track_xy",
            "reference_tokens",
            "reference_guided_target",
            "hybrid",
        }
        if source_mode not in valid_modes:
            raise ValueError(
                f"Unsupported source_mode: {source_mode!r}. "
                f"Expected one of {sorted(valid_modes)}."
            )

        if not self.use_ic_lora_summary:
            return

        if self.ic_lora_summarizer is None or self.ic_lora_summary_config is None:
            raise RuntimeError(
                "set_summary_source_mode() must be called after "
                "initialize_track_summary(): the IC-LoRA Summary module has "
                "not been created yet."
            )
        # ICLoraSummaryConfig is a frozen dataclass, so in-place assignment
        # raises FrozenInstanceError. Build a new config via dataclasses.replace
        # and rebind it on both the block and the summarizer.
        new_config = replace(
            self.ic_lora_summary_config,
            source_mode=source_mode,
        )
        self.ic_lora_summary_config = new_config
        self.ic_lora_summarizer.config = new_config

    def _compute_ic_lora_summary_delta(
        self,
        *,
        video: TransformerArgs,
        audio: TransformerArgs,
        video_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
        num_audio_tokens: int,
    ) -> torch.Tensor | None:
        if not self.use_ic_lora_summary or self.ic_lora_summarizer is None:
            return None

        output = self.ic_lora_summarizer(
            video_hidden=video_hidden,
            target_audio_tokens=num_audio_tokens,
            track_xy=getattr(audio, "track_xy", None),
            track_valid=getattr(audio, "track_valid", None),
            audio_time=getattr(audio, "audio_time", None),
            # Normal LTX IC-LoRA does not need this field because the reference
            # tokens are already appended to video_hidden.  It remains available
            # for experiments that keep the reference sequence separate.
            explicit_reference_hidden=getattr(video, "reference_tokens", None),
            target_token_count=getattr(video, "target_token_count", None),
            reference_token_count=getattr(video, "reference_token_count", None),
            reference_valid=getattr(video, "reference_token_valid", None),
        )
        if output is None:
            self.last_reference_summary_attention = None
            self.last_target_summary_attention = None
            self.last_inferred_track_xy = None
            self.last_audio_to_summary_attention = None
            return None

        if self.ic_lora_summary_config.store_attention_maps:
            self.last_reference_summary_attention = (
                output.reference_attention.detach()
                if output.reference_attention is not None
                else None
            )
            self.last_target_summary_attention = (
                output.target_attention.detach()
                if output.target_attention is not None
                else None
            )
            self.last_inferred_track_xy = (
                output.inferred_track_xy.detach()
                if output.inferred_track_xy is not None
                else None
            )
        else:
            self.last_reference_summary_attention = None
            self.last_target_summary_attention = None
            self.last_inferred_track_xy = None

        audio_time = getattr(audio, "audio_time", None)
        if self.use_audio_to_summary_attention:
            if self.audio_to_summary_attn is None:
                raise RuntimeError(
                    "use_audio_to_summary_attention=True but module is missing"
                )
            summary_delta, attention = self.audio_to_summary_attn(
                audio_hidden=audio_hidden,
                summary_tokens=output.summary_tokens,
                summary_valid=output.summary_valid,
                audio_time=audio_time,
                summary_time=audio_time,
            )
            if self.audio_to_summary_config.store_attention_map:
                self.last_audio_to_summary_attention = attention.detach()
            else:
                self.last_audio_to_summary_attention = None
        else:
            if self.summary_to_audio is None:
                raise RuntimeError("Direct Summary projection is missing")
            summary_delta = self.summary_to_audio(output.summary_tokens)
            summary_delta = summary_delta * output.summary_valid.unsqueeze(-1).to(
                summary_delta.dtype
            )
            self.last_audio_to_summary_attention = None

        return summary_delta * self.summary_scale.to(
            device=summary_delta.device,
            dtype=summary_delta.dtype,
        )

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """Run the original LTX block and add IC-LoRA Summary in its V2A branch.

        Most of this method intentionally follows ``BasicAVTransformerBlock`` so
        inherited layers, checkpoint keys, LoRA target names and AdaLN behaviour
        remain unchanged.  The only semantic change is marked ``IC-LORA SUMMARY``.
        """

        # breakpoint()        
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0
        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            norm_vx = self.ada_zero_function(vx, self.norm_eps, vscale_msa, vshift_msa)
            del vshift_msa, vscale_msa

            vx_msa_out = self.attn1(
                norm_vx,
                pe=video.positional_embeddings,
                mask=video.self_attention_mask,
                perturbation_mask=video.self_attn_perturbation_mask,
                all_perturbed=video.self_attn_all_perturbed,
            )
            vx, vx_normed = self.post_sa_function(vx, vx_msa_out, None, self.norm_eps, vgate_msa)
            del vgate_msa, norm_vx, vx_msa_out
            vx = vx + self._apply_text_cross_attention(
                vx_normed,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            del vx_normed

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )
            norm_ax = self.ada_zero_function(ax, self.norm_eps, ascale_msa, ashift_msa)
            del ashift_msa, ascale_msa

            ax_msa_out = self.audio_attn1(
                norm_ax,
                pe=audio.positional_embeddings,
                mask=audio.self_attention_mask,
                perturbation_mask=audio.self_attn_perturbation_mask,
                all_perturbed=audio.self_attn_all_perturbed,
            )

            if (
                self.use_track_prope
                and self.audio_track_prope is not None
                and getattr(audio, "track_xy", None) is not None
            ):
                track_delta = self.audio_track_prope(
                    audio_hidden=norm_ax,
                    track_xy=audio.track_xy,
                    track_valid=getattr(audio, "track_valid", None),
                    attention_mask=audio.self_attention_mask,
                )
                ax_msa_out = ax_msa_out + track_delta

            ax, ax_normed = self.post_sa_function(ax, ax_msa_out, None, self.norm_eps, agate_msa)
            del agate_msa, norm_ax, ax_msa_out
            ax = ax + self._apply_text_cross_attention(
                ax_normed,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            del ax_normed

        # Audio <-> Video cross attention.
        if run_a2v or run_v2a:
            # Both directions use the same pre-AV snapshot, matching the parent.
            vx_pre_av = vx
            ax_pre_av = ax

            if run_a2v and not video.cross_attn_skip_all:
                scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_vx_scaled = self.ada_zero_function(
                    vx_pre_av, self.norm_eps, scale_ca_video_a2v, shift_ca_video_a2v
                )
                del scale_ca_video_a2v, shift_ca_video_a2v

                scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_ax_scaled = self.ada_zero_function(
                    ax_pre_av, self.norm_eps, scale_ca_audio_a2v, shift_ca_audio_a2v
                )
                del scale_ca_audio_a2v, shift_ca_audio_a2v

                vx = vx + (
                    self.audio_to_video_attn(
                        a2v_vx_scaled,
                        context=a2v_ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                    * video.cross_attn_perturbation_mask
                )
                del gate_out_a2v, a2v_vx_scaled, a2v_ax_scaled

            if run_v2a and not audio.cross_attn_skip_all:
                scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_ax_scaled = self.ada_zero_function(
                    ax_pre_av, self.norm_eps, scale_ca_audio_v2a, shift_ca_audio_v2a
                )
                del scale_ca_audio_v2a, shift_ca_audio_v2a

                scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_vx_scaled = self.ada_zero_function(
                    vx_pre_av, self.norm_eps, scale_ca_video_v2a, shift_ca_video_v2a
                )
                del scale_ca_video_v2a, shift_ca_video_v2a

                v2a_delta = self.video_to_audio_attn(
                    v2a_ax_scaled,
                    context=v2a_vx_scaled,
                    pe=audio.cross_positional_embeddings,
                    k_pe=video.cross_positional_embeddings,
                )

                # ================= IC-LORA SUMMARY =================
                summary_delta = self._compute_ic_lora_summary_delta(
                    video=video,
                    audio=audio,
                    video_hidden=v2a_vx_scaled,
                    audio_hidden=v2a_ax_scaled,
                    num_audio_tokens=v2a_ax_scaled.shape[1],
                )
                if summary_delta is not None:
                    v2a_delta = v2a_delta + summary_delta
                # =================================================

                # The summary branch shares the original V2A AdaLN gate and the
                # perturbation mask.  This keeps denoising-time conditioning and
                # perturbation semantics identical to the parent block.
                ax = ax + (
                    v2a_delta
                    * gate_out_v2a
                    * audio.cross_attn_perturbation_mask
                )
                del gate_out_v2a, v2a_vx_scaled, v2a_ax_scaled, v2a_delta

            del vx_pre_av, ax_pre_av

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = self.ada_zero_function(vx, self.norm_eps, vscale_mlp, vshift_mlp)
            vx = vx + self.ff(vx_scaled) * vgate_mlp
            del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = self.ada_zero_function(ax, self.norm_eps, ascale_mlp, ashift_mlp)
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp
            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return (
            replace(video, x=vx) if video is not None else None,
            replace(audio, x=ax) if audio is not None else None,
        )
