
        self.out_proj = nn.Linear(
            config.adapter_dim,
            config.model_dim,
            bias=True,
        )
