    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """Split flat [Q|K|V|Z] from MergedColumnParallelLinear."""
        nkh = self.num_k_heads // self.tp_size
        nvh = self.num_v_heads // self.tp_size
        qk = nkh * self.head_k_dim
        vz = nvh * self.head_v_dim
        q, k, v, z = torch.split(mixed_qkvz, [qk,qk,vz,vz], dim=-1)
        b, a = torch.split(mixed_ba, [nvh, nvh], dim=-1)
        q = q.view(q.size(0), nkh, self.head_k_dim)
        k = k.view(k.size(0), nkh, self.head_k_dim)
        v = v.view(v.size(0), nvh, self.head_v_dim)
        z = z.view(z.size(0), nvh, self.head_v_dim)
        return q, k, v, z, b, a

