"""Qwen3.5 MoE GGUF->HF weight name map."""
def build(cfg):
    tc = cfg.get_text_config()
    lt = getattr(tc, "layer_types", [])
    m = {"token_embd.weight": "model.embed_tokens.weight",
         "output_norm.weight": "model.norm.weight",
         "output.weight": "lm_head.weight"}
    for i in range(tc.num_hidden_layers):
        p, h = f"blk.{i}", f"model.layers.{i}"
        m[f"{p}.attn_norm.weight"] = f"{h}.input_layernorm.weight"
        m[f"{p}.post_attention_norm.weight"] = f"{h}.post_attention_layernorm.weight"
        if i < len(lt) and lt[i] == "linear_attention":
            la = f"{h}.linear_attn"
            for g, hh in [("attn_qkv","in_proj_qkv"),("attn_gate","in_proj_z"),
                          ("ssm_alpha","in_proj_a"),("ssm_beta","in_proj_b"),
                          ("ssm_conv1d","conv1d"),("ssm_norm","norm"),
                          ("ssm_out","out_proj")]:
                m[f"{p}.{g}.weight"] = f"{la}.{hh}.weight"
            m[f"{p}.ssm_dt.bias"] = f"{la}.dt_bias"
            m[f"{p}.ssm_a"] = f"{la}.A_log"
        else:
            sa = f"{h}.self_attn"
            for g, hh in [("attn_q","q_proj"),("attn_k","k_proj"),
                          ("attn_v","v_proj"),("attn_output","o_proj"),
                          ("attn_q_norm","q_norm"),("attn_k_norm","k_norm")]:
                m[f"{p}.{g}.weight"] = f"{sa}.{hh}.weight"
        mlp = f"{h}.mlp"
        m[f"{p}.ffn_gate_inp.weight"] = f"{mlp}.gate.weight"
        m[f"{p}.ffn_gate_inp_shexp.weight"] = f"{mlp}.shared_expert_gate.weight"
        se = f"{mlp}.shared_expert"
        for g, hh in [("ffn_gate_shexp","gate_proj"),("ffn_up_shexp","up_proj"),
                      ("ffn_down_shexp","down_proj")]:
            m[f"{p}.{g}.weight"] = f"{se}.{hh}.weight"
        for g, hh in [("ffn_down_exps","down_proj"),("ffn_gate_exps","gate_proj"),
                      ("ffn_up_exps","up_proj")]:
            m[f"{p}.{g}.weight"] = f"{mlp}.experts.0.{hh}.weight"
    return m
