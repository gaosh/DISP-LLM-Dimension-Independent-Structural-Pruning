from __future__ import annotations

import gc
import math
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from pruning.hypernetwork import hypernetwork
from pruning.pruning_helper import collect_info_reg_llama, help_functions_hn

from models.modeling_qwen3_pruning import Qwen3ForCausalLM as PruneQwen3ForCausalLM
from models.modeling_qwen3_pruned import Qwen3ForCausalLM as PrunedQwen3ForCausalLM

try:
    from accelerate import init_empty_weights
except Exception:
    init_empty_weights = None


def evaluate(model, tokenizer, datasets: str = "wikitext", block_size: int = 4096):
    model.eval()
    model.cuda()

    for dsname in datasets.split(","):
        test_string = load_eval_data(dsname)
        encoded_text = tokenizer.encode(test_string, return_tensors="pt")
        encoded_text = encoded_text[:, : 256 * 2048]

        nlls = 0.0
        toks = 0
        with torch.inference_mode():
            for i in tqdm.tqdm(range(0, encoded_text.shape[1], block_size)):
                inp = encoded_text[:, i : i + block_size]
                model_output = model(inp.cuda())
                logits = model_output.logits if hasattr(model_output, "logits") else model_output
                nll = torch.nn.functional.cross_entropy(
                    logits[0, :-1],
                    inp[0, 1:].to(dtype=torch.long).cuda(),
                    reduction="sum",
                )
                toks += inp.size(1) - 1
                nlls += nll.item()

        ppl = math.exp(nlls / toks)
        print(f"Perplexity on {dsname}: {ppl:.2f}")


def load_eval_data(dataset_name: str) -> str:
    if dataset_name == "wikitext":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(testdata["text"])
    if dataset_name == "ptb":
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        return "\n\n".join(testdata["sentence"])
    if dataset_name == "c4":
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        return " ".join(testdata[:1100]["text"])
    raise ValueError("invalid dataset name (wikitext, ptb, c4 are allowed)")


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    return new_state_dict


def _clone_tensor(t: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    return t.detach().to(device=device, copy=True)


def _set_linear_weight_bias(
    module: torch.nn.Linear,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    module.weight = torch.nn.Parameter(weight.contiguous())
    if bias is None:
        module.bias = None
    else:
        module.bias = torch.nn.Parameter(bias.contiguous())


def _set_embedding_weight(module: torch.nn.Embedding, weight: torch.Tensor) -> None:
    module.weight = torch.nn.Parameter(weight.contiguous())


def _set_norm_weight(module: torch.nn.Module, weight: torch.Tensor) -> None:
    module.weight = torch.nn.Parameter(weight.contiguous())


def _materialize_mlp_layer(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    vectors: list[torch.Tensor],
    vec_idx: int,
    device: str = "cpu",
) -> int:
    in_vector = vectors[vec_idx]
    mid_vector = vectors[vec_idx + 1]
    out_vector = vectors[vec_idx + 2]

    select_index = (in_vector == 1).nonzero(as_tuple=False).squeeze(-1).to(torch.long)
    mid_index = (mid_vector == 1).nonzero(as_tuple=False).squeeze(-1).to(torch.long)
    copy_index = (out_vector == 1).nonzero(as_tuple=False).squeeze(-1).to(torch.long)

    gate_w = _clone_tensor(src_layer.gate_proj.weight.data[mid_index][:, select_index], device=device)
    up_w = _clone_tensor(src_layer.up_proj.weight.data[mid_index][:, select_index], device=device)
    down_w = _clone_tensor(src_layer.down_proj.weight.data[copy_index][:, mid_index], device=device)

    _set_linear_weight_bias(dst_layer.gate_proj, gate_w, None)
    _set_linear_weight_bias(dst_layer.up_proj, up_w, None)
    _set_linear_weight_bias(dst_layer.down_proj, down_w, None)

    return vec_idx + 3


def _materialize_attn_layer(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    vectors: list[torch.Tensor],
    vec_idx: int,
    device: str = "cpu",
) -> int:
    in_vector = vectors[vec_idx]
    out_vector = vectors[vec_idx + 1]

    select_index = (in_vector == 1).nonzero(as_tuple=False).squeeze(-1).to(torch.long)
    copy_index = (out_vector == 1).nonzero(as_tuple=False).squeeze(-1).to(torch.long)

    q_w = _clone_tensor(src_layer.q_proj.weight.data[:, select_index], device=device)
    k_w = _clone_tensor(src_layer.k_proj.weight.data[:, select_index], device=device)
    v_w = _clone_tensor(src_layer.v_proj.weight.data[:, select_index], device=device)
    o_w = _clone_tensor(src_layer.o_proj.weight.data[copy_index, :], device=device)

    q_b = _clone_tensor(src_layer.q_proj.bias.data, device=device) if src_layer.q_proj.bias is not None else None
    k_b = _clone_tensor(src_layer.k_proj.bias.data, device=device) if src_layer.k_proj.bias is not None else None
    v_b = _clone_tensor(src_layer.v_proj.bias.data, device=device) if src_layer.v_proj.bias is not None else None
    o_b = (
        _clone_tensor(src_layer.o_proj.bias.data[copy_index], device=device)
        if src_layer.o_proj.bias is not None
        else None
    )

    _set_linear_weight_bias(dst_layer.q_proj, q_w, q_b)
    _set_linear_weight_bias(dst_layer.k_proj, k_w, k_b)
    _set_linear_weight_bias(dst_layer.v_proj, v_w, v_b)
    _set_linear_weight_bias(dst_layer.o_proj, o_w, o_b)

    _set_norm_weight(dst_layer.q_norm, _clone_tensor(src_layer.q_norm.weight.data, device=device))
    _set_norm_weight(dst_layer.k_norm, _clone_tensor(src_layer.k_norm.weight.data, device=device))

    return vec_idx + 2


def _materialize_decoder_layers(
    src_model: torch.nn.Module,
    dst_model: torch.nn.Module,
    vectors: list[torch.Tensor],
    device: str = "cpu",
) -> None:
    vec_idx = 0
    num_layers = len(src_model.model.layers)

    for layer_id in range(num_layers):
        src_layer = src_model.model.layers[layer_id]
        dst_layer = dst_model.model.layers[layer_id]

        _set_norm_weight(
            dst_layer.input_layernorm,
            _clone_tensor(src_layer.input_layernorm.weight.data, device=device),
        )
        _set_norm_weight(
            dst_layer.post_attention_layernorm,
            _clone_tensor(src_layer.post_attention_layernorm.weight.data, device=device),
        )

        vec_idx = _materialize_attn_layer(src_layer.self_attn, dst_layer.self_attn, vectors, vec_idx, device=device)
        vec_idx = _materialize_mlp_layer(src_layer.mlp, dst_layer.mlp, vectors, vec_idx, device=device)

        # release original dense layer as early as possible
        src_model.model.layers[layer_id] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _patch_saved_modeling_file(out_dir: str, vector_list: list[list[int]]) -> None:
    out_path = Path(out_dir)
    target_file = out_path / "modeling_qwen3_pruned.py"

    # Prefer the local source file if it exists beside this script.
    candidate_sources = [
        Path(__file__).resolve().parent / "models" / "modeling_qwen3_pruned.py",
        Path(__file__).resolve().parent / "modeling_qwen3_pruned.py",
    ]
    src_file = next((p for p in candidate_sources if p.exists()), None)
    if src_file is not None and src_file.resolve() != target_file.resolve():
        shutil.copyfile(src_file, target_file)

    if not target_file.exists():
        return

    text = target_file.read_text()
    if "cfgs = None" not in text:
        return

    text = text.replace("cfgs = None", f"cfgs = {vector_list!r}", 1)
    target_file.write_text(text)


def _instantiate_pruned_model_memory_aware(config, vectors):
    PrunedQwen3ForCausalLM.cfgs = [v.clone().cpu() if torch.is_tensor(v) else v for v in vectors]

    if init_empty_weights is None:
        print("Warning: accelerate.init_empty_weights is unavailable; falling back to normal init.")
        return PrunedQwen3ForCausalLM(config)

    with init_empty_weights():
        pruned_model = PrunedQwen3ForCausalLM(config)
    return pruned_model


def main(
    hf_model: str = "Qwen/Qwen3-8B",
    hn_path: str = "to/your/hn_path/hn-ckpt-final-0.50.pt",
    out_dir: str = "to/your/out_dir",
    p: float = 0.5,
    evaluate_ppl: bool = False,
    block_size: int = 4096,
) -> None:
    # Load the pruning-aware Qwen3 model; low_cpu_mem_usage reduces load-time peaks.
    model = PruneQwen3ForCausalLM.from_pretrained(hf_model, low_cpu_mem_usage=True)
    model.config.use_cache = False
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model)

    reg = collect_info_reg_llama(model, p=p)
    hn_helper = help_functions_hn(reg.structures, constrained=False)

    hn_state_dict = torch.load(hn_path, map_location=torch.device("cpu"))
    hn_state_dict = _strip_module_prefix(hn_state_dict)

    hn = hypernetwork(t_structures=reg.structures)
    hn.load_state_dict(hn_state_dict)
    hn.eval()

    if torch.cuda.is_available():
        hn = hn.cuda()
        with torch.no_grad():
            vectors = hn()
        vectors = [v.detach().cpu() if torch.is_tensor(v) else v for v in vectors]
        hn = hn.cpu()
        torch.cuda.empty_cache()
    else:
        with torch.no_grad():
            vectors = hn()
        vectors = [v.detach().cpu() if torch.is_tensor(v) else v for v in vectors]

    # Keep the same interface as the repo's original prune_model.py.
    hn_helper.set_gate_vectors(model, vectors)

    print(model)
    print(reg.structures)

    # Build a pruned shell without allocating a second dense model.
    config = model.config
    pruned_model = _instantiate_pruned_model_memory_aware(config, vectors)

    # Materialize the unchanged modules first.
    _set_embedding_weight(
        pruned_model.model.embed_tokens,
        _clone_tensor(model.model.embed_tokens.weight.data, device="cpu"),
    )
    _set_norm_weight(
        pruned_model.model.norm,
        _clone_tensor(model.model.norm.weight.data, device="cpu"),
    )
    _set_linear_weight_bias(
        pruned_model.lm_head,
        _clone_tensor(model.lm_head.weight.data, device="cpu"),
        _clone_tensor(model.lm_head.bias.data, device="cpu") if model.lm_head.bias is not None else None,
    )

    # Release the big unchanged pieces from the dense model before layer-by-layer copy.
    model.model.embed_tokens = None
    model.model.norm = None
    model.lm_head = None
    gc.collect()

    _materialize_decoder_layers(model, pruned_model, vectors, device="cpu")

    # Dense model is no longer needed.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sum_params = sum(p.numel() for p in pruned_model.parameters())
    print("number of pruned parameters: %.3f" % (sum_params / 10 ** 6))

    pruned_model.register_for_auto_class("AutoModelForCausalLM")
    pruned_model.save_pretrained(out_dir)
    hf_tokenizer.save_pretrained(out_dir)

    vector_list = [v.tolist() if torch.is_tensor(v) else v for v in vectors]
    _patch_saved_modeling_file(out_dir, vector_list)

    if evaluate_ppl:
        pruned_model.cuda()
        torch.set_float32_matmul_precision("high")
        evaluate(pruned_model, hf_tokenizer, datasets="wikitext,ptb", block_size=block_size)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
