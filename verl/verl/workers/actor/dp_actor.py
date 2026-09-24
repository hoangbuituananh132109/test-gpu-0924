# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    # FRONTIER_GRAD_INTERACTION_DIAGNOSTICS_V1
    def _frontier_gradient_interaction(self, micro_batch, temperature, loss_scale_factor):
        """Measure two native branches without stepping or retaining the train graph.

        This runs after the real combined optimizer step. It recomputes the same
        frozen rollout microbatch twice at the post-update parameters, restores RNG,
        and leaves .grad empty. The selected FSDP parameter is fixed by model order
        and size bounds so the reported cosine is a stable parameter-subset probe.
        """
        cfg = self.config.get("frontier_grad_diagnostics", {})
        min_numel = int(cfg.get("min_param_numel", 100000))
        max_numel = int(cfg.get("max_param_numel", 80000000))
        candidates = [
            (name, param) for name, param in self.actor_module.named_parameters()
            if param.requires_grad and min_numel <= param.numel() <= max_numel
        ]
        if not candidates:
            raise RuntimeError(
                f"gradient diagnostic found no parameter in [{min_numel}, {max_numel}] elements"
            )
        parameter_name, parameter = candidates[-1]
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        response_mask = model_inputs.get("frontier_policy_loss_mask", model_inputs["response_mask"])
        opd_advantages = model_inputs["advantages"]
        grpo_advantages = model_inputs.get("frontier_grpo_advantages")
        if grpo_advantages is None or opd_advantages.dim() != 3:
            raise RuntimeError("gradient diagnostic requires native OPD [B,T,K] and GRPO [B,T]")
        top_k = opd_advantages.shape[-1]
        top_k_ids = model_inputs.get("union_top_k_ids", model_inputs.get("student_top_k_ids"))
        if top_k_ids is None:
            raise RuntimeError("gradient diagnostic has no student/union top-k token IDs")
        policy_loss_fn = get_policy_loss_fn(self.config.policy_loss.get("loss_mode", "vanilla"))
        format_mask = model_inputs.get("format_mask")

        def branch_backward(branch):
            self.actor_optimizer.zero_grad(set_to_none=True)
            _, sampled_log_prob, _, topk_log_probs = self._forward_micro_batch(
                model_inputs,
                temperature=temperature,
                calculate_entropy=False,
                top_k=top_k,
                student_top_k_ids=top_k_ids,
            )
            if branch == "opd":
                loss, _ = policy_loss_fn(
                    old_log_prob=topk_log_probs.detach(),
                    log_prob=topk_log_probs,
                    advantages=opd_advantages,
                    response_mask=response_mask,
                    loss_agg_mode=self.config.loss_agg_mode,
                    config=self.config,
                    rollout_is_weights=None,
                    format_mask=format_mask,
                )
            else:
                loss, _ = policy_loss_fn(
                    old_log_prob=sampled_log_prob.detach(),
                    log_prob=sampled_log_prob,
                    advantages=grpo_advantages,
                    response_mask=response_mask,
                    loss_agg_mode=self.config.loss_agg_mode,
                    config=self.config,
                    rollout_is_weights=None,
                    format_mask=format_mask,
                )
            (loss * loss_scale_factor).backward()
            if parameter.grad is None:
                raise RuntimeError(f"gradient diagnostic parameter has no {branch} gradient: {parameter_name}")
            return loss.detach(), parameter.grad.detach().float().clone()

        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            opd_loss, g_opd = branch_backward("opd")
            grpo_loss, g_grpo = branch_backward("grpo")

        local = torch.stack(
            [g_grpo.square().sum(), g_opd.square().sum(), (g_grpo * g_opd).sum()]
        ).to(dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.SUM)
        grpo_sq, opd_sq, dot = local
        eps = torch.finfo(torch.float64).eps
        grpo_norm = grpo_sq.clamp_min(0).sqrt()
        opd_norm = opd_sq.clamp_min(0).sqrt()
        cosine = dot / (grpo_norm * opd_norm).clamp_min(eps)
        combined_norm = (grpo_sq + opd_sq + 2 * dot).clamp_min(0).sqrt()
        del g_grpo, g_opd
        self.actor_optimizer.zero_grad(set_to_none=True)
        return {
            "frontier_grad/grpo_norm": grpo_norm.item(),
            "frontier_grad/opd_norm": opd_norm.item(),
            "frontier_grad/norm_ratio_grpo_opd": (grpo_norm / opd_norm.clamp_min(eps)).item(),
            "frontier_grad/cosine": cosine.item(),
            "frontier_grad/dot": dot.item(),
            "frontier_grad/combined_norm": combined_norm.item(),
            "frontier_grad/grpo_loss_probe": grpo_loss.item(),
            "frontier_grad/opd_loss_probe": opd_loss.item(),
            "frontier_grad/parameter_numel": float(parameter.numel()),
            "frontier_grad/measured": 1.0,
        }, parameter_name

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, top_k=0, student_top_k_ids=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_ids: # (bs, response_len, k)
            topk_log_probs: # (bs, response_len, k)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk_ids = None
            topk_log_probs = None
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                need_logits = top_k > 0
                fused_topk_ids = None
                use_fused_selected = self.use_fused_kernels and need_logits and student_top_k_ids is not None
                if use_fused_selected:
                    if student_top_k_ids.ndim != 3:
                        raise RuntimeError("fused OPD actor requires padded student_top_k_ids [B,T,K]")
                    if student_top_k_ids.shape[1] != seqlen:
                        full_student_top_k_ids = torch.zeros(
                            (batch_size, seqlen, top_k), dtype=student_top_k_ids.dtype,
                            device=student_top_k_ids.device,
                        )
                        full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                        student_top_k_ids = full_student_top_k_ids
                    fused_topk_ids = student_top_k_ids.view(-1, top_k)[indices]
                    fused_selected_ids = torch.cat(
                        [input_ids_rmpad_rolled.squeeze(0).unsqueeze(-1), fused_topk_ids], dim=-1
                    ).unsqueeze(0)
                    extra_args["frontier_selected_ids"] = fused_selected_ids
                elif self.use_fused_kernels and need_logits:
                    # FRONTIER_FUSED_TOPK_ACTOR_V2: discovery is no-grad and
                    # needs ordinary logits to choose the student's own top-k.
                    extra_args["frontier_force_logits"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels and not (need_logits and not use_fused_selected):
                    fused_log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                    if need_logits:
                        # FRONTIER_FUSED_TOPK_ACTOR_V1: column 0 is the sampled token;
                        # columns 1..K are the unchanged OPD candidate token log-probs.
                        log_probs = fused_log_probs[:, 0]
                        topk_log_probs = fused_log_probs[:, 1:]
                        topk_ids = fused_topk_ids
                    else:
                        log_probs = fused_log_probs

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    # FRONTIER_GATHERED_LOGPROBS_V1: exact selected-token
                    # log-softmax without materializing [tokens, vocab] log_probs_all.
                    need_topk = top_k > 0
                    if need_topk:
                        log_normalizer = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        target_logits = logits_rmpad.gather(
                            dim=-1, index=input_ids_rmpad_rolled.unsqueeze(-1)
                        )
                        log_probs = (target_logits - log_normalizer).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    
                    if need_topk and not use_fused_selected:
                        if student_top_k_ids is not None:
                             # Use specific IDs (from rollout)
                             topk_ids = student_top_k_ids
                             if student_top_k_ids.ndim == 3: # (bsz, seqlen, k)
                                 # We are in rmpad mode, but student_top_k_ids is padded 3D tensor
                                 # We need to extract the relevant tokens aligning with input_ids_rmpad_rolled
                                 
                                 # This is tricky because student_top_k_ids is shaped (batch, seq, k)
                                 # and logits_rmpad is (total_nnz, vocab)
                                 # We need to flatten student_top_k_ids to (total_nnz, k) using indices
                                 
                                 # Re-use the indices computed from unpad_input
                                 # indices: (total_nnz,) 
                                 # student_top_k_ids: (batch, seq, k)
                                 
                                 # 1. If student_top_k_ids only covers the response, pad it to match full sequence length
                                 if student_top_k_ids.shape[1] != seqlen:
                                     full_student_top_k_ids = torch.zeros((batch_size, seqlen, top_k), 
                                                                         dtype=student_top_k_ids.dtype, 
                                                                         device=student_top_k_ids.device)
                                     full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                                     student_top_k_ids = full_student_top_k_ids

                                 # 2. Flatten student_top_k_ids to (batch*seq, k)
                                 flat_ids = student_top_k_ids.view(-1, top_k)
                                 
                                 # 3. Select using indices
                                 # Note: indices are from attention_mask, which aligns with how logits_rmpad represents data
                                 topk_ids_rmpad = flat_ids[indices] # (total_nnz, k)
                                 
                                 # If 'student_top_k_ids' in batch has shape (batch, seq_len, k), then:
                                 topk_ids = topk_ids_rmpad
                                 
                             else:
                                 # If it's already flattened? Unlikely.
                                 pass

                        else:
                             # Legacy/Resample behavior
                             _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)

                        selected_topk_logits = logits_rmpad.gather(dim=-1, index=topk_ids)
                        topk_log_probs = selected_topk_logits - log_normalizer

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if top_k > 0:
                         topk_ids = gather_outputs_and_unpad(
                            topk_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                         topk_log_probs = gather_outputs_and_unpad(
                            topk_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                
                if top_k > 0:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                if top_k > 0:
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0
                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    # FRONTIER_GATHERED_LOGPROBS_V1: padded equivalent.
                    need_topk = top_k > 0
                    if need_topk:
                        log_normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
                        target_logits = logits.gather(
                            dim=-1, index=micro_batch["responses"].unsqueeze(-1)
                        )
                        log_probs = (target_logits - log_normalizer).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                             # Ensure shape alignment if needed, but for non-rmpad (bsz, seq, k) should match logits (bsz, seq, vocab) dim 0,1
                        else:
                             _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
                        
                        selected_topk_logits = logits.gather(dim=-1, index=topk_ids)
                        topk_log_probs = selected_topk_logits - log_normalizer

            return entropy, log_probs, topk_ids, topk_log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability for specific token ids
        Args:
            data (DataProto): a DataProto containing input_ids, attention_mask, position_ids, responses, 
                             and target_ids (batch, response_len, k) in batch
        Returns:
            torch.Tensor: (batch, response_len, k) log probs for target_ids
        """
        # set to eval
        self.actor_module.eval()

        target_ids = data.batch["target_ids"]
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "target_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_log_probs_lst = []
        top_k = target_ids.shape[-1]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_target_ids = model_inputs["target_ids"]
            with torch.no_grad():
                # We reuse _forward_micro_batch. It returns (entropy, log_probs, topk_ids, topk_log_probs)
                _, _, _, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False, 
                    top_k=top_k, student_top_k_ids=mb_target_ids
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # topk_log_probs = topk_log_probs.to("cpu")
            topk_log_probs_lst.append(topk_log_probs)

        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        """Compute the distillation reward (rm_scores) on GPU
        Args:
            data (DataProto): containing all necessary tensors for distillation reward calculation
        Returns:
            DataProto: containing rm_scores and other updated tensors (e.g., union_ids)
        """
        # Set to eval mode for forward passes
        self.actor_module.eval()

        # 1. Extract parameters from meta_info
        top_k = data.meta_info.get("log_prob_top_k", 0)
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        kl_estimator = data.meta_info.get("kl_estimator", "k1")
        reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")  # "student_p", "teacher_p", or "none"
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # 2. Compute Student Log Probs on Teacher IDs if needed
        # (This replaces the previous call to compute_log_probs_for_ids in ray_trainer)
        S_on_T = None
        if strategy in ["only_tch", "intersection", "union", "union-intersection"]:
            target_ids = data.batch["teacher_top_k_ids"]
            
            # Select keys for micro-batching
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
            
            # We need to pass target_ids to _forward_micro_batch, but since we are micro-batching, 
            # we should split target_ids as well.
            mb_data = data.select(batch_keys=select_keys + ["teacher_top_k_ids"], 
                                 non_tensor_batch_keys=non_tensor_select_keys)
            
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(mb_data, max_token_len=max_token_len)
            else:
                micro_batches = mb_data.split(micro_batch_size)

            S_on_T_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                mb_target_ids = model_inputs["teacher_top_k_ids"]
                with torch.no_grad():
                    _, _, _, topk_log_probs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False, 
                        top_k=top_k, student_top_k_ids=mb_target_ids
                    )
                S_on_T_lst.append(topk_log_probs)

            S_on_T = torch.concat(S_on_T_lst, dim=0)
            if use_dynamic_bsz:
                S_on_T = restore_dynamic_batch(S_on_T, batch_idx_list)
        
        # 3. Compute rm_scores on GPU
        # Move all necessary tensors to GPU (they should already be there if passed from fsdp_workers)
        device = get_device_id()
        S_ids = data.batch["student_top_k_ids"].to(device)
        S_logp = data.batch["student_top_k_log_probs"].to(device)
        T_on_S = data.batch["teacher_on_student_log_probs"].to(device)
        
        T_ids = data.batch.get("teacher_top_k_ids", None)
        if T_ids is not None: T_ids = T_ids.to(device)
        T_logp = data.batch.get("teacher_top_k_log_probs", None)
        if T_logp is not None: T_logp = T_logp.to(device)
        overlap_mask = data.batch.get("overlap_mask", None)
        if overlap_mask is not None: overlap_mask = overlap_mask.to(device)

        def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
            """Compute weights for reward calculation.
            
            Args:
                S_logp: Student log probabilities (batch, seq, K)
                T_logp: Teacher log probabilities (batch, seq, K)
                valid_mask: Boolean mask for valid tokens (batch, seq, K)
                weight_mode: "student_p", "teacher_p", or "none"
                normalize: If True, apply softmax normalization across K dim.
                          If False, use raw probabilities (masked by valid_mask).
            
            Returns:
                Weights (batch, seq, K)
            """
            if weight_mode == "student_p":
                log_probs = S_logp
            elif weight_mode == "teacher_p":
                log_probs = T_logp
            elif weight_mode == "none":
                # 对于"none"模式，使用均匀分布
                log_probs = torch.zeros_like(S_logp)
            else:
                raise ValueError(f"Unknown reward_weight_mode: {weight_mode}")
            
            log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, -float('inf')))
            
            if normalize:
                norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
                weights = torch.exp(norm_log_weights)
            else:
                weights = torch.exp(log_probs)
            
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            
            return weights

        res_tensors = {}
        
        if strategy == "only_stu":
            kl_val = S_logp - T_on_S
            valid_mask = torch.ones_like(S_logp, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "only_tch":
            kl_val = S_on_T - T_logp
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_on_T, T_logp, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            res_tensors["union_top_k_ids"] = T_ids
            
        elif strategy == "intersection":
            valid_mask = overlap_mask.bool()
            kl_val = S_logp - T_on_S
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "union":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)
            
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                torch.ones_like(S_ids, dtype=torch.bool),
                ~T_in_S
            ], dim=-1)
            
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
        
        elif strategy == "union-intersection":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)

            S_in_T = overlap_mask.bool().to(device)
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                ~S_in_T,    # S_ids is valid if not in T
                ~T_in_S     # T_ids is valid if not in S
            ], dim=-1)
                
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=False)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
            
        res_tensors["rm_scores"] = rm_scores
        return DataProto.from_dict(tensors=res_tensors)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        top_k = data.meta_info.get("top_k", 0)
        print(f"In compute_log_prob, top_k: {top_k}")
        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_ids, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, top_k=top_k
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # log_probs = log_probs.to("cpu")
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                # entropy = entropy.to("cpu")
                entropy_lst.append(entropy)
            if top_k > 0:
                # topk_ids = topk_ids.to("cpu")
                # topk_log_probs = topk_log_probs.to("cpu")
                topk_ids_lst.append(topk_ids)
                topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        topk_ids_tensor = None
        topk_log_probs_tensor = None
        if top_k > 0:
            topk_ids_tensor = torch.concat(topk_ids_lst, dim=0)
            topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if top_k > 0:
                topk_ids_tensor = restore_dynamic_batch(topk_ids_tensor, batch_idx_list)
                topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return log_probs, entropys, topk_ids_tensor, topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        if "format_mask" in data.batch.keys():
            select_keys.append("format_mask") # (bsz, 1)

        # FRONTIER_TWO_BRANCH_V1 tensors and row-wise logging diagnostics.
        if "frontier_grpo_advantages" in data.batch.keys():
            select_keys.append("frontier_grpo_advantages")
        # FRONTIER_ZERO_DEFER_ACTIVE_LOSS_MASK_V3 keeps no-credit rows out
        # of token-mean denominators while preserving their rollout logs.
        if "frontier_policy_loss_mask" in data.batch.keys():
            select_keys.append("frontier_policy_loss_mask")
        for key in data.batch.keys():
            if key.startswith("frontier_metric__"):
                select_keys.append(key)
        
        # Include student_top_k_log_probs if present (for top-k distillation)
        if "student_top_k_log_probs" in data.batch.keys():
            select_keys.append("student_top_k_log_probs")

        # Include student_top_k_ids if present (for fixing "apples-to-oranges" bug)
        if "student_top_k_ids" in data.batch.keys():
            select_keys.append("student_top_k_ids")

        # Include union_top_k_ids/log_probs for union strategy
        if "union_top_k_ids" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_ids")
            select_keys.append("union_top_k_ids")
            # now we don't need to store student_top_k_ids and student_top_k_log_probs for union strategy
            if "student_top_k_ids" in select_keys:
                select_keys.remove("student_top_k_ids")

        if "union_top_k_log_probs" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_log_probs")
            select_keys.append("union_top_k_log_probs")
            # now we don't need to store student_top_k_log_probs for union strategy
            if "student_top_k_log_probs" in select_keys:
                select_keys.remove("student_top_k_log_probs")   

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                # FRONTIER_GRAD_INTERACTION_DIAGNOSTICS_V1: retain only one
                # eligible microbatch reference for a sparse post-update probe.
                diagnostic_micro_batch = None
                diagnostic_scale_factor = None
                diagnostic_cfg = self.config.get("frontier_grad_diagnostics", {})
                diagnostic_interval = int(diagnostic_cfg.get("interval", 0))
                diagnostic_step = int(data.meta_info.get("global_steps", 0))

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs.get(
                        "frontier_policy_loss_mask", model_inputs["response_mask"]
                    )
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    # FRONTIER_TWO_BRANCH_V1: a single forward produces both
                    # sampled-token and top-k log-probs. They feed separate native losses.
                    frontier_grpo_advantages = model_inputs.get("frontier_grpo_advantages", None)
                    if advantages.dim() == 3:
                        top_k = advantages.shape[-1]
                        student_top_k_ids = None
                        if "union_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["union_top_k_ids"]
                        elif "student_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["student_top_k_ids"]

                        entropy, sampled_log_prob, _, topk_log_probs = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            top_k=top_k, student_top_k_ids=student_top_k_ids
                        )
                        log_prob_for_loss = topk_log_probs
                    else:
                        entropy, sampled_log_prob, *_ = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        log_prob_for_loss = sampled_log_prob

                    if frontier_grpo_advantages is not None:
                        if advantages.dim() != 3 or frontier_grpo_advantages.dim() != 2:
                            raise RuntimeError("frontier hybrid requires OPD [B,T,K] and GRPO [B,T]")
                        if advantages.shape[:2] != frontier_grpo_advantages.shape:
                            raise RuntimeError("frontier hybrid branch shapes do not align")

                    format_mask = None
                    if "format_mask" in model_inputs.keys():
                        format_mask = model_inputs["format_mask"]
            

                    # FRONTIER_TWO_BRANCH_V1 is synchronous/on-policy. Refuse the
                    # fully-async recipe rather than silently changing scientific semantics.
                    if frontier_grpo_advantages is not None and (
                        hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs
                    ):
                        raise RuntimeError("frontier hybrid does not support stale fully-async rollout log-probs")

                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            print("on_policy")
                            old_log_prob = log_prob_for_loss.detach()
                            frontier_grpo_old_log_prob = sampled_log_prob.detach()
                        else:
                            print("off_policy")
                            if advantages.dim() == 3:
                                if "union_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["union_top_k_log_probs"]
                                elif "student_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["student_top_k_log_probs"]
                                else:
                                    old_log_prob = model_inputs["old_log_probs"]
                            else:
                                old_log_prob = model_inputs["old_log_probs"]
                            frontier_grpo_old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Main branch: unchanged upstream OPD top-k policy loss.
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob_for_loss,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        format_mask=format_mask,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Second branch: unchanged native sampled-token GRPO policy loss.
                    if frontier_grpo_advantages is not None:
                        if rollout_is_weights is not None:
                            raise RuntimeError("frontier hybrid rollout correction is not validated")
                        opd_pg_loss = pg_loss
                        grpo_pg_loss, grpo_pg_metrics = policy_loss_fn(
                            old_log_prob=frontier_grpo_old_log_prob,
                            log_prob=sampled_log_prob,
                            advantages=frontier_grpo_advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=None,
                            format_mask=format_mask,
                        )
                        pg_loss = opd_pg_loss + grpo_pg_loss
                        micro_batch_metrics["frontier/opd_pg_loss"] = (
                            opd_pg_loss.detach().item() * loss_scale_factor
                        )
                        micro_batch_metrics["frontier/grpo_pg_loss"] = (
                            grpo_pg_loss.detach().item() * loss_scale_factor
                        )
                        for name, value in grpo_pg_metrics.items():
                            micro_batch_metrics[f"frontier/grpo_{name.split('/')[-1]}"] = value

                        metric_names = (
                            "grpo_native_rms", "opd_native_rms", "native_rms_ratio",
                            "grpo_effective_rms", "opd_effective_rms",
                            "grpo_token_l2_proxy", "opd_token_l2_proxy", "token_l2_proxy_ratio",
                            "grpo_weight_mean", "grpo_active_frac", "opd_weight_mean",
                            "opd_active_frac", "both_active_frac", "neither_active_frac",
                            "group_success_rate_mean", "group_all_wrong_frac",
                            "group_all_correct_frac", "compatibility_mean", "disagreement_mean",
                            "diagnostic_unit",
                            "student_topk_mass_mean", "teacher_reachable_mass_mean",
                            "teacher_unreachable_mass_mean", "compatibility_gate_pass_frac",
                            "topk_cond_kl_teacher_student", "topk_cond_kl_student_teacher",
                            "topk_cond_js",
                            # FRONTIER_ZERO_DEFER_QUEUE_METRICS_V3
                            "group_success_count", "group_size", "queue_visit_phase",
                            "teacher_scored_frac",
                            "first_zero_deferred_frac", "revisit_all_correct_retired_frac",
                            "queue_padding_frac", "grpo_eligible_before_queue_frac",
                            "opd_eligible_before_queue_frac", "training_credit_spent_frac",
                        )
                        for metric_name in metric_names:
                            tensor_key = f"frontier_metric__{metric_name}"
                            if tensor_key in model_inputs:
                                value = model_inputs[tensor_key].float().mean()
                                micro_batch_metrics[f"frontier/{metric_name}"] = (
                                    value.detach().item() * loss_scale_factor
                                )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor

                    if (
                        bool(diagnostic_cfg.get("enable", False))
                        and diagnostic_interval > 0
                        and diagnostic_step > 0
                        and (
                            diagnostic_step == int(diagnostic_cfg.get("first_step", -1))
                            or diagnostic_step % diagnostic_interval == 0
                        )
                        and diagnostic_micro_batch is None
                        and frontier_grpo_advantages is not None
                    ):
                        both = model_inputs.get("frontier_metric__both_active_frac")
                        if both is not None and float(both.detach().sum().cpu().item()) > 0.0:
                            diagnostic_micro_batch = micro_batch
                            diagnostic_scale_factor = loss_scale_factor
                    loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
                if diagnostic_micro_batch is not None:
                    diagnostic_metrics, diagnostic_parameter = self._frontier_gradient_interaction(
                        diagnostic_micro_batch, temperature, diagnostic_scale_factor
                    )
                    append_to_dict(metrics, diagnostic_metrics)
                    print(
                        "FRONTIER_GRAD_INTERACTION_DIAGNOSTICS_V1 "
                        f"step={diagnostic_step} parameter={diagnostic_parameter} "
                        f"metrics={diagnostic_metrics}",
                        flush=True,
                    )
        self.actor_optimizer.zero_grad()
        return metrics
