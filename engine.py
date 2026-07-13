import os
from typing import AsyncIterator, List, Union
from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer

from config import EngineConfig, ModelConfig
from core.llama import LlamaModel


class GenerationOutput:
    def __init__(self, text: str, finished: bool = False):
        self.text = text
        self.finished = finished


class LlamaEngine:
    """Minimalistic pure Python/PyTorch inference engine for Llama 1B."""

    def __init__(self, model_cfg: ModelConfig, engine_cfg: EngineConfig):
        self.model_cfg = model_cfg
        self.engine_cfg = engine_cfg
        self.device = engine_cfg.device if torch.cuda.is_available() else 'cpu'

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_path)
        self.eos_token_id = self.tokenizer.eos_token_id

        # Load model
        self.model = self._init_model()

    def _init_model(self) -> LlamaModel:
        safetensor_path = os.path.join(self.model_cfg.hf_path, 'model.safetensors')
        # Load to CPU as bfloat16 first to avoid double-GPU allocation
        state = load_file(safetensor_path, device='cpu')
        state = {k: v.to(torch.bfloat16) for k, v in state.items()}

        # Init empty model in bfloat16 on CPU, load weights, then ship to GPU
        model = LlamaModel(self.model_cfg).to(dtype=torch.bfloat16)
        model.load_hf_weights(state)
        del state  # free CPU RAM before GPU move
        model.to(self.device)
        model.eval()
        return model

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> AsyncIterator[GenerationOutput]:
        """Async generator that yields token outputs one-by-one."""
        with torch.inference_mode():
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)

            for _ in range(max_new_tokens):
                position_ids = torch.arange(input_ids.size(1), device=self.device).unsqueeze(0)
                logits = self.model(input_ids, position_ids)
                next_logit = logits[:, -1, :]

                if temperature > 0:
                    next_logit = next_logit / temperature

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_logit, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        next_logit[indices_to_remove] = float('-inf')

                    if top_k > 0:
                        top_k_vals, _ = torch.topk(next_logit, top_k, dim=-1)
                        next_logit[next_logit < top_k_vals[:, -1:]] = float('-inf')

                    probs = torch.softmax(next_logit, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_token = torch.argmax(next_logit, dim=-1)

                token_id = int(next_token.item())
                if token_id == self.eos_token_id:
                    break

                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                decoded_text = self.tokenizer.decode([token_id])
                yield GenerationOutput(decoded_text, finished=False)

            yield GenerationOutput('', finished=True)
