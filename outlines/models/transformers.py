import logging
import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Dict

import torch
from transformers.file_utils import SPIECE_UNDERLINE

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

__all__ = ["transformers"]


class TransformerWrapper:
    """Represents a `transformers` model."""

    def __init__(self, model: "PreTrainedModel", tokenizer: Union["TokenizerWrapper", "PreTrainedTokenizerBase"]):
        self.device = model.device
        self.model = model
        if not isinstance(tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(tokenizer)
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        # `transformers` model accept `input_ids` of size at most equal to 2. We
        # thus reshape the input array, call the model and reshape the output
        # logits.
        batch_shape = input_ids.shape[:-1]
        num_tokens = input_ids.shape[-1]
        input_ids = input_ids.reshape(math.prod(batch_shape), num_tokens)

        kwargs.setdefault("output_attentions", False)
        kwargs.setdefault("output_hidden_states", False)

        output = self.model(input_ids, attention_mask=attention_mask, return_dict=True, **kwargs)

        # reshape back to original shape
        next_token_logits = output.logits[:, -1, :]
        next_token_logits = next_token_logits.reshape(batch_shape + (-1,))

        return next_token_logits


class TokenizerWrapper:
    """A wrapper around `transformers` tokenizers to extend some additional functionality
    and work around some issues.

    """

    tokenizer: "PreTrainedTokenizerBase"

    def __init__(self, tokenizer: "PreTrainedTokenizerBase"):
        """Instantiate a `TokenizerWrapper` instance.

        Parameters:
        -----------
        tokenizer_or_name
            A string for the name of the tokenizer as listed on Hugging Face's model page,
            or a preloaded tokenizer instance.
        **kwargs
            Keyword arguments to pass to the `from_pretrained` method when loading the tokenizer.
            Ignored if `tokenizer` is a preloaded tokenizer instance.


        Returns:
        --------
        A `TokenizerWrapper` instance.
        """
        self.tokenizer = tokenizer

        if self.tokenizer.padding_side != "left":
            logging.warning(
                f"Enforcing padding side to be `left`. Found padding side `{self.tokenizer.padding_side}` instead."
            )
            self.tokenizer.padding_side = "left"

        # making sure tokenizer is consistent
        # avoiding annoying warnings on tokens not being set on check
        with self.suppress_tokenizer_warnings():
            if self.tokenizer.eos_token is None:
                logging.warning("`eos_token` not set, behavior may be inconsistent")

            if self.tokenizer.bos_token is None:
                self.tokenizer.bos_token = self.tokenizer.eos_token
                logging.info('`bos_token` not set. Using `eos_token` "%s" instead', self.tokenizer.eos_token)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logging.info('`pad_token` not set. Using `eos_token` "%s" instead', self.tokenizer.eos_token)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, "PreTrainedTokenizerBase"], *args,
                        **kwargs) -> "TokenizerWrapper":
        """Instantiate a `TokenizerWrapper` instance from a pretrained tokenizer.
        Mimics `transformers.AutoTokenizer.from_pretrained` behavior and parameters.

        Parameters:
        -----------
        pretrained_model_name_or_path
            A string for the name of the tokenizer as listed on Hugging Face's model page,
            or a preloaded tokenizer instance.
        **kwargs
            Keyword arguments to pass to the `from_pretrained` method when loading the tokenizer.
            Ignored if `tokenizer` is a preloaded tokenizer instance.

        Returns:
        --------
        A `TokenizerWrapper` instance.
        """

        try:
            from transformers import AutoTokenizer, PreTrainedTokenizerBase
        except ImportError:
            raise ImportError("The `tokenizers` and `transformers` libraries need to be installed in order "
                              "to use `transformers` models.")

        if isinstance(pretrained_model_name_or_path, str):
            kwargs.setdefault("padding_side", "left")
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        elif isinstance(pretrained_model_name_or_path, PreTrainedTokenizerBase):
            tokenizer = pretrained_model_name_or_path
        else:
            raise TypeError("`tokenizer_or_name` must be a string or a `transformers.PreTrainedTokenizer` instance, "
                            f"{type(pretrained_model_name_or_path)} found.")

        return cls(tokenizer)

    # remapping properties
    @property
    def vocabulary(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    @property
    def eos_token(self) -> str:
        """The end-of-sequence token of the internal tokenizer."""
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self) -> int:
        """The end-of-sequence token id of the internal tokenizer."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token(self) -> str:
        """The padding token of the internal tokenizer."""
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self) -> int:
        """The padding token id of the internal tokenizer."""
        return self.tokenizer.pad_token_id

    # overrides
    def encode(self, prompt: Union[str, List[str]], **kwargs) -> Tuple[torch.LongTensor, torch.LongTensor]:
        kwargs.setdefault("padding", True)
        kwargs.setdefault("return_tensors", "pt")

        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: Union[torch.LongTensor, List[int]], keep_leading_space_if_llama=False, **kwargs) -> List[str]:
        text = self.tokenizer.batch_decode(token_ids, **kwargs)

        if self._is_llama_tokenizer and keep_leading_space_if_llama and len(token_ids) > 0:
            first_token = self.tokenizer.convert_ids_to_tokens([token_ids[0]])[0]
            if first_token.startswith(SPIECE_UNDERLINE) or first_token == "<0x20>":
                text[0] = " " + text[0]

        return text

    def convert_token_to_string(self, token: str) -> str:
        string = self.tokenizer.convert_tokens_to_string([token])

        if self._is_llama_tokenizer:
            # A hack to handle missing spaces in HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

        return string

    # helpers
    @property
    def _is_llama_tokenizer(self) -> bool:
        """Checks whether an object is a `transformers` tokenizer for LLAMA or CodeLLAMA.
        """

        # after the first import, imports should resolve immediately
        try:
            from transformers.models.llama import LlamaTokenizer
        except ImportError:

            class LlamaTokenizer:  # type: ignore
                pass

        try:
            from transformers.models.llama import LlamaTokenizerFast
        except ImportError:

            class LlamaTokenizerFast:  # type: ignore
                pass

        try:
            from transformers.models.code_llama import CodeLlamaTokenizer
        except ImportError:

            class CodeLlamaTokenizer:  # type: ignore
                pass

        try:
            from transformers.models.code_llama import CodeLlamaTokenizerFast
        except ImportError:

            class CodeLlamaTokenizerFast:  # type: ignore
                pass

        return isinstance(self.tokenizer,
                          (LlamaTokenizer, LlamaTokenizerFast, CodeLlamaTokenizer, CodeLlamaTokenizerFast))

    @contextmanager
    def suppress_tokenizer_warnings(self):
        """Suppresses warnings from the internal tokenizer."""
        orig_verbose = self.tokenizer.verbose
        self.tokenizer.verbose = False

        yield

        self.tokenizer.verbose = orig_verbose


def transformers(model_or_name: Union[str, "PreTrainedModel"], *,
                 tokenizer_or_name: Optional[Union[str, "PreTrainedTokenizerBase"]] = None,
                 device: Optional[str] = None, model_kwargs=None, tokenizer_kwargs=None):
    """
    Instantiate a model from the `transformers` library and its tokenizer.

    Parameters
    ----------
    model_or_name
        A string for the name of the model as listed on Hugging Face's model page, or a preloaded model instance.
    tokenizer_or_name
        A string for the name of the tokenizer as listed on Hugging Face's model page,
        or a preloaded tokenizer instance. If `None`, the tokenizer will be loaded using `model.name_or_path` value.
    device
        The device(s) on which the model should be loaded.
        This overrides the value passed for `device_map` in `model_kwargs`.
    model_kwargs
        A dictionary that contains the keyword arguments to pass to the `from_pretrained` method
        when loading the model. Ignored if `model_or_name` is a preloaded model instance.
    tokenizer_kwargs
        A dictionary that contains the keyword arguments to pass to the `from_pretrained` method
        when loading the tokenizer. Ignored if `tokenizer` is a preloaded tokenizer instance.

    Returns
    -------
    A `TransformersModel` model instance.

    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
    except ImportError:
        raise ImportError("The `transformers` library needs to be installed in order to use `transformers` models.")

    try:
        from peft import PeftModel
    except ImportError:
        class PeftModel:
            pass

    try:
        from torch.nn import Module as TorchModule
    except ImportError:
        class TorchModule:
            pass

    if isinstance(model_or_name, str):
        if model_kwargs is None:
            model_kwargs = dict()
        if device is not None:
            model_kwargs["device_map"] = device
        model = AutoModelForCausalLM.from_pretrained(model_or_name, **(model_kwargs or {}))
    elif isinstance(model_or_name, (PreTrainedModel, PeftModel, TorchModule)):
        if isinstance(model_or_name, TorchModule):
            logging.warning(
                "Using a `torch.nn.Module` instance as a model is experimental and may not work. "
                "Make sure the model has a compatible interface to PretrainedModel.generate()."
            )
        model = model_or_name
        if device is not None:
            model = model.to(device)
    else:
        raise TypeError("`model_or_name` must be a string, a `transformers.PreTrainedModel` "
                        f"or a `peft.PeftModel` instance, {type(model_or_name)} found.")

    tokenizer = TokenizerWrapper.from_pretrained(tokenizer_or_name or model.name_or_path, **(tokenizer_kwargs or {}))

    return TransformerWrapper(model, tokenizer)


# ensuring compatibility with older versions
Transformers = TransformerWrapper


class TransformersTokenizer(TokenizerWrapper):
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, model_name: str, **kwargs):
        from transformers import AutoTokenizer

        kwargs.setdefault("padding_side", "left")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

        super().__init__(tokenizer)
