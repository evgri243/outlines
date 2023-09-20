import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2 import GPT2TokenizerFast

from outlines.models.transformers import TokenizerWrapper, transformers, TransformersTokenizer, Transformers, \
    TransformerWrapper

TEST_MODEL_NAME = "hf-internal-testing/tiny-random-GPTJForCausalLM"
TEST_TOKENIZER = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
TEST_MODEL = AutoModelForCausalLM.from_pretrained(TEST_MODEL_NAME)


@pytest.mark.parametrize("TEST_TOKENIZER", [TEST_MODEL_NAME, TEST_TOKENIZER])
def test_tokenizer(TEST_TOKENIZER):
    tokenizer = TokenizerWrapper.from_pretrained(TEST_TOKENIZER)
    assert tokenizer.eos_token_id == 0
    assert tokenizer.pad_token_id == 0
    assert isinstance(tokenizer.tokenizer, GPT2TokenizerFast)

    token_ids, attention_mask = tokenizer.encode("Test")
    assert token_ids.ndim == 2
    assert token_ids.shape[0] == 1
    assert isinstance(token_ids, torch.LongTensor)
    assert token_ids.shape == attention_mask.shape

    token_ids, attention_mask = tokenizer.encode(["Test", "Test"])
    assert token_ids.ndim == 2
    assert token_ids.shape[0] == 2
    assert isinstance(token_ids, torch.LongTensor)
    assert token_ids.shape == attention_mask.shape

    token_ids, attention_mask = tokenizer.encode(["A long", "A long sentence"])
    assert token_ids.shape == attention_mask.shape
    assert token_ids[0][0] == tokenizer.pad_token_id
    assert attention_mask[0][0] == 0

    text = tokenizer.decode(torch.tensor([[0, 1, 2]]))
    isinstance(text, str)

    text = tokenizer.decode(torch.tensor([[0, 1, 2], [3, 4, 5]]))
    isinstance(text, list)
    isinstance(text[0], str)
    isinstance(text[1], str)


def test_llama_tokenizer():
    tokenizer = TokenizerWrapper.from_pretrained("hf-internal-testing/llama-tokenizer")

    # Broken
    assert tokenizer.tokenizer.convert_tokens_to_string(["▁baz"]) == "baz"
    assert tokenizer.tokenizer.convert_tokens_to_string(["<0x20>"]) == ""
    assert tokenizer.tokenizer.convert_tokens_to_string(["▁▁▁"]) == "  "

    # Not broken
    assert tokenizer.convert_token_to_string("▁baz") == " baz"
    assert tokenizer.convert_token_to_string("<0x20>") == " "
    assert tokenizer.convert_token_to_string("▁▁▁") == "   "


@pytest.mark.parametrize("TEST_MODEL,TEST_TOKENIZER", [
    (TEST_MODEL_NAME, None),
    (TEST_MODEL, None),
    (TEST_MODEL, TEST_MODEL_NAME),
    (TEST_MODEL, TEST_TOKENIZER)
])
def test_model(TEST_MODEL, TEST_TOKENIZER):
    with pytest.raises((ValueError, RuntimeError),
                       match="When passing device_map as a string|Invalid device string"
                       ):
        transformers(TEST_MODEL, tokenizer_or_name=TEST_TOKENIZER, device="non existent device")

    model = transformers(TEST_MODEL, tokenizer_or_name=TEST_TOKENIZER, device="cpu")
    assert isinstance(model.tokenizer, TokenizerWrapper)
    assert model.device.type == "cpu"

    input_ids = torch.tensor([[0, 1, 2]])
    logits = model(input_ids, torch.ones_like(input_ids))
    assert logits.type() == "torch.FloatTensor"
    assert logits.ndim == 2
    assert logits.shape[0] == 1

    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    logits = model(input_ids, torch.ones_like(input_ids))
    assert logits.type() == "torch.FloatTensor"
    assert logits.ndim == 2
    assert logits.shape[0] == 3

    input_ids = torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [0, 1, 2]]])
    logits = model(input_ids, torch.ones_like(input_ids))
    assert logits.ndim == 3
    assert logits.shape[0] == 2
    assert logits.shape[1] == 2
    assert torch.equal(logits[0][0], logits[1][1])

def test_legacy_loaders():
    legacy_tokenizer = TransformersTokenizer(TEST_MODEL_NAME)
    assert isinstance(legacy_tokenizer, TransformersTokenizer)
    assert isinstance(legacy_tokenizer, TokenizerWrapper)

    legacy_model = Transformers(TEST_MODEL, legacy_tokenizer)
    assert isinstance(legacy_model, Transformers)
    assert isinstance(legacy_model, TransformerWrapper)