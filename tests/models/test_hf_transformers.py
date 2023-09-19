import numpy as np
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM

from outlines.models.hf_transformers import HuggingFaceCompletion

TEST_MODEL_NAME = "hf-internal-testing/tiny-random-GPTJForCausalLM"
TEST_MODEL_TUPLE = (
    AutoModelForCausalLM.from_pretrained(TEST_MODEL_NAME), AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
)


@pytest.mark.parametrize("TEST_MODEL", [TEST_MODEL_NAME, TEST_MODEL_TUPLE])
def test_samples(TEST_MODEL):
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)

    answer = model("test", samples=1)
    assert isinstance(answer, str)

    answer = model("test")
    assert isinstance(answer, str)

    answers = model("test", samples=3)
    assert isinstance(answers, np.ndarray)
    assert len(answers) == 3


@pytest.mark.parametrize("TEST_MODEL", [TEST_MODEL_NAME, TEST_MODEL_TUPLE])
def test_prompt_array(TEST_MODEL):
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)
    prompts = [["Hello", "Bonjour"], ["Ciao", "Hallo"]]
    answers = model(prompts)
    assert isinstance(answers, np.ndarray)
    assert answers.shape == (2, 2)

    answers = model(prompts, samples=5)
    assert isinstance(answers, np.ndarray)
    assert answers.shape == (2, 2, 5)


@pytest.mark.parametrize("TEST_MODEL", [TEST_MODEL_NAME, TEST_MODEL_TUPLE])
def test_type_int(TEST_MODEL):
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)
    answer = model("test", type="int")
    int(answer)

    answers = model(["test", "other_test"], type="int")
    for answer in answers:
        int(answer)


@pytest.mark.parametrize("TEST_MODEL", [TEST_MODEL_NAME, TEST_MODEL_TUPLE])
def test_type_float(TEST_MODEL):
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)
    answer = model("test", type="float")
    float(answer)

    answers = model(["test", "other_test"], type="float")
    for answer in answers:
        float(answer)


@pytest.mark.parametrize("TEST_MODEL", [TEST_MODEL_NAME, TEST_MODEL_TUPLE])
def test_incompatible_constraints(TEST_MODEL):
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)

    with pytest.raises(ValueError):
        model("test", type="float", is_in=["test"])


@pytest.mark.parametrize("TEST_MODEL", [TEST_MODEL_NAME, TEST_MODEL_TUPLE])
def test_choices(TEST_MODEL):
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=50)

    choices = ["a", "and a long sequence", "with\n line break"]
    answer = model("test", is_in=choices)
    assert answer in choices

    answers = model(["test", "other_test"], is_in=choices)
    for answer in answers:
        assert answer in choices


@pytest.mark.parametrize("TEST_MODEL", [TEST_MODEL_NAME, TEST_MODEL_TUPLE])
def test_stop(TEST_MODEL):
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=1000)

    stop = [" ", "\n"]
    answer = model("test", stop_at=stop)
    for seq in stop:
        assert seq not in answer

    answers = model(["test", "other_test"], stop_at=stop)
    for seq in stop:
        for answer in answers:
            assert seq not in answer


@pytest.mark.xfail
@pytest.mark.parametrize("TEST_MODEL", [TEST_MODEL_NAME, TEST_MODEL_TUPLE])
def test_type_multiple_samples(TEST_MODEL):
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)
    answer = model("test", type="int", samples=2)
    int(answer)


@pytest.mark.xfail
@pytest.mark.parametrize("TEST_MODEL", [TEST_MODEL_NAME, TEST_MODEL_TUPLE])
def test_is_in_multiple_samples(TEST_MODEL):
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)
    model("test", is_in=["a", "b"], samples=2)


@pytest.mark.xfail
@pytest.mark.parametrize("TEST_MODEL", [TEST_MODEL_NAME, TEST_MODEL_TUPLE])
def test_stop_at_multiple_samples(TEST_MODEL):
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)
    model("test", stop_at=[" "], samples=2)
