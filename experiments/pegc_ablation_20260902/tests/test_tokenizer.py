from types import SimpleNamespace
from unittest.mock import patch

from transformers import ProcessorMixin

from verl.utils.tokenizer import get_processor


class AutoProcessorFallback:
    pass


class CompositeProcessor(ProcessorMixin):
    attributes = []

    def __init__(self):
        self.chat_template = None


@patch("verl.utils.tokenizer._load_qwen3_vl_processor")
@patch("verl.utils.tokenizer.AutoConfig.from_pretrained")
@patch("verl.utils.tokenizer.AutoProcessor.from_pretrained")
def test_qwen3_vl_falls_back_to_composite_processor(auto_processor, auto_config, load_qwen3_vl):
    automatic_fallback = AutoProcessorFallback()
    composite_processor = CompositeProcessor()
    auto_processor.return_value = automatic_fallback
    auto_config.return_value = SimpleNamespace(model_type="qwen3_vl")
    load_qwen3_vl.return_value = composite_processor

    processor = get_processor(
        "checkpoint",
        override_chat_template="custom-template",
        trust_remote_code=True,
        use_fast=True,
    )

    assert processor is composite_processor
    assert processor.chat_template == "custom-template"
    auto_config.assert_called_once_with("checkpoint", trust_remote_code=True, use_fast=True)
    load_qwen3_vl.assert_called_once_with("checkpoint", trust_remote_code=True, use_fast=True)


@patch("verl.utils.tokenizer._load_qwen3_vl_processor")
@patch("verl.utils.tokenizer.AutoConfig.from_pretrained")
@patch("verl.utils.tokenizer.AutoProcessor.from_pretrained")
def test_non_qwen_auto_processor_fallback_preserves_none(auto_processor, auto_config, load_qwen3_vl):
    auto_processor.return_value = AutoProcessorFallback()
    auto_config.return_value = SimpleNamespace(model_type="text_model")

    assert get_processor("checkpoint", trust_remote_code=True) is None
    load_qwen3_vl.assert_not_called()


@patch("verl.utils.tokenizer._load_qwen3_vl_processor")
@patch("verl.utils.tokenizer.AutoConfig.from_pretrained")
@patch("verl.utils.tokenizer.AutoProcessor.from_pretrained")
def test_composite_processor_does_not_use_fallback(auto_processor, auto_config, load_qwen3_vl):
    composite_processor = CompositeProcessor()
    auto_processor.return_value = composite_processor

    assert get_processor("checkpoint") is composite_processor
    auto_config.assert_not_called()
    load_qwen3_vl.assert_not_called()
