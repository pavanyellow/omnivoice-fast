from vllm.model_executor.models.registry import (
    _VLLM_MODELS,
    _LazyRegisteredModel,
    _ModelRegistry,
)

# OmniVoice-only build: this fork keeps just the entries needed to load
# k2-fsa/OmniVoice. See the upstream registry for the full set.
_OMNI_MODELS = {
    "OmniVoiceModel": (
        "omnivoice",
        "omnivoice",
        "OmniVoiceModel",
    ),
}


_VLLM_OMNI_MODELS = {
    **_VLLM_MODELS,
    **_OMNI_MODELS,
}

OmniModelRegistry = _ModelRegistry(
    {
        **{
            model_arch: _LazyRegisteredModel(
                module_name=f"vllm.model_executor.models.{mod_relname}",
                class_name=cls_name,
            )
            for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
        },
        **{
            model_arch: _LazyRegisteredModel(
                module_name=f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}",
                class_name=cls_name,
            )
            for model_arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items()
        },
    }
)
