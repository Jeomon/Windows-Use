"""The Mistral provider must import on every supported mistralai layout.

mistralai 1.x exports ``Mistral`` from the top level and keeps a deprecated
stub in ``mistralai.client``. mistralai 2.x and later dropped the top level
package and moved the client into ``mistralai.client``. The two layouts have
mutually exclusive import paths, so the provider has to try both.

Because ``windows_use/providers/__init__.py`` imports every provider
eagerly, a broken import here takes down ``import windows_use`` completely.
"""

from windows_use.providers.mistral import ChatMistral


def test_mistral_provider_imports():
    assert ChatMistral is not None


def test_mistral_provider_has_a_client_class():
    import windows_use.providers.mistral.llm as mistral_llm

    assert hasattr(mistral_llm, "Mistral")
