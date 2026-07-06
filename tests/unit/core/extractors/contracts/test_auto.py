"""Unit tests for the auto extraction-contract resolution."""

from docling_graph.core.extractors.contracts.auto import (
    CHARS_PER_TOKEN,
    DIRECT_OVERFLOW_RATIO,
    ContractDecision,
    resolve_auto_contract,
)


class TestResolveAutoContract:
    def test_small_document_resolves_direct(self):
        decision = resolve_auto_contract(
            markdown_chars=3_000,
            output_budget_tokens=4092,
            context_limit_tokens=131_072,
            chunking_available=True,
        )
        assert decision.contract == "direct"
        assert "fits" in decision.reason

    def test_document_exceeding_context_resolves_dense(self):
        # The 2026-07-05 IBM annual report: 626,746 chars vs 131k context.
        decision = resolve_auto_contract(
            markdown_chars=626_746,
            output_budget_tokens=8192,
            context_limit_tokens=131_072,
            chunking_available=True,
        )
        assert decision.contract == "dense"
        assert "context window" in decision.reason

    def test_document_exceeding_output_budget_resolves_dense(self):
        # Fits the context window but a single response cannot represent it.
        decision = resolve_auto_contract(
            markdown_chars=100_000,
            output_budget_tokens=4092,
            context_limit_tokens=131_072,
            chunking_available=True,
        )
        assert decision.contract == "dense"
        assert "response can represent" in decision.reason

    def test_unknown_context_limit_uses_output_budget_only(self):
        decision = resolve_auto_contract(
            markdown_chars=10_000,
            output_budget_tokens=4092,
            context_limit_tokens=None,
            chunking_available=True,
        )
        assert decision.contract == "direct"

    def test_chunking_unavailable_forces_direct(self):
        decision = resolve_auto_contract(
            markdown_chars=626_746,
            output_budget_tokens=8192,
            context_limit_tokens=131_072,
            chunking_available=False,
        )
        assert decision.contract == "direct"
        assert "chunking disabled" in decision.reason

    def test_boundary_exactly_at_output_capacity(self):
        # Exactly at the overflow ratio boundary is still direct (<=).
        budget = 4092
        boundary_chars = int(budget * CHARS_PER_TOKEN * DIRECT_OVERFLOW_RATIO)
        decision = resolve_auto_contract(
            markdown_chars=boundary_chars,
            output_budget_tokens=budget,
            context_limit_tokens=1_000_000,
            chunking_available=True,
        )
        assert decision.contract == "direct"
        over = resolve_auto_contract(
            markdown_chars=boundary_chars + 1,
            output_budget_tokens=budget,
            context_limit_tokens=1_000_000,
            chunking_available=True,
        )
        assert over.contract == "dense"

    def test_output_pressure_ratio_is_one(self):
        """R=1.0 by decision: silent self-rationing of verbatim fields costs
        more than dense's extra calls. Guards against quietly relaxing it."""
        assert DIRECT_OVERFLOW_RATIO == 1.0

    def test_cgv_sized_document_now_routes_dense(self):
        """Regression for the 2026-07-06 run: 62,540 chars with an 8192-token
        output budget resolved direct at ratio 2.0 and lost 57% of verbatim
        exclusion texts; at R=1.0 it must route to dense."""
        decision = resolve_auto_contract(
            markdown_chars=62_540,
            output_budget_tokens=8_192,
            context_limit_tokens=131_072,
            chunking_available=True,
        )
        assert decision.contract == "dense"
        assert "response can represent" in decision.reason

    def test_describe_includes_numbers(self):
        decision = resolve_auto_contract(
            markdown_chars=4_000,
            output_budget_tokens=4092,
            context_limit_tokens=131_072,
            chunking_available=True,
        )
        text = decision.describe()
        assert "contract=direct" in text
        assert "1000" in text  # 4000 chars / 4
        assert "131072" in text

    def test_decision_is_frozen_dataclass(self):
        decision = ContractDecision(
            contract="direct",
            reason="r",
            estimated_input_tokens=1,
            output_budget_tokens=2,
            context_limit_tokens=None,
        )
        try:
            decision.contract = "dense"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised


class TestStrategyContractResolution:
    """ManyToOneStrategy._resolve_contract: passthrough for explicit contracts,
    size-driven decision (with logging) for auto."""

    def _strategy(self, contract: str, chunker: bool = True) -> object:
        from unittest.mock import MagicMock

        from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy

        strategy = ManyToOneStrategy.__new__(ManyToOneStrategy)
        strategy._extraction_contract = contract
        strategy.doc_processor = MagicMock()
        strategy.doc_processor.chunker = MagicMock() if chunker else None
        return strategy

    def _backend(self, budget: int = 4092, context: int = 131_072) -> object:
        from unittest.mock import MagicMock

        backend = MagicMock()
        backend._estimated_output_token_budget = lambda: budget
        backend.client.context_limit = context
        return backend

    def test_explicit_contract_passes_through(self):
        strategy = self._strategy("dense")
        assert strategy._resolve_contract(self._backend(), "x" * 1_000_000) == "dense"
        strategy = self._strategy("direct")
        assert strategy._resolve_contract(self._backend(), "tiny") == "direct"

    def test_auto_picks_dense_for_oversized_document(self):
        strategy = self._strategy("auto")
        assert strategy._resolve_contract(self._backend(budget=8192), "x" * 626_746) == "dense"

    def test_auto_picks_direct_for_small_document(self):
        strategy = self._strategy("auto")
        assert strategy._resolve_contract(self._backend(), "x" * 2_000) == "direct"

    def test_auto_without_chunker_picks_direct(self):
        strategy = self._strategy("auto", chunker=False)
        assert strategy._resolve_contract(self._backend(budget=8192), "x" * 626_746) == "direct"


class TestFormatNeutralDecision:
    """The auto decision measures content chars, not serialized chars: the same
    document resolves to the same contract in every LLM serialization."""

    def _strategy(self, contract: str, llm_format: str) -> object:
        from unittest.mock import MagicMock

        from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy

        strategy = ManyToOneStrategy.__new__(ManyToOneStrategy)
        strategy._extraction_contract = contract
        strategy.doc_processor = MagicMock()
        strategy.doc_processor.chunker = MagicMock()
        strategy.doc_processor.llm_input_format = llm_format
        return strategy

    def _backend(self, budget: int = 8192, context: int = 131_072) -> object:
        from unittest.mock import MagicMock

        backend = MagicMock()
        backend._estimated_output_token_budget = lambda: budget
        backend.client.context_limit = context
        return backend

    def test_doclang_geo_markup_does_not_flip_the_contract(self):
        # ~20k chars of content: direct at an 8192-token budget (capacity 32k).
        content = "word " * 4_000
        markdown_strategy = self._strategy("auto", "markdown")
        assert markdown_strategy._resolve_contract(self._backend(), content) == "direct"

        # The same content wrapped in enough geo markup to cross 32k raw chars
        # must STILL resolve direct: markup is not information.
        geo_text = "".join(
            f'<text><location value="{i}"/><location value="{i}"/><![CDATA[{chunk}]]></text>\n'
            for i, chunk in enumerate([content[i : i + 100] for i in range(0, len(content), 100)])
        )
        assert len(geo_text) > 32_768 * DIRECT_OVERFLOW_RATIO
        geo_strategy = self._strategy("auto", "doclang-geo")
        assert geo_strategy._resolve_contract(self._backend(), geo_text) == "direct"


class TestAutoFormatPairing:
    """llm_input_format='auto' pairs the serialization to the resolved contract."""

    def _strategy_and_backend(self, llm_format: str = "auto") -> tuple:
        from unittest.mock import MagicMock

        from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy

        strategy = ManyToOneStrategy.__new__(ManyToOneStrategy)
        strategy._extraction_contract = "auto"
        strategy.doc_processor = MagicMock()
        strategy.doc_processor.llm_input_format = llm_format
        backend = MagicMock()
        backend.llm_input_format = "auto"
        backend._dense_config_raw = {"llm_input_format": "auto"}
        return strategy, backend

    def test_direct_pairs_with_doclang_geo(self):
        strategy, backend = self._strategy_and_backend()
        assert strategy._resolve_llm_format(backend, "direct") is True
        strategy.doc_processor.set_llm_input_format.assert_called_once_with("doclang-geo")
        assert backend.llm_input_format == "doclang-geo"
        assert backend._dense_config_raw["llm_input_format"] == "doclang-geo"

    def test_dense_pairs_with_doclang(self):
        strategy, backend = self._strategy_and_backend()
        assert strategy._resolve_llm_format(backend, "dense") is True
        strategy.doc_processor.set_llm_input_format.assert_called_once_with("doclang")
        assert backend.llm_input_format == "doclang"

    def test_text_input_resolves_markdown(self):
        strategy, backend = self._strategy_and_backend()
        assert strategy._resolve_llm_format(backend, "dense", text_input=True) is True
        strategy.doc_processor.set_llm_input_format.assert_called_once_with("markdown")

    def test_explicit_format_is_never_touched(self):
        strategy, backend = self._strategy_and_backend(llm_format="markdown")
        assert strategy._resolve_llm_format(backend, "direct") is False
        strategy.doc_processor.set_llm_input_format.assert_not_called()
