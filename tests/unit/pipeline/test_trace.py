"""Unit tests for event-based trace models."""

from docling_graph.pipeline.trace import EventTrace, event_trace_to_jsonable


class TestEventTrace:
    def test_emit_records_ordered_events(self):
        trace = EventTrace()
        trace.emit("pipeline_started", "pipeline", {"source": "doc.pdf"})
        trace.emit("extraction_completed", "extraction", {"extraction_id": 0})

        assert len(trace.events) == 2
        assert trace.events[0].sequence == 0
        assert trace.events[1].sequence == 1
        assert trace.events[0].event_type == "pipeline_started"
        assert trace.events[1].event_type == "extraction_completed"

    def test_find_and_latest_payload(self):
        trace = EventTrace()
        trace.emit("extraction_completed", "extraction", {"value": 1})
        trace.emit("extraction_completed", "extraction", {"value": 2})

        events = trace.find_events("extraction_completed")
        assert len(events) == 2
        latest = trace.latest_payload("extraction_completed")
        assert latest == {"value": 2}

    def test_event_trace_to_jsonable_truncates_large_strings(self):
        trace = EventTrace()
        trace.emit(
            "extraction_completed",
            "extraction",
            {"structured_primary_attempt_raw": "x" * 5000},
        )

        payload = event_trace_to_jsonable(trace, max_text_len=120)
        assert "summary" in payload
        assert "steps" in payload
        extraction_step = next(
            step for step in payload["steps"] if step["name"] == "data_extraction"
        )
        out_text = extraction_step["artifacts"]["extractions"][0]["structured_primary_attempt_raw"]
        assert "truncated" in out_text.lower()
        assert len(out_text) < 5000

    def test_event_trace_to_jsonable_exports_compact_steps_with_runtime_seconds(self):
        trace = EventTrace()
        trace.emit("pipeline_started", "pipeline", {"source": "doc.pdf"})
        trace.emit("page_markdown_extracted", "extraction", {"page_number": 1})
        trace.emit("extraction_completed", "extraction", {"extraction_id": 0})
        trace.emit("graph_created", "graph_conversion", {"node_count": 1, "edge_count": 0})

        payload = event_trace_to_jsonable(trace)
        assert "runtime_seconds" in payload["summary"]
        assert payload["summary"]["page_count"] == 1
        assert payload["summary"]["extraction_success"] is True
        assert payload["summary"]["fallback_used"] is False
        assert payload["summary"]["node_count"] == 1
        assert payload["summary"]["edge_count"] == 0
        step_names = [step["name"] for step in payload["steps"]]
        assert step_names == ["pipeline", "docling_conversion", "data_extraction", "graph_mapping"]

        for step in payload["steps"]:
            assert "runtime_seconds" in step
            assert "status" in step
            assert "artifacts" in step
            assert "started_at" not in step
            assert "finished_at" not in step
            assert "events" not in step
