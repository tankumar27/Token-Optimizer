from app.models import ChatMessage
from optimizer.cheap_layer import cheap_layer_backend, run_self_checks, canonical_fact_keys
from optimizer.pipeline import optimize_messages


def test_cheap_layer_self_checks_pass():
    assert run_self_checks() == []


def test_cheap_layer_removes_exact_duplicate_sentences():
    text = "Do not promise refund approval. Do not promise refund approval. Inform the customer that the request is under review."
    output, traces = cheap_layer_backend(text, "balanced")
    assert output.count("Do not promise refund approval.") == 1
    assert traces[0]["accepted"]
    assert traces[0]["surface_changes"]


def test_cheap_layer_detects_number_word_equivalent_metrics():
    text = (
        "Database write latency increased from 12 ms to 28 ms. "
        "Database write latency increased from twelve ms to twenty-eight ms."
    )
    output, traces = cheap_layer_backend(text, "balanced")
    assert traces[0]["accepted"]
    assert output in {
        "Database write latency increased from 12 ms to 28 ms.",
        "Database write latency increased from twelve ms to twenty-eight ms.",
    }
    assert canonical_fact_keys(text) == canonical_fact_keys(output)


def test_pipeline_compresses_numeric_surface_equivalents():
    text = (
        "Migration reached 84%. "
        "Migration reached eighty-four percent. "
        "Latency decreased from 720 ms to 290 ms. "
        "Latency decreased from seven hundred twenty ms to two hundred ninety ms."
    )
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert result["optimized_tokens"] < result["original_tokens"]
    assert output.count("Migration reached") == 1
    assert output.count("Latency decreased") == 1
    assert "84%" in output


def test_task_context_boundary_keeps_context_compressible():
    text = (
        "System:\n"
        "You are careful. If information is unavailable, say unavailable.\n\n"
        "Task:\n"
        "Write a concise leadership summary without losing protected facts.\n\n"
        "Context:\n"
        "The remediation started on 2026-07-07. "
        "The remediation started on July 7, 2026. "
        "Refund exposure was $8,200,000. "
        "Refund exposure was eight million two hundred thousand dollars. "
        "Support logged 84 timeout complaints. "
        "Support logged eighty-four timeout complaints. "
        "Support logged 37 audit-export complaints. "
        "Support logged thirty-seven audit export complaints. "
        "Region me-south-1 status is unavailable.\n\n"
        "Final instruction:\n"
        "Return a concise summary."
    )
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert result["optimized_tokens"] < result["original_tokens"]
    assert output.count("The remediation started") == 1
    assert output.count("Refund exposure was") == 1
    assert "Support logged 84 timeout complaints and 37 audit-export complaints." in output
    assert "eighty-four timeout complaints" not in output
    assert "thirty-seven audit export complaints" not in output
    assert "If information is unavailable, say unavailable." in output
    assert "Region me-south-1 status is unavailable." in output


def test_cheap_layer_compresses_repeated_entity_frames():
    text = (
        "Retrieved Context Chunk 1:\n"
        "Cohort NA-A was affected. Cohort EU-B was affected. Cohort APAC-C was affected. "
        "Cell cl-01 was affected. Cell cl-02 was affected. "
        "Biomarker IL-6 status is healthy. Biomarker TNF-alpha status is healthy."
    )
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert "Affected cohorts: NA-A, EU-B, and APAC-C." in output
    assert "Affected cells: cl-01 and cl-02." in output
    assert "Healthy biomarkers: IL-6 and TNF-alpha." in output


def test_cheap_layer_compresses_negative_safety_frames_without_ambiguity():
    text = (
        "No patient records were deleted. No patient records were corrupted. "
        "No patient records were exposed. Patient records were not exposed. "
        "Service GENOMIC-CACHE is not offline. Service GENOMIC-CACHE is not disabled."
    )
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert "No patient records were deleted, corrupted, or exposed." in output
    assert "Service GENOMIC-CACHE is not offline and is not disabled." in output
    assert "No Service GENOMIC-CACHE is offline" not in output


def test_cheap_layer_compresses_airline_operations_frames():
    text = (
        "Retrieved Context Chunk 1:\n"
        "Airport ORD was affected. Airport JFK was affected. Airport LAX was affected. "
        "Gate B-17 was affected. Gate C-04 was affected. "
        "Flight AA-217 is operational. Flight UA-442 is operational. "
        "Passenger PAX-0041 baggage was recovered. Passenger PAX-0042 baggage was recovered. "
        "Baggage BAG-7712 was recovered. Baggage BAG-7713 was recovered. "
        "Ops logged 41 baggage-delay complaints. Ops logged forty-one baggage delay complaints."
    )
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert "Affected airports: ORD, JFK, and LAX." in output
    assert "Affected gates: B-17 and C-04." in output
    assert "Operational flights: AA-217 and UA-442." in output
    assert "Recovered passenger baggage: PAX-0041 and PAX-0042." in output
    assert "Recovered baggage IDs: BAG-7712 and BAG-7713." in output
    assert output.count("Ops logged") == 1


def test_progress_events_with_same_value_are_not_merged():
    text = (
        "Recovery reached 79%. "
        "Recovery reached seventy-nine percent. "
        "Recovery paused at 79%. "
        "Recovery paused at seventy-nine percent."
    )
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert output.count("Recovery reached 79%.") == 1
    assert output.count("Recovery paused at 79%.") == 1


def test_repeated_frame_preserves_original_object_noun():
    text = (
        "Zone-A routing tables were migrated. "
        "Zone-B routing tables were migrated. "
        "Zone-C routing tables were migrated."
    )
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert output == "Zone-A, Zone-B, and Zone-C routing tables were migrated."
    assert "routing zones" not in output


def test_repeated_singular_conflict_warnings_become_one_plural_warning():
    text = (
        "Record A says service alpha is enabled but Record B says service alpha is disabled. "
        "Do not resolve this conflict. "
        "Invoice source says total is $40 but ledger source says total is $44. "
        "Do not resolve this conflict. "
        "Policy version one allows export, however policy version two prohibits export. "
        "Do not resolve this conflict."
    )
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert "service alpha is enabled" in output
    assert "ledger source says total is $44" in output
    assert "policy version two prohibits export" in output
    assert output.count("Do not resolve these conflicts.") == 1
    assert "Do not resolve this conflict." not in output


def test_conflict_warning_compression_preserves_section_boundaries():
    text = (
        "Retrieved Context Chunk 1:\n"
        "North source says quota is 10 but South source says quota is 12. Do not merge this contradiction. "
        "Primary log says job is complete but audit log says job is blocked. Do not merge this contradiction. "
        "Manual says access is allowed, however policy says access is denied. Do not merge this contradiction.\n\n"
        "Task:\n"
        "Do not merge this contradiction."
    )
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert output.count("Do not merge these contradictions.") == 1
    assert "Task:\nDo not merge this contradiction." in output
    assert "quota is 10" in output and "quota is 12" in output


def test_unavailable_status_paraphrase_compresses_with_same_entity_state():
    text = "Region me-central-1 status is unavailable. The me-central-1 region status is unavailable."
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert output == "Region me-central-1 status is unavailable."


def test_unavailable_status_paraphrase_compresses_for_reversible_entity_labels():
    cases = [
        (
            "Control CTRL-411 status is unavailable. The CTRL-411 control status is unavailable.",
            "Control CTRL-411 status is unavailable.",
        ),
        (
            "Policy PCI-DSS-3-2-1 status is unavailable. The PCI-DSS-3-2-1 policy status is unavailable.",
            "Policy PCI-DSS-3-2-1 status is unavailable.",
        ),
    ]
    for text, expected in cases:
        result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
        assert result["quality_gate_status"]["accepted"]
        assert result["optimized_messages"][0].content == expected


def test_time_and_money_word_equivalents_compress_only_when_values_match():
    cases = [
        (
            "Meeting window started at 09:30 AM. Meeting window started at nine thirty AM.",
            "Meeting window started at 09:30 AM.",
        ),
        (
            "Meeting window paused at 11:45 AM. Meeting window paused at eleven forty-five AM.",
            "Meeting window paused at 11:45 AM.",
        ),
        (
            "Refund exposure was $1,250,000. Refund exposure was one million two hundred fifty thousand dollars.",
            "Refund exposure was $1,250,000.",
        ),
        (
            "Penalty reserve increased from $320,000 to $480,000. Penalty reserve increased from three hundred twenty thousand dollars to four hundred eighty thousand dollars.",
            "Penalty reserve increased from $320,000 to $480,000.",
        ),
        (
            "Chargeback reserve decreased from $210,000 to $150,000. Chargeback reserve decreased from two hundred ten thousand dollars to one hundred fifty thousand dollars.",
            "Chargeback reserve decreased from $210,000 to $150,000.",
        ),
    ]
    for text, expected in cases:
        result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
        assert result["quality_gate_status"]["accepted"]
        assert result["optimized_messages"][0].content == expected


def test_date_equivalent_duplicates_compress_only_with_same_event_frame():
    cases = [
        (
            "Release started on 2026-07-01. Release started on July 1, 2026.",
            "Release started on 2026-07-01.",
        ),
        (
            "Filing deadline is 2026-07-15. Filing deadline is July 15, 2026.",
            "Filing deadline is 2026-07-15.",
        ),
        (
            "Legal hold expires on 2026-08-31. Legal hold expires on August 31, 2026.",
            "Legal hold expires on 2026-08-31.",
        ),
    ]
    for text, expected in cases:
        result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
        assert result["quality_gate_status"]["accepted"]
        assert result["optimized_messages"][0].content == expected


def test_date_equivalence_rejects_different_or_unsafe_date_frames():
    cases = [
        "Release started on 2026-07-01. Release started on July 2, 2026.",
        "Release started on 2026-07-01. Release paused on July 1, 2026.",
        "Filing deadline is 2026-07-15. Legal hold expires on July 15, 2026.",
        "Legal hold expired on 2026-08-31. Legal hold expires on August 31, 2026.",
        "Release started on 2026-07-01. Release started tomorrow.",
        "Release started on 2026-07-01. Release started around July 1, 2026.",
        "Conflict record A says release started on 2026-07-01. Conflict record B says release started on July 1, 2026.",
    ]
    for text in cases:
        result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
        output = result["optimized_messages"][0].content
        if "July 2" in text:
            assert "July 2, 2026" in output
        if "paused" in text:
            assert "paused on July 1, 2026" in output
        if "Legal hold expires" in text and "Filing deadline" in text:
            assert "Filing deadline is 2026-07-15." in output
            assert "Legal hold expires on July 15, 2026." in output
        if "expired" in text:
            assert "expired on 2026-08-31" in output
            assert "expires on August 31, 2026" in output
        if "tomorrow" in text:
            assert "tomorrow" in output
        if "around" in text:
            assert "around July 1, 2026" in output
        if "Conflict record" in text:
            assert "Conflict record A" in output and "Conflict record B" in output


def test_time_and_money_equivalence_rejects_different_values_or_events():
    cases = [
        "Meeting window started at 09:30 AM. Meeting window started at nine forty AM.",
        "Meeting window started at 09:30 AM. Meeting window paused at nine thirty AM.",
        "Meeting window paused at 11:45 AM. Meeting window paused at 11:45 PM.",
        "Refund exposure was $1,250,000. Refund exposure was one million three hundred thousand dollars.",
        "Penalty reserve increased from $320,000 to $480,000. Penalty reserve decreased from three hundred twenty thousand dollars to four hundred eighty thousand dollars.",
        "Penalty reserve increased from $320,000 to $480,000. Penalty reserve increased from four hundred eighty thousand dollars to three hundred twenty thousand dollars.",
    ]
    for text in cases:
        result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
        output = result["optimized_messages"][0].content
        if "09:30 AM" in text:
            assert "09:30 AM" in output
        if "$1,250,000" in text:
            assert "$1,250,000" in output
        if "11:45 AM" in text:
            assert "11:45 AM" in output
        if "nine forty" in text:
            assert "nine forty AM" in output
        if "paused at nine thirty" in text:
            assert "paused at nine thirty AM" in output
        if "11:45 PM" in text:
            assert "11:45 PM" in output
        if "one million three hundred" in text:
            assert "one million three hundred thousand dollars" in output
        if "decreased from three hundred" in text:
            assert "decreased from three hundred twenty thousand dollars" in output
        if "four hundred eighty thousand dollars to three hundred" in text:
            assert "four hundred eighty thousand dollars to three hundred twenty thousand dollars" in output


def test_list_frame_compresses_only_clean_same_type_same_state_items():
    safe = "Airport ORD was affected. Airport JFK was affected. Airport LAX was affected."
    result = optimize_messages([ChatMessage(role="user", content=safe)], "balanced", "dry-run", "dry-run")
    assert result["quality_gate_status"]["accepted"]
    assert result["optimized_messages"][0].content == "Affected airports: ORD, JFK, and LAX."

    unsafe_cases = [
        "Service CACHE is active. Service API is inactive.",
        "Policy POL-A was migrated. Policy POL-B was not migrated.",
        "Airport ORD was affected. Airport JFK was excluded.",
        "Finding F-1 is approved. Finding F-2 is pending. Finding F-3 is denied.",
        "The Q4 note pool is excluded from migration. The Q5 note pool is excluded from migration. Do not claim Q5 was reviewed.",
    ]
    for text in unsafe_cases:
        result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
        output = result["optimized_messages"][0].content
        for sentence in text.split(". "):
            key = sentence.strip().strip(".")
            if key:
                assert key in output


def test_formula_appearance_is_subsumed_by_exact_preservation_instruction():
    text = (
        "Formula x + y = 10 appears in the validation notes. "
        "Formula x + y = 10 must be preserved exactly. "
        "Formula P(A|B) appears in the probability notes. "
        "Formula P(A|B) must be preserved exactly."
    )
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert "Formula x + y = 10 must be preserved exactly." in output
    assert "Formula P(A|B) must be preserved exactly." in output
    assert "appears in" not in output


def test_same_state_negation_confirmation_compresses_safely():
    text = "No admin secrets were rotated. Admin secrets were not rotated. Do not claim admin secrets were rotated."
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert output.count("rotated") == 2
    assert "No admin secrets were rotated; do not claim they were rotated." in output
    assert "Admin secrets were not rotated." not in output


def test_do_not_claim_same_predicate_groups_without_losing_subjects():
    text = (
        "Do not claim API-GATEWAY is healthy. "
        "Do not claim CACHE-WARMER is healthy. "
        "Do not claim FINANCE-BRIDGE is healthy."
    )
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert output == "Do not claim API-GATEWAY, CACHE-WARMER, and FINANCE-BRIDGE are healthy."


def test_fact_warning_pair_fuses_without_losing_factual_state():
    text = "JWT signing key JWT-KID-77 is not exposed. Do not claim JWT-KID-77 was exposed."
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert output == "JWT signing key JWT-KID-77 is not exposed; do not claim it was exposed."
    assert "JWT signing key JWT-KID-77 is not exposed" in output


def test_fact_warning_fusion_supports_multiple_safe_entity_types():
    cases = [
        (
            "API keys were not exposed. Do not claim API keys were exposed.",
            "API keys were not exposed; do not claim they were exposed.",
        ),
        (
            "File /var/log/app.log was not modified. Do not claim /var/log/app.log was modified.",
            "File /var/log/app.log was not modified; do not claim it was modified.",
        ),
        (
            "Customer records were not deleted. Do not claim customer records were deleted.",
            "Customer records were not deleted; do not claim they were deleted.",
        ),
        (
            "Settlement bridge T+1 was not migrated. Do not claim T+1 was migrated.",
            "Settlement bridge T+1 was not migrated; do not claim it was migrated.",
        ),
        (
            "Rollback flag emergency_final_rollback is not deleted. Do not claim rollback flag emergency_final_rollback was deleted.",
            "Rollback flag emergency_final_rollback is not deleted; do not claim it was deleted.",
        ),
    ]
    for text, expected in cases:
        result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
        assert result["quality_gate_status"]["accepted"]
        assert result["optimized_messages"][0].content == expected


def test_fact_warning_deletion_without_fact_preservation_is_rejected():
    text = "JWT signing key JWT-KID-77 is not exposed. Do not claim JWT-KID-77 was rotated."
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert "JWT signing key JWT-KID-77 is not exposed." in output
    assert "Do not claim JWT-KID-77 was rotated." in output
    for trace in result["removed_or_changed_text"]:
        if trace.get("backend") == "cheap_layer":
            assert not any(
                item.get("removed_sentence", "").startswith("JWT signing key")
                and "Do not claim" in item.get("kept_sentence", "")
                for item in trace.get("semantic_removed", [])
            )


def test_fact_warning_fusion_rejects_mismatched_entities_predicates_and_conflicts():
    cases = [
        "File /tmp/a was not modified. Do not claim /tmp/a was deleted.",
        "File /tmp/a was not modified. Do not claim /tmp/b was modified.",
        "Conflict record A says Account ACCT-1 is not deleted. Do not claim ACCT-1 was deleted.",
        "Rule-10b-5 disclosure note is pending and is not approved. Do not claim Rule-10b-5 disclosure note is approved.",
        "Service CACHE is active and is not deleted. Do not claim Service CACHE was deleted.",
    ]
    for text in cases:
        result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
        output = result["optimized_messages"][0].content
        assert "; do not claim" not in output
        assert "Do not claim" in output


def test_fact_fact_and_instruction_instruction_compression_still_work():
    fact_dup = "API keys were not exposed. API keys were not exposed."
    result = optimize_messages([ChatMessage(role="user", content=fact_dup)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    assert result["quality_gate_status"]["accepted"]
    assert output == "API keys were not exposed."

    warnings = (
        "Do not claim API-GATEWAY is healthy. "
        "Do not claim CACHE-WARMER is healthy. "
        "Do not claim FINANCE-BRIDGE is healthy."
    )
    result = optimize_messages([ChatMessage(role="user", content=warnings)], "balanced", "dry-run", "dry-run")
    assert result["optimized_messages"][0].content == "Do not claim API-GATEWAY, CACHE-WARMER, and FINANCE-BRIDGE are healthy."


def test_refinements_preserve_dangerous_state_and_event_distinctions():
    cases = [
        "Recovery reached 79%. Recovery paused at 79%.",
        "Service CACHE is active. Service CACHE is inactive.",
        "HTTP-429 handling was patched. HTTP-503 handling was not patched.",
        "Policy POL-A was migrated. Policy POL-D was not migrated.",
        "CUSIP 9128285M8 was retained. CUSIP 9128285M8 was not deleted.",
        "Conflict record A says Account ACCT-9900 is active. Conflict record B says Account ACCT-9900 is inactive. Do not resolve this conflict.",
    ]
    for text in cases:
        result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
        output = result["optimized_messages"][0].content
        for word in ["reached", "paused", "active", "inactive", "patched", "not patched", "migrated", "not migrated", "retained", "not deleted"]:
            if word in text:
                assert word in output
        if "active. Service CACHE is inactive" in text:
            assert "active and is inactive" not in output


def test_cheap_layer_rejects_different_entities_with_same_state():
    text = "Region us-east-1 was affected. Region us-west-2 was affected."
    output, traces = cheap_layer_backend(text, "balanced")
    assert output == "Regions us-east-1 and us-west-2 were affected."
    assert traces[0]["accepted"]


def test_cheap_layer_rejects_same_entity_different_states():
    text = "Service CACHE-WARMER is degraded. Service CACHE-WARMER is not offline."
    output, traces = cheap_layer_backend(text, "balanced")
    assert output == "Service CACHE-WARMER is degraded and is not offline."
    assert traces[0]["accepted"]


def test_cheap_layer_pipeline_handles_obvious_instruction_duplicates():
    result = optimize_messages(
        [ChatMessage(role="user", content="Agent Instructions:\nDo not promise refund approval.\nDo not promise refund approval.\nTask:\nWrite the response.")],
        "balanced",
        "dry-run",
        "dry-run",
    )
    output = result["optimized_messages"][0].content
    assert "cheap_layer" in result["backend_used"]
    assert output.count("Do not promise refund approval.") == 1
    assert "Task:\nWrite the response." in output
    assert "\n\nTask:" in output


def test_cheap_layer_audit_has_required_fields():
    _, traces = cheap_layer_backend("Please help. Please help.", "balanced")
    trace = traces[0]
    for key in [
        "version",
        "before",
        "after",
        "surface_changes",
        "semantic_removed",
        "rejected_removals",
        "validation_gate",
        "missing_protected_facts",
        "missing_state_signatures",
        "final_action",
    ]:
        assert key in trace


def test_general_safe_list_frame_compression():
    cases = [
        (
            "Region us-east-1 was affected. Region us-west-2 was affected. Region eu-west-1 was affected.",
            "Regions us-east-1, us-west-2, and eu-west-1 were affected.",
        ),
        (
            "Account ACCT-5001 was reviewed. Account ACCT-5002 was reviewed. Account ACCT-5003 was reviewed.",
            "Accounts ACCT-5001, ACCT-5002, and ACCT-5003 were reviewed.",
        ),
        (
            "User USER-7001 data was migrated. User USER-7002 data was migrated. User USER-7003 data was migrated.",
            "Users USER-7001, USER-7002, and USER-7003 data were migrated.",
        ),
        (
            "Route ROUTE-A was migrated. Route ROUTE-B was migrated. Do not claim ROUTE-B was skipped.",
            "Routes ROUTE-A and ROUTE-B were migrated. Do not claim ROUTE-B was skipped.",
        ),
        (
            "API endpoint /v1/refunds was tested. API endpoint /v1/orders was tested.",
            "API endpoints /v1/refunds and /v1/orders were tested.",
        ),
        (
            "Support logged 42 timeout complaints. Support logged 17 audit-export complaints. Support logged 0 customer-data leak complaints.",
            "Support logged 42 timeout complaints, 17 audit-export complaints, and 0 customer-data leak complaints.",
        ),
        (
            "Cluster CLUSTER-A was restarted. Cluster CLUSTER-B was restarted. Cluster CLUSTER-C was restarted. Cluster CLUSTER-D was restarted.",
            "Clusters CLUSTER-A, CLUSTER-B, CLUSTER-C, and CLUSTER-D were restarted.",
        ),
    ]
    for text, expected in cases:
        output, traces = cheap_layer_backend(text, "balanced")
        assert output == expected
        assert traces[0]["accepted"]
        assert "general_list_frame_compression" in [c.get("type") for c in traces[0].get("surface_changes", [])]


def test_general_list_frame_removes_duplicate_source_sentences():
    text = (
        "Account ACCT-8001 was reviewed. Account ACCT-8002 was reviewed. "
        "Account ACCT-8003 was reviewed. Account ACCT-8004 was reviewed. "
        "Account ACCT-8001 was reviewed. Account ACCT-8002 was reviewed. "
        "Account ACCT-8003 was reviewed. Account ACCT-8004 was reviewed."
    )
    output, traces = cheap_layer_backend(text, "balanced")
    assert output == "Accounts ACCT-8001, ACCT-8002, ACCT-8003, and ACCT-8004 were reviewed."
    assert traces[0]["accepted"]
    assert output.count("Account ACCT-8001 was reviewed.") == 0
    assert output.count("Account ACCT-8002 was reviewed.") == 0


def test_existing_list_source_cleanup_removes_covered_individual_lines():
    cases = [
        (
            "Accounts ACCT-8001, ACCT-8002, ACCT-8003, and ACCT-8004 were reviewed. "
            "Account ACCT-8001 was reviewed. Account ACCT-8002 was reviewed. "
            "Account ACCT-8003 was reviewed. Account ACCT-8004 was reviewed.",
            "Accounts ACCT-8001, ACCT-8002, ACCT-8003, and ACCT-8004 were reviewed.",
        ),
        (
            "Routes ROUTE-P, ROUTE-Q, and ROUTE-R were migrated. "
            "Route ROUTE-P was migrated. Route ROUTE-Q was migrated. Route ROUTE-R was migrated. "
            "Route ROUTE-S was not migrated. Route ROUTE-S was excluded. "
            "Do not claim ROUTE-Q and ROUTE-R were skipped.",
            "Routes ROUTE-P, ROUTE-Q, and ROUTE-R were migrated. "
            "Route ROUTE-S was not migrated. Route ROUTE-S was excluded. "
            "Do not claim ROUTE-Q and ROUTE-R were skipped.",
        ),
    ]
    for text, expected in cases:
        output, traces = cheap_layer_backend(text, "balanced")
        assert output == expected
        assert traces[0]["accepted"]
        assert any(
            c.get("type") in {"existing_list_source_cleanup", "general_list_frame_compression"}
            for c in traces[0].get("surface_changes", [])
        )


def test_general_list_frame_keeps_mixed_or_item_specific_warnings_safe():
    mixed = "Region us-east-1 was affected. Region us-west-2 was excluded. Region eu-west-1 was affected."
    output, traces = cheap_layer_backend(mixed, "balanced")
    assert output == "Regions us-east-1 and eu-west-1 were affected. Region us-west-2 was excluded."
    assert traces[0]["accepted"]

    unsafe_cases = [
        "Route ROUTE-A was migrated. Route ROUTE-B was migrated. Do not claim ROUTE-B was migrated.",
        "Policy POL-A was migrated. Policy POL-B was not migrated.",
        "Recovery reached 79%. Recovery paused at 79%.",
        "Conflict record A says Account ACCT-1 is active. Conflict record B says Account ACCT-1 is inactive. Do not resolve this conflict.",
    ]
    for text in unsafe_cases:
        output, traces = cheap_layer_backend(text, "balanced")
        assert "general_list_frame_compression" not in [c.get("type") for c in traces[0].get("surface_changes", [])]
        for required in ["not migrated", "reached", "paused", "Conflict record", "Do not claim ROUTE-B was migrated"]:
            if required in text:
                assert required in output


def test_item_extra_fact_survives_list_compression_and_warning():
    text = (
        "Policy POL-101 was reviewed. Policy POL-102 was reviewed. "
        "Policy POL-103 was reviewed. Policy POL-104 was reviewed. "
        "Policy POL-103 was also flagged for legal review. "
        "Do not claim POL-103 was fully cleared."
    )
    output, traces = cheap_layer_backend(text, "balanced")
    assert output == (
        "Policies POL-101, POL-102, POL-103, and POL-104 were reviewed. "
        "Policy POL-103 was also flagged for legal review. "
        "Do not claim POL-103 was fully cleared."
    )
    assert traces[0]["accepted"]
    assert "Policy POL-103 was also flagged for legal review." in output


def test_semantic_dedupe_rejects_fact_replaced_by_warning():
    text = "Policy POL-103 was also flagged for legal review. Do not claim POL-103 was fully cleared."
    output, traces = cheap_layer_backend(text, "balanced")
    assert "Policy POL-103 was also flagged for legal review." in output
    assert "Do not claim POL-103 was fully cleared." in output
    for trace in traces:
        assert not any(
            item.get("removed_sentence") == "Policy POL-103 was also flagged for legal review."
            and item.get("kept_sentence") == "Do not claim POL-103 was fully cleared."
            for item in trace.get("semantic_removed", [])
        )


def test_alias_relation_blocks_route_list_flattening():
    text = (
        "Route ROUTE-ALIAS-1 is also called primary-failover-route. "
        "Route ROUTE-ALIAS-1 was migrated. "
        "Route primary-failover-route was migrated. "
        "Do not count ROUTE-ALIAS-1 and primary-failover-route as two separate routes."
    )
    output, traces = cheap_layer_backend(text, "balanced")
    assert "Routes ROUTE-ALIAS-1 and primary-failover-route were migrated." not in output
    assert "Route ROUTE-ALIAS-1 was migrated." in output
    assert "Route primary-failover-route was migrated." in output
    assert "Do not count ROUTE-ALIAS-1 and primary-failover-route as two separate routes." in output


def test_drained_queue_list_preserves_extra_queue_state():
    text = (
        "Queue QUEUE-1 was drained. Queue QUEUE-2 was drained. "
        "Queue QUEUE-3 was drained. Queue QUEUE-4 was drained. "
        "Queue QUEUE-2 was later paused. Do not claim QUEUE-2 stayed active after draining. "
        "Queue QUEUE-5 was not drained. Queue QUEUE-5 was excluded."
    )
    output, traces = cheap_layer_backend(text, "balanced")
    assert "Queues QUEUE-1, QUEUE-3, and QUEUE-4 were drained." in output
    assert "Queue QUEUE-2 was drained." in output
    assert "Queue QUEUE-2 was later paused." in output
    assert "Queue QUEUE-5 was not drained." in output
    assert "Queue QUEUE-5 was excluded." in output


def test_extra_item_state_excludes_item_from_plain_list():
    text = (
        "Cluster CLUSTER-A was restarted. Cluster CLUSTER-B was restarted. "
        "Cluster CLUSTER-C was restarted. Cluster CLUSTER-D was restarted. "
        "Cluster CLUSTER-F was restarted. Cluster CLUSTER-F is still degraded. "
        "Do not list-compress CLUSTER-F with fully healthy restarted clusters."
    )
    output, traces = cheap_layer_backend(text, "balanced")
    assert "Clusters CLUSTER-A, CLUSTER-B, CLUSTER-C, and CLUSTER-D were restarted." in output
    assert "Cluster CLUSTER-F was restarted." in output
    assert "Cluster CLUSTER-F is still degraded." in output
    assert "CLUSTER-F were restarted" not in output
    assert traces[0]["accepted"]


def test_positive_extra_item_state_survives_with_warning_across_entity_types():
    cases = [
        (
            "Device DEVICE-11 was scanned. Device DEVICE-12 was scanned. Device DEVICE-13 was scanned. "
            "Device DEVICE-14 was scanned. Device DEVICE-13 was also quarantined. Do not claim DEVICE-13 was clean.",
            [
                "Devices DEVICE-11, DEVICE-12, and DEVICE-14 were scanned.",
                "Device DEVICE-13 was scanned.",
                "Device DEVICE-13 was also quarantined.",
                "Do not claim DEVICE-13 was clean.",
            ],
            "Device DEVICE-13 was also quarantined.",
        ),
        (
            "Account ACCT-1 was reviewed. Account ACCT-2 was reviewed. Account ACCT-3 was reviewed. "
            "Account ACCT-3 was also placed on watchlist. Do not claim ACCT-3 was fully cleared.",
            [
                "Accounts ACCT-1 and ACCT-2 were reviewed.",
                "Account ACCT-3 was reviewed.",
                "Account ACCT-3 was also placed on watchlist.",
                "Do not claim ACCT-3 was fully cleared.",
            ],
            "Account ACCT-3 was also placed on watchlist.",
        ),
        (
            "Policy POL-1 was reviewed. Policy POL-2 was reviewed. Policy POL-3 was reviewed. "
            "Policy POL-3 was also flagged for legal review. Do not claim POL-3 was fully cleared.",
            [
                "Policies POL-1, POL-2, and POL-3 were reviewed.",
                "Policy POL-3 was also flagged for legal review.",
                "Do not claim POL-3 was fully cleared.",
            ],
            "Policy POL-3 was also flagged for legal review.",
        ),
        (
            "Cluster CLUSTER-A was restarted. Cluster CLUSTER-B was restarted. Cluster CLUSTER-C was restarted. "
            "Cluster CLUSTER-C is still degraded. Do not claim CLUSTER-C is healthy.",
            [
                "Clusters CLUSTER-A and CLUSTER-B were restarted.",
                "Cluster CLUSTER-C was restarted.",
                "Cluster CLUSTER-C is still degraded.",
                "Do not claim CLUSTER-C is healthy.",
            ],
            "Cluster CLUSTER-C is still degraded.",
        ),
        (
            "Queue QUEUE-1 was drained. Queue QUEUE-2 was drained. Queue QUEUE-3 was drained. "
            "Queue QUEUE-2 was later paused. Do not claim QUEUE-2 stayed active after draining.",
            [
                "Queues QUEUE-1 and QUEUE-3 were drained.",
                "Queue QUEUE-2 was drained.",
                "Queue QUEUE-2 was later paused.",
                "Do not claim QUEUE-2 stayed active after draining.",
            ],
            "Queue QUEUE-2 was later paused.",
        ),
        (
            "User USER-1 data was migrated. User USER-2 data was migrated. User USER-3 data was migrated. "
            "User USER-2 consent flag was not migrated. Do not claim USER-2 consent flag was migrated.",
            [
                "Users USER-1 and USER-3 data were migrated.",
                "User USER-2 data was migrated.",
                "User USER-2 consent flag was not migrated; do not claim it was migrated.",
            ],
            "User USER-2 consent flag was not migrated",
        ),
        (
            "Route ROUTE-A was migrated. Route ROUTE-B was migrated. Route ROUTE-C was migrated. "
            "Route ROUTE-B is also called primary-route. Do not count ROUTE-B and primary-route as two separate routes.",
            [
                "Routes ROUTE-A and ROUTE-C were migrated.",
                "Route ROUTE-B was migrated.",
                "Route ROUTE-B is also called primary-route.",
                "Do not count ROUTE-B and primary-route as two separate routes.",
            ],
            "Route ROUTE-B is also called primary-route.",
        ),
        (
            "Service SVC-A was tested. Service SVC-B was tested. Service SVC-C was tested. "
            "Service SVC-B is blocked. Do not claim SVC-B is healthy.",
            [
                "Services SVC-A and SVC-C were tested.",
                "Service SVC-B was tested.",
                "Service SVC-B is blocked.",
                "Do not claim SVC-B is healthy.",
            ],
            "Service SVC-B is blocked.",
        ),
    ]
    for text, expected_parts, protected_fact in cases:
        output, traces = cheap_layer_backend(text, "balanced")
        assert traces[0]["accepted"]
        for expected in expected_parts:
            assert expected in output
        assert protected_fact in output
        assert "semantic duplicate rejected by safety firewall" in str(traces[0]) or "general_list_frame_compression" in str(traces[0])


def test_fact_warning_fusion_uses_pronoun_auxiliary_agreement():
    text = "Stage-III label STAGE-III-7 was not migrated. Do not claim Stage-III label STAGE-III-7 were migrated."
    output, traces = cheap_layer_backend(text, "balanced")
    assert output == "Stage-III label STAGE-III-7 was not migrated; do not claim it was migrated."
    assert traces[0]["accepted"]
    assert "it were" not in output
