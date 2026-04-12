"""Scenario-based smoke tests — loads YAML cartridges and plays them through."""

import asyncio
import os
import time
from pathlib import Path

import pytest
import yaml

from actions.queue import ChatAction, SpeakAction

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


def load_scenario(name: str) -> dict:
    path = SCENARIOS_DIR / name
    with open(path) as f:
        return yaml.safe_load(f)


def get_all_scenarios() -> list[str]:
    return sorted(f.name for f in SCENARIOS_DIR.glob("*.yaml"))


def _could_trigger(text: str, keywords: list[str] = None) -> bool:
    """Check if a segment text could trigger the Watcher."""
    keywords = keywords or ["nilo", "hey nilo"]
    return any(kw in text.lower() for kw in keywords)


async def play_scenario(session, scenario: dict) -> dict:
    """Play a scenario through the session and collect results.

    Waits longer after trigger segments (Claude CLI needs ~15-20s).
    Returns a report dict with what happened.
    """
    segments = scenario.get("segments", [])
    results = {
        "segments_played": 0,
        "triggers": [],
        "actions": [],
        "action_after_segment": {},
        "start_time": time.time(),
    }

    for i, seg in enumerate(segments):
        await session.inject_segment(
            speaker=seg["who"],
            text=seg["says"],
            timestamp=float(seg.get("t", i * 10)),
        )
        results["segments_played"] += 1

        is_trigger = _could_trigger(seg["says"])

        if is_trigger:
            # Wait for Quick-Ack + Deep Agent (Claude CLI ~15-20s)
            await session.collector.wait_for_actions(
                count=len(results["actions"]) + 1,
                timeout=30.0,
            )
        else:
            # Small delay for non-trigger segments
            await asyncio.sleep(0.3)

        # Collect new actions
        new_actions = session.collector.actions[len(results["actions"]):]
        if new_actions:
            for action in new_actions:
                results["action_after_segment"][i] = action
            results["actions"].extend(new_actions)

    # Final wait for any remaining async responses (Deep Agent may still be running)
    final_actions = await session.collector.wait_for_actions(
        count=len(results["actions"]) + 1,
        timeout=25.0,
    )
    remaining = final_actions[len(results["actions"]):]
    results["actions"].extend(remaining)

    results["end_time"] = time.time()
    results["duration_real"] = results["end_time"] - results["start_time"]
    results["final_state"] = await session.shared_state.read()

    return results


async def check_expectations(results: dict, expectations: list[dict], scenario: dict = None) -> list[dict]:
    """Check scenario expectations against actual results.
    Uses Judge LLM as fallback when keyword assertions fail.
    """
    from tests.smoke.judge import judge_response

    checks = []

    for exp in expectations:
        exp_type = exp.get("type")

        if exp_type == "no_action":
            before = exp.get("before_segment")
            if before is not None:
                early_actions = {
                    seg_idx: a for seg_idx, a in results["action_after_segment"].items()
                    if seg_idx < before
                }
                passed = len(early_actions) == 0
                checks.append({
                    "name": f"no_action before segment {before}",
                    "passed": passed,
                    "detail": f"Found {len(early_actions)} early actions" if not passed else "OK",
                    "reason": exp.get("reason", ""),
                })
            else:
                # No action at all in entire scenario
                passed = len(results["actions"]) == 0
                checks.append({
                    "name": "no_action (entire scenario)",
                    "passed": passed,
                    "detail": f"Found {len(results['actions'])} actions" if not passed else "OK",
                    "reason": exp.get("reason", ""),
                })

        elif exp_type == "trigger":
            after = exp.get("after_segment", 0)
            triggered = any(
                seg_idx >= after for seg_idx in results["action_after_segment"]
            )
            checks.append({
                "name": f"trigger after segment {after}",
                "passed": triggered,
                "detail": "Triggered" if triggered else "No trigger detected",
                "reason": exp.get("reason", ""),
            })

        elif exp_type == "quick_ack":
            after = exp.get("after_segment", 0)
            ack_actions = [
                a for seg_idx, a in results["action_after_segment"].items()
                if seg_idx >= after and isinstance(a, ChatAction)
            ]
            passed = len(ack_actions) > 0
            checks.append({
                "name": f"quick_ack after segment {after}",
                "passed": passed,
                "detail": f"Got: {ack_actions[0].text[:50]}" if ack_actions else "No Quick-Ack",
            })

        elif exp_type == "state_updated":
            field = exp.get("field", "")
            state = results["final_state"]
            value = state.get(field, "")

            if "contains" in exp:
                passed = exp["contains"].lower() in str(value).lower()
            elif "contains_any" in exp:
                passed = any(kw.lower() in str(value).lower() for kw in exp["contains_any"])
            elif "min_count" in exp:
                passed = isinstance(value, list) and len(value) >= exp["min_count"]
            else:
                passed = bool(value)

            checks.append({
                "name": f"state_updated: {field}",
                "passed": passed,
                "detail": f"Value: {str(value)[:80]}" if value else "Empty",
            })

        elif exp_type == "action":
            action_type = exp.get("action_type", "chat")
            type_map = {"chat": ChatAction, "speak": SpeakAction}
            target_type = type_map.get(action_type, ChatAction)

            matching = [a for a in results["actions"] if isinstance(a, target_type)]

            if "contains_any" in exp:
                # Step 1: Keyword match (free)
                matched = False
                matched_text = ""
                for a in matching:
                    text = getattr(a, "text", "")
                    if any(kw.lower() in text.lower() for kw in exp["contains_any"]):
                        matched = True
                        matched_text = text
                        break

                if matched:
                    checks.append({
                        "name": f"action {action_type} contains keywords",
                        "passed": True,
                        "detail": f"Keyword match: {matched_text[:50]}",
                    })
                elif matching:
                    # Step 2: Judge LLM fallback (cheap)
                    all_texts = " | ".join(getattr(a, "text", "") for a in matching)
                    trigger_text = ""
                    if scenario:
                        segs = scenario.get("segments", [])
                        triggers = [s for s in segs if "nilo" in s.get("says", "").lower()]
                        if triggers:
                            trigger_text = triggers[-1].get("says", "")

                    verdict = await judge_response(
                        question=trigger_text or "Meeting-Zusammenfassung",
                        response=all_texts,
                        expected_keywords=exp["contains_any"],
                    )
                    checks.append({
                        "name": f"action {action_type} (judge)",
                        "passed": verdict["passed"],
                        "detail": f"{verdict['method']}: {verdict['detail'][:60]}",
                    })
                else:
                    checks.append({
                        "name": f"action {action_type} contains keywords",
                        "passed": False,
                        "detail": f"No {action_type} actions found",
                    })
            else:
                checks.append({
                    "name": f"action {action_type} exists",
                    "passed": len(matching) > 0,
                    "detail": f"Found {len(matching)}",
                })

    return checks


# --- Parametrized test for all scenarios ---

@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_file", get_all_scenarios())
async def test_scenario(session, scenario_file):
    """Run a scenario cartridge and check all expectations."""
    scenario = load_scenario(scenario_file)
    meta = scenario.get("meta", {})
    expectations = scenario.get("expect", [])

    print(f"\n{'='*60}")
    print(f"Scenario: {meta.get('title', scenario_file)}")
    print(f"Difficulty: {meta.get('difficulty', '?')}")
    print(f"{'='*60}")

    # Init Judge LLM
    from tests.smoke.judge import set_judge_llm
    if "quick" in session.llm_providers:
        set_judge_llm(session.llm_providers["quick"])

    # Play the scenario
    results = await play_scenario(session, scenario)

    # Check expectations (with Judge LLM fallback)
    checks = await check_expectations(results, expectations, scenario)

    # Print report
    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)
    print(f"\nResults: {passed}/{total} checks passed")
    print(f"Actions: {len(results['actions'])}")
    print(f"Duration: {results['duration_real']:.1f}s")

    for check in checks:
        icon = "PASS" if check["passed"] else "FAIL"
        print(f"  [{icon}] {check['name']}: {check.get('detail', '')}")
        if check.get("reason"):
            print(f"         Reason: {check['reason']}")

    # Assert all checks pass
    failed = [c for c in checks if not c["passed"]]
    assert not failed, f"{len(failed)} checks failed: {[c['name'] for c in failed]}"
