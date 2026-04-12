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


async def play_scenario(session, scenario: dict) -> dict:
    """Play a scenario through the session and collect results.

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

        # Small delay to let async agents react
        await asyncio.sleep(0.3)

        # Check if new actions appeared
        new_actions = session.collector.actions[len(results["actions"]):]
        if new_actions:
            for action in new_actions:
                results["action_after_segment"][i] = action
            results["actions"].extend(new_actions)

    # Final wait for any remaining async responses
    await asyncio.sleep(5.0)
    remaining = session.collector.actions[len(results["actions"]):]
    results["actions"].extend(remaining)

    results["end_time"] = time.time()
    results["duration_real"] = results["end_time"] - results["start_time"]
    results["final_state"] = await session.shared_state.read()

    return results


def check_expectations(results: dict, expectations: list[dict]) -> list[dict]:
    """Check scenario expectations against actual results."""
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
                matched = False
                for a in matching:
                    text = getattr(a, "text", "").lower()
                    if any(kw.lower() in text for kw in exp["contains_any"]):
                        matched = True
                        break
                checks.append({
                    "name": f"action {action_type} contains keywords",
                    "passed": matched,
                    "detail": f"Found {len(matching)} {action_type} actions",
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

    # Play the scenario
    results = await play_scenario(session, scenario)

    # Check expectations
    checks = check_expectations(results, expectations)

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
