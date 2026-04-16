"""Edge-case stress test for Complexity Gate v2."""

from __future__ import annotations

from aifactory.core.complexity_gate import ComplexityGate

gate = ComplexityGate()

scenarios: list[tuple[str, str]] = [
    # === BYPASS (math) ===
    ("2 + 2", "bypass"),
    ("What is 0.1 + 0.2?", "bypass"),
    ("999999 * 999999", "bypass"),
    ("What is -5 + -3?", "bypass"),
    ("sqrt(0)", "bypass"),
    ("factorial(0)", "bypass"),
    ("2^32", "bypass"),
    ("100 / 0", "pipeline"),  # div by zero
    ("twice 50", "bypass"),
    ("half of 1000", "bypass"),
    ("a dozen times a dozen", "bypass"),
    ("What is twenty three plus seventy seven?", "bypass"),
    # === BYPASS (static fact) ===
    ("Capital of France?", "bypass"),
    ("What is the capital of India?", "bypass"),
    ("CAPITAL OF JAPAN?", "bypass"),
    ("what's the capital of Germany?", "bypass"),
    ("How many continents?", "bypass"),
    ("How many planets in the solar system?", "bypass"),
    ("Tallest mountain in the world?", "bypass"),
    ("What is the speed of light?", "bypass"),
    # === FAST LLM ===
    ("Who wrote Hamlet?", "fast_llm"),
    ("Define entropy", "fast_llm"),
    ("What is machine learning?", "fast_llm"),
    ("Where is the Sahara desert?", "fast_llm"),
    ("When did World War 2 end?", "fast_llm"),
    ("Is Pluto a planet?", "fast_llm"),
    ("Translate hello to Japanese", "fast_llm"),
    ("Explain the theory of relativity", "fast_llm"),
    ("Name the 7 wonders of the world", "fast_llm"),
    ("How old is the universe?", "fast_llm"),
    ("What does DNA stand for?", "fast_llm"),
    ("Summarize the French Revolution", "fast_llm"),
    # === FULL PIPELINE ===
    ("Build a FastAPI todo app with auth", "pipeline"),
    ("Create a pull request on GitHub", "pipeline"),
    ("Send an email to the team", "pipeline"),
    ("Research AI trends and write a report", "pipeline"),
    ("Deploy the service to Kubernetes", "pipeline"),
    ("Scrape the top 100 HN posts", "pipeline"),
    ("Create a Jira ticket for this bug", "pipeline"),
    ("Design a microservices architecture", "pipeline"),
    ("Refactor the payment module", "pipeline"),
    ("First build the API, then write tests", "pipeline"),
    # === TRICKY EDGE CASES ===
    ("How to run Docker containers?", "fast_llm"),
    ("What is 2 + 2 and build an app?", "pipeline"),
    ("Tell me about Python", "fast_llm"),
    ("Hi", "pipeline"),
    ("", "empty"),
    ("Can you help me?", "fast_llm"),
    ("Should I use React or Vue?", "fast_llm"),
    ("Write a poem about love", "pipeline"),
    ("Who built the Great Wall and deploy it", "pipeline"),
    ("What is the best database for my app?", "fast_llm"),
    ("List all my GitHub issues", "pipeline"),
    ("Explain how to build a REST API", "pipeline"),
    ("42", "bypass"),
    ("pi", "bypass"),
    ("hello world", "pipeline"),
]

print(f"{'TASK':<52} {'EXPECT':<10} {'ACTUAL':<10} {'CONF':>5}  {'LABELS':<30} {'OK'}")
print("─" * 130)

passed = 0
failed = 0
details: list[str] = []

for task, expected in scenarios:
    d = gate.evaluate(task)
    if d.reason == "empty_task":
        actual = "empty"
    elif d.bypass:
        actual = "bypass"
    elif d.fast_llm:
        actual = "fast_llm"
    else:
        actual = "pipeline"

    ok = "✅" if actual == expected else "❌"
    if actual != expected:
        failed += 1
        details.append(
            f"  ❌ {task!r:<48} expected={expected:<10} got={actual:<10} conf={d.confidence:.2f} labels={d.matched_labels}"  # noqa: E501
        )
    else:
        passed += 1

    labels = ", ".join(d.matched_labels) if d.matched_labels else d.method or "—"
    ans = f" → {d.answer}" if d.answer else ""
    print(
        f"{(task or '(empty)'):<52} {expected:<10} {actual:<10} {d.confidence:>5.2f}  {labels:<30} {ok}{ans}"  # noqa: E501
    )

print("─" * 130)
print(f"\n✅ {passed} passed  |  ❌ {failed} failed  |  Total: {passed + failed}")

if details:
    print("\n── FAILURES ──────────────────────────────────────────────────")
    for line in details:
        print(line)
