import copy
import json
import unittest

from auditor.context import compact
from auditor.settings import ContextSettings


def conversation(results, size):
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "audit this"},
    ]
    for index in range(results):
        call_id = f"call-{index}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": f"tool_{index}", "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call_id, "content": f"{index}" * size})
    return messages


def total(messages):
    return sum(len(m.get("content") or "") for m in messages)


class CompactTests(unittest.TestCase):
    def test_older_results_become_stubs_and_recent_ones_stay_whole(self):
        original = conversation(5, 400)
        result = compact(original, ContextSettings(2, 10**9))
        tools = [m for m in result if m["role"] == "tool"]
        for message in tools[:3]:
            stub = json.loads(message["content"])
            self.assertTrue(stub["compacted"])
            self.assertEqual(stub["chars"], 400)
            self.assertIn("re-run", stub["note"])
        self.assertEqual([m["content"] for m in tools[3:]], ["3" * 400, "4" * 400])
        self.assertEqual(json.loads(tools[0]["content"])["tool"], "tool_0")

    def test_system_and_first_user_message_are_never_changed(self):
        result = compact(conversation(6, 5000), ContextSettings(1, 100))
        self.assertEqual(result[0]["content"], "system prompt")
        self.assertEqual(result[1]["content"], "audit this")

    def test_tool_calls_keep_their_replies(self):
        result = compact(conversation(8, 3000), ContextSettings(2, 4000))
        asked = [c["id"] for m in result for c in m.get("tool_calls") or []]
        answered = [m["tool_call_id"] for m in result if m["role"] == "tool"]
        self.assertEqual(asked, answered)

    def test_protected_results_are_stubbed_oldest_first_to_fit_the_budget(self):
        result = compact(conversation(10, 1000), ContextSettings(4, 2500))
        self.assertLessEqual(total(result), 2500)
        tools = [m for m in result if m["role"] == "tool"]
        self.assertEqual(tools[-1]["content"], "9" * 1000)

    def test_input_is_not_modified_and_compaction_is_idempotent(self):
        original = conversation(6, 500)
        snapshot = copy.deepcopy(original)
        settings = ContextSettings(2, 3000)
        once = compact(original, settings)
        self.assertEqual(original, snapshot)
        self.assertEqual(compact(once, settings), once)

    def test_results_shorter_than_their_stub_are_left_alone(self):
        result = compact(conversation(4, 3), ContextSettings(1, 10**9))
        self.assertEqual([m["content"] for m in result if m["role"] == "tool"][0], "000")

    def test_fewer_results_than_the_keep_count_are_untouched(self):
        original = conversation(2, 400)
        self.assertEqual(compact(original, ContextSettings(6, 10**9)), original)


if __name__ == "__main__":
    unittest.main()
