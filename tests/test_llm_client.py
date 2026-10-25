import dataclasses
import io
import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from auditor.llm_client import LLMClient, LLMError
from tests.support import default_settings

KEY = "sk-test-secret-key-123"
ENVIRON = {"OPENAI_API_KEY": KEY, "OPENAI_MODEL": "test-model"}
GOOD = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()


def http_error(code, headers=None):
    return urllib.error.HTTPError(
        "https://api.example.com/v1/chat/completions",
        code,
        "error",
        headers or {},
        io.BytesIO(b"provider body with details"),
    )


def response(body=GOOD):
    handle = MagicMock()
    handle.__enter__.return_value.read.return_value = body
    return handle


class LLMClientTests(unittest.TestCase):
    def setUp(self):
        self.settings = default_settings().llm
        self.sleeps = []
        self.client = self.make_client()

    def make_client(self, settings=None, environ=None):
        return LLMClient(
            settings or self.settings,
            environ or ENVIRON,
            sleep=self.sleeps.append,
            random_fraction=lambda: 0.5,
        )

    def chat(self, side_effect, client=None):
        with patch("auditor.llm_client._opener.open", side_effect=side_effect) as urlopen:
            try:
                return (client or self.client).chat([{"role": "user", "content": "hi"}], []), urlopen
            except LLMError as exc:
                exc.urlopen = urlopen
                raise

    def test_returns_the_assistant_message(self):
        message, urlopen = self.chat([response()])
        self.assertEqual(message["content"], "ok")
        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(self.sleeps, [])

    def test_request_carries_temperature_model_and_tools(self):
        with patch("auditor.llm_client._opener.open", return_value=response()) as urlopen:
            self.client.chat([], [{"type": "function"}], tool_choice="required")
        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["temperature"], self.settings.temperature)
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["tool_choice"], "required")
        self.assertEqual(request.full_url, self.settings.default_base_url + "/chat/completions")

    def test_each_retryable_status_is_retried(self):
        for code in (429, 500, 502, 503, 504):
            self.sleeps.clear()
            message, urlopen = self.chat([http_error(code), response()])
            self.assertEqual(urlopen.call_count, 2, code)
            self.assertEqual(len(self.sleeps), 1, code)
            self.assertEqual(message["content"], "ok")

    def test_url_error_and_timeouts_are_retried(self):
        for failure in (urllib.error.URLError("down"), TimeoutError(), urllib.error.URLError(TimeoutError())):
            self.sleeps.clear()
            _, urlopen = self.chat([failure, response()])
            self.assertEqual(urlopen.call_count, 2)

    def test_client_error_is_not_retried(self):
        with self.assertRaisesRegex(LLMError, "HTTP 400") as caught:
            self.chat([http_error(400), response()])
        self.assertEqual(caught.exception.urlopen.call_count, 1)
        self.assertEqual(self.sleeps, [])

    def test_backoff_is_exponential_with_full_jitter_and_capped(self):
        settings = dataclasses.replace(
            self.settings, max_retries=4, backoff_base_seconds=1.0, backoff_max_seconds=3.0
        )
        client = self.make_client(settings)
        with self.assertRaises(LLMError):
            self.chat([http_error(503)] * 5, client)
        self.assertEqual(self.sleeps, [0.5, 1.0, 1.5, 1.5])

    def test_retry_after_is_honoured_and_capped(self):
        settings = dataclasses.replace(self.settings, backoff_max_seconds=10.0)
        client = self.make_client(settings)
        self.chat([http_error(429, {"Retry-After": "7"}), response()], client)
        self.chat([http_error(429, {"Retry-After": "999"}), response()], client)
        self.assertEqual(self.sleeps, [7.0, 10.0])

    def test_non_numeric_retry_after_falls_back_to_backoff(self):
        self.chat([http_error(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), response()])
        self.assertEqual(self.sleeps, [0.5 * self.settings.backoff_base_seconds])

    def test_exhaustion_message_has_no_secrets(self):
        with self.assertRaises(LLMError) as caught:
            self.chat([http_error(503)] * (self.settings.max_retries + 1))
        text = str(caught.exception)
        self.assertEqual(text, "LLM provider request failed with HTTP 503.")
        self.assertNotIn(KEY, text)
        self.assertNotIn("provider body", text)
        self.assertEqual(len(self.sleeps), self.settings.max_retries)

    def test_zero_retries_fails_on_first_error(self):
        client = self.make_client(dataclasses.replace(self.settings, max_retries=0))
        with self.assertRaises(LLMError):
            self.chat([http_error(503), response()], client)
        self.assertEqual(self.sleeps, [])

    def test_timeout_exhaustion_message(self):
        with self.assertRaisesRegex(LLMError, "timed out"):
            self.chat([TimeoutError()] * (self.settings.max_retries + 1))

    def test_malformed_responses_are_rejected(self):
        bodies = (
            b"not json",
            b"[]",
            b"{}",
            json.dumps({"choices": []}).encode(),
            json.dumps({"choices": [{"message": "text"}]}).encode(),
            json.dumps({"choices": ["x"]}).encode(),
        )
        for body in bodies:
            with self.assertRaises(LLMError, msg=body):
                self.chat([response(body)])

    def test_oversized_response_is_rejected(self):
        settings = dataclasses.replace(self.settings, max_response_bytes=10)
        with self.assertRaisesRegex(LLMError, "size limit"):
            self.chat([response(GOOD)], self.make_client(settings))

    def test_missing_key_or_model(self):
        with self.assertRaisesRegex(LLMError, "OPENAI_API_KEY"):
            self.make_client(environ={"OPENAI_MODEL": "m"})
        with self.assertRaisesRegex(LLMError, "OPENAI_MODEL"):
            self.make_client(environ={"OPENAI_API_KEY": KEY})

    def test_base_url_uses_settings_validator(self):
        for url in ("http://api.example.com/v1", "https://user:pw@api.example.com/v1", "https://a.com/v1?x=1"):
            with self.assertRaisesRegex(LLMError, "OPENAI_BASE_URL"):
                self.make_client(environ=dict(ENVIRON, OPENAI_BASE_URL=url))

    def test_base_url_override_and_trailing_slash(self):
        client = self.make_client(environ=dict(ENVIRON, OPENAI_BASE_URL="https://api.example.com/v1/"))
        self.assertEqual(client.base_url, "https://api.example.com/v1")


class LlmSettingsTests(unittest.TestCase):
    def test_new_defaults(self):
        llm = default_settings().llm
        self.assertEqual(
            (llm.temperature, llm.max_retries, llm.backoff_base_seconds, llm.backoff_max_seconds),
            (0.1, 4, 1.0, 30.0),
        )

    def test_temperature_out_of_range_is_a_config_error(self):
        from auditor.settings import ConfigError, load_settings

        with patch.dict("os.environ", {"AUDITOR_LLM_TEMPERATURE": "2.5"}):
            with self.assertRaises(ConfigError):
                load_settings()


if __name__ == "__main__":
    unittest.main()


class TemperatureRefusalTests(unittest.TestCase):
    """Some models reject `temperature`; the client then drops it and carries on."""

    def setUp(self):
        self.client = LLMClient(default_settings().llm, ENVIRON, sleep=lambda _: None)

    @staticmethod
    def refusal(text):
        return urllib.error.HTTPError(
            "https://api.example.com/v1/chat/completions", 400, "bad request", {}, io.BytesIO(text)
        )

    def send(self, side_effect):
        with patch("auditor.llm_client._opener.open", side_effect=side_effect) as opener:
            message = self.client.chat([{"role": "user", "content": "hi"}], [])
        return message, opener

    @staticmethod
    def body_of(call):
        return json.loads(call.args[0].data)

    def test_retries_once_without_temperature_and_remembers(self):
        message, opener = self.send([self.refusal(b'{"error": {"message": "`temperature` is deprecated"}}'), response()])
        self.assertEqual(message["content"], "ok")
        self.assertIn("temperature", self.body_of(opener.call_args_list[0]))
        self.assertNotIn("temperature", self.body_of(opener.call_args_list[1]))
        _, later = self.send([response()])
        self.assertNotIn("temperature", self.body_of(later.call_args_list[0]))

    def test_other_bad_requests_still_fail_fast(self):
        with self.assertRaises(LLMError) as caught:
            self.send([self.refusal(b'{"error": {"message": "unknown model"}}')])
        self.assertIn("HTTP 400", str(caught.exception))
        self.assertNotIn("unknown model", str(caught.exception))

    def test_a_second_refusal_is_not_retried_again(self):
        with self.assertRaises(LLMError):
            self.send([self.refusal(b"temperature"), self.refusal(b"temperature")])


class RetryVisibilityTests(unittest.TestCase):
    """A retry must be visible in verbose mode, or a rate-limited run looks frozen."""

    def test_retry_is_logged_with_attempt_and_delay(self):
        client = LLMClient(default_settings().llm, ENVIRON, sleep=lambda _: None, random_fraction=lambda: 0.5)
        with patch("auditor.llm_client._opener.open", side_effect=[http_error(429), response()]):
            with self.assertLogs("auditor.llm_client", level="INFO") as logs:
                client.chat([{"role": "user", "content": "hi"}], [])
        text = "\n".join(logs.output)
        self.assertIn("HTTP 429", text)
        self.assertIn("Retry 1 of", text)
