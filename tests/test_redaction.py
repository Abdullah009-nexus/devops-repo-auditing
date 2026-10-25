import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import agent
from auditor import redaction
from auditor.checks import audit_inspected_files, supports_finding
from auditor.redaction import (
    RULES_PATH,
    find_secret_assignments,
    load_secret_rules,
    name_components,
    packaged_secret_rules,
    redact_secrets,
    redacted_lines,
)
from auditor.repository import GitHubApiSource, RepositoryMetadata
from auditor.settings import ConfigError
from tests.support import default_settings
from tools import GitHubRepository, RepositoryAddress


# Built from obviously fake parts so nothing credential-shaped is committed.
FAKE_VALUE = "fake" + "x" * 12
OPENAI_STYLE = "sk-" + "x" * 24
GOOGLE_STYLE = "AIza" + "x" * 35
GROQ_STYLE = "gsk_" + "x" * 24
NPM_STYLE = "npm_" + "x" * 36
PYPI_STYLE = "pypi-" + "x" * 56
GITHUB_STYLE = "ghp_" + "x" * 36
AWS_STYLE = "AKIA" + "X" * 16
AWS_SESSION_STYLE = "ASIA" + "X" * 16
GITLAB_STYLE = "glpat-" + "x" * 20
HUGGING_FACE_STYLE = "hf_" + "x" * 34
STRIPE_STYLES = ("sk_live_" + "x" * 24, "rk_live_" + "x" * 24, "sk_test_" + "x" * 24)
SLACK_WEBHOOK = "https://hooks.slack.com/services/" + "/".join(("T" + "x" * 8, "B" + "x" * 8, "x" * 24))
JWT_STYLE = ".".join(("eyJ" + "x" * 16, "eyJ" + "x" * 24, "x" * 32))
KEY_MATERIAL = "fake" + "Q" * 60
PEM_BLOCK = "\n".join((
    "-----BEGIN RSA PRIVATE KEY-----",
    KEY_MATERIAL,
    KEY_MATERIAL,
    "-----END RSA PRIVATE KEY-----",
))
# Large enough to defeat any per-line shortcut; an earlier matcher took seconds here.
PATHOLOGICAL_LINE_CHARS = 256 * 1024
# A quadratic pattern takes minutes on these inputs; the generous limit only
# keeps slow CI machines from failing a linear one.
TIME_LIMIT_SECONDS = 5.0


class RedactionTestCase(unittest.TestCase):
    def assertHidden(self, secret, text):
        # A failure message must not echo the value it was guarding.
        self.assertFalse(secret in text, "a secret value survived redaction")


class SensitiveKeyTests(unittest.TestCase):
    def setUp(self):
        self.rules = packaged_secret_rules()

    def test_splits_snake_kebab_dotted_and_camel_case_names(self):
        self.assertEqual(name_components("OPENAI_API_KEY"), ("openai", "api", "key"))
        self.assertEqual(name_components("dbPassword"), ("db", "password"))
        self.assertEqual(name_components("APIKey"), ("api", "key"))
        self.assertEqual(name_components("db.client-secret"), ("db", "client", "secret"))

    def test_trailing_components_decide_sensitivity(self):
        for key in ("OPENAI_API_KEY", "DB_PASSWORD", "GITHUB_TOKEN", "dbPassword", "password",
                    "AWS_SECRET_ACCESS_KEY", "self.api_token", "apikey", "DB_PASS",
                    "PGPASSWORD", "SECRET_KEY_BASE"):
            with self.subTest(key=key):
                self.assertTrue(self.rules.is_sensitive_key(key))
        for key in ("max_tokens", "TOKEN_URL", "password_min_length", "tokenizer", "keyboard",
                    "secrets", "bypass", "passwordless"):
            with self.subTest(key=key):
                self.assertFalse(self.rules.is_sensitive_key(key))

    def test_auth_is_sensitive_only_as_the_whole_key(self):
        self.assertTrue(self.rules.is_sensitive_key("auth"))
        for key in ("oauth", "use_auth", "auth_url", "OAuth"):
            with self.subTest(key=key):
                self.assertFalse(self.rules.is_sensitive_key(key))


class AssignmentRedactionTests(RedactionTestCase):
    def test_redacts_prefixed_camel_case_and_quoted_keys(self):
        lines = (
            f"OPENAI_API_KEY={FAKE_VALUE}",
            f"DB_PASSWORD={FAKE_VALUE}",
            f"GITHUB_TOKEN={FAKE_VALUE}",
            f'dbPassword = "{FAKE_VALUE}"',
            f'{{"password": "{FAKE_VALUE}"}}',
            f"  password: {FAKE_VALUE}",
            f"export API_TOKEN={FAKE_VALUE}",
            f"client_secret = '{FAKE_VALUE}'",
            f"PGPASSWORD={FAKE_VALUE}",
            f"DB_PASS={FAKE_VALUE}",
            f"SECRET_KEY_BASE={FAKE_VALUE}",
            f'{{"auths": {{"registry.example.com": {{"auth": "{FAKE_VALUE}"}}}}}}',
        )
        for index, line in enumerate(lines):
            with self.subTest(index=index):
                redacted = redact_secrets(line)
                self.assertHidden(FAKE_VALUE, redacted)
                self.assertIn("[REDACTED]", redacted)

    def test_keeps_the_key_and_surrounding_text(self):
        self.assertEqual(
            redact_secrets(f'DB_PASSWORD = "{FAKE_VALUE}"  # local'),
            "DB_PASSWORD = [REDACTED]  # local",
        )

    def test_leaves_non_secret_names_untouched(self):
        lines = (
            "max_tokens = 4096",
            'TOKEN_URL = "https://auth.example.com/token"',
            "password_min_length = 8",
            "tokenizer = x",
            "oauth = enabled",
            "use_auth: true",
        )
        for line in lines:
            with self.subTest(line=line):
                self.assertEqual(redact_secrets(line), line)

    def test_comparisons_and_other_operators_are_not_assignments(self):
        lines = (
            'if password == "x":',
            'if token != "x":',
            "while password <= limit:",
            "if (api_key := lookup()):",
            "password => value",
            "Token::parse(text)",
        )
        for line in lines:
            with self.subTest(line=line):
                self.assertEqual(redact_secrets(line), line)

    def test_finds_a_secret_inside_a_harmless_value(self):
        redacted = redact_secrets(f"callback=https://example.com/cb?token={FAKE_VALUE}")
        self.assertHidden(FAKE_VALUE, redacted)

    def test_does_not_join_a_bare_key_with_the_next_line(self):
        self.assertEqual(redact_secrets("password:\n  nested: value"), "password:\n  nested: value")

    def test_redaction_is_idempotent(self):
        once = redact_secrets(
            f"api_key = {GITHUB_STYLE}\nAuthorization: Bearer {FAKE_VALUE}\n"
            f"DATABASE_URL=postgres://app:{FAKE_VALUE}@db/app\n{PEM_BLOCK}"
        )
        self.assertEqual(redact_secrets(once), once)

    def test_placeholders_are_redacted_too(self):
        self.assertEqual(redact_secrets("API_KEY=changeme"), "API_KEY=[REDACTED]")

    def test_reports_value_span_for_sensitive_keys_only(self):
        line = f"user = admin; DB_PASSWORD = '{FAKE_VALUE}'"
        found = find_secret_assignments(line, packaged_secret_rules())
        self.assertEqual([item.key for item in found], ["DB_PASSWORD"])
        self.assertEqual(line[found[0].start:found[0].end], f"'{FAKE_VALUE}'")


class QuotedValueTests(RedactionTestCase):
    def test_triple_quoted_values_are_redacted_whole(self):
        for quote in ('"""', "'''"):
            with self.subTest(quote=quote):
                line = f"PASSWORD = {quote}{FAKE_VALUE} tail{quote}"
                self.assertEqual(redact_secrets(line), "PASSWORD = [REDACTED]")

    def test_escaped_and_doubled_quotes_stay_inside_the_value(self):
        lines = (
            f"token = 'ab\\'{FAKE_VALUE} more'",
            f'token = "ab\\"{FAKE_VALUE} more"',
            f"token = 'it''s {FAKE_VALUE} more'",
        )
        for index, line in enumerate(lines):
            with self.subTest(index=index):
                self.assertEqual(redact_secrets(line), "token = [REDACTED]")

    def test_escaped_quote_directly_before_the_value_does_not_end_it(self):
        backslash = chr(92)
        for quote in ("'", '"'):
            with self.subTest(quote=quote):
                line = f"token = {quote}ab{backslash}{quote}{FAKE_VALUE}{quote}"
                self.assertEqual(redact_secrets(line), "token = [REDACTED]")

    def test_text_glued_to_a_closing_quote_is_part_of_the_value(self):
        for quote in ("'", '"'):
            with self.subTest(quote=quote):
                line = f"token = {quote}ab{quote}{FAKE_VALUE}{quote}"
                self.assertEqual(redact_secrets(line), "token = [REDACTED]")

    def test_closing_brackets_and_separators_after_a_quote_are_kept(self):
        cases = (
            (f'{{"password": "{FAKE_VALUE}"}}', '{"password": [REDACTED]}'),
            (f"call(password='{FAKE_VALUE}')", "call(password=[REDACTED])"),
            (f"['password', TOKEN='{FAKE_VALUE}']", "['password', TOKEN=[REDACTED]]"),
            (f'password: "{FAKE_VALUE}", user: "x"', 'password: [REDACTED], user: "x"'),
        )
        for index, (line, expected) in enumerate(cases):
            with self.subTest(index=index):
                self.assertEqual(redact_secrets(line), expected)

    def test_an_unterminated_quote_is_redacted_to_the_end_of_the_line(self):
        redacted = redact_secrets(f'password = "abc {FAKE_VALUE}')
        self.assertEqual(redacted, "password = [REDACTED]")


class ReferenceValueTests(unittest.TestCase):
    def test_pure_references_stay_visible(self):
        lines = (
            "password: ${{ secrets.DB_PASSWORD }}",
            'password: "${{ secrets.DB_PASSWORD }}"',
            "DB_PASSWORD=${DB_PASSWORD}",
            "DB_PASSWORD=$DB_PASSWORD",
            '{"token": "${API_TOKEN}"}',
            "<password>${DB_PASSWORD}</password>",
        )
        for line in lines:
            with self.subTest(line=line):
                self.assertEqual(redact_secrets(line), line)

    def test_references_with_inline_text_are_redacted(self):
        lines = (
            f"DB_PASSWORD=${{DB_PASSWORD:-{FAKE_VALUE}}}",
            f"password: ${{{{ secrets.DB }}}}{FAKE_VALUE}",
            f"password: {FAKE_VALUE}${{{{ secrets.DB }}}}",
        )
        for index, line in enumerate(lines):
            with self.subTest(index=index):
                self.assertFalse(FAKE_VALUE in redact_secrets(line), "inline default leaked")


class ContextPatternTests(RedactionTestCase):
    def test_url_userinfo_hides_only_the_password(self):
        cases = (
            (f"postgres://app:{FAKE_VALUE}@db:5432/app", "postgres://app:[REDACTED]@db:5432/app"),
            (f"mongodb+srv://app:{FAKE_VALUE}@cluster0.example.net/db",
             "mongodb+srv://app:[REDACTED]@cluster0.example.net/db"),
            (f"redis://:{FAKE_VALUE}@cache:6379/0", "redis://:[REDACTED]@cache:6379/0"),
            (f"https://oauth2:{FAKE_VALUE}@gitlab.example.com/group/repo.git",
             "https://oauth2:[REDACTED]@gitlab.example.com/group/repo.git"),
        )
        for index, (line, expected) in enumerate(cases):
            with self.subTest(index=index):
                self.assertEqual(redact_secrets(line), expected)

    def test_urls_without_a_password_are_untouched(self):
        for line in (
            "https://example.com:8443/path?q=1",
            "ssh://git@github.com/example/project.git",
            "git@github.com:example/project.git",
            "http://localhost:8080/health",
        ):
            with self.subTest(line=line):
                self.assertEqual(redact_secrets(line), line)

    def test_authorization_basic_and_token_schemes(self):
        self.assertEqual(
            redact_secrets(f"Authorization: Basic {FAKE_VALUE}=="),
            "Authorization: Basic [REDACTED]",
        )
        self.assertEqual(
            redact_secrets(f'-H "Authorization: token {FAKE_VALUE}"'),
            '-H "Authorization: token [REDACTED]"',
        )

    def test_token_scheme_is_limited_to_authorization(self):
        for line in ("Use a token for automation", "X-Request: token reused", "Basic setup steps"):
            with self.subTest(line=line):
                self.assertEqual(redact_secrets(line), line)

    def test_arrow_and_subscript_assignments(self):
        cases = (
            (f"'password' => '{FAKE_VALUE}',", "'password' => [REDACTED],"),
            (f":password => '{FAKE_VALUE}'", ":password => [REDACTED]"),
            (f'config["password"] = "{FAKE_VALUE}"', 'config["password"] = [REDACTED]'),
            (f"ENV['API_TOKEN'] = '{FAKE_VALUE}'", "ENV['API_TOKEN'] = [REDACTED]"),
        )
        for index, (line, expected) in enumerate(cases):
            with self.subTest(index=index):
                self.assertEqual(redact_secrets(line), expected)

    def test_arrow_and_subscript_ignore_harmless_keys_and_comparisons(self):
        for line in ("'host' => 'db.local',", ":timeout => 30", 'config["password"] == other',
                     'config["user"] = "admin"'):
            with self.subTest(line=line):
                self.assertEqual(redact_secrets(line), line)

    def test_long_cli_flags_named_like_secrets(self):
        cases = (
            (f"app --password {FAKE_VALUE} --verbose", "app --password [REDACTED] --verbose"),
            (f"app --password={FAKE_VALUE}", "app --password=[REDACTED]"),
            (f"app --api-key {FAKE_VALUE}", "app --api-key [REDACTED]"),
            (f"curl -s -u admin:{FAKE_VALUE} https://example.com",
             "curl -s -u admin:[REDACTED] https://example.com"),
        )
        for index, (line, expected) in enumerate(cases):
            with self.subTest(index=index):
                self.assertEqual(redact_secrets(line), expected)

    def test_other_flags_and_short_options_are_untouched(self):
        for line in ("app --max-tokens 4096", "app --password-file /run/secrets/db",
                     "mysql -p -u root", "docker run -u 1000:1000 image"):
            with self.subTest(line=line):
                self.assertEqual(redact_secrets(line), line)

    def test_netrc_and_xml_elements(self):
        self.assertEqual(
            redact_secrets(f"machine api.example.com login bot password {FAKE_VALUE}"),
            "machine api.example.com login bot password [REDACTED]",
        )
        self.assertEqual(redact_secrets(f"  password {FAKE_VALUE}"), "  password [REDACTED]")
        self.assertEqual(
            redact_secrets(f"<password>{FAKE_VALUE}</password>"), "<password>[REDACTED]</password>"
        )
        self.assertEqual(
            redact_secrets(f"<db.password>{FAKE_VALUE}</db.password>"),
            "<db.password>[REDACTED]</db.password>",
        )

    def test_prose_and_harmless_elements_are_untouched(self):
        for line in ("Set the password in the env file.", "<token_url>https://x</token_url>",
                     "<passwordPolicy>strict</passwordPolicy>"):
            with self.subTest(line=line):
                self.assertEqual(redact_secrets(line), line)


class TokenRedactionTests(RedactionTestCase):
    def test_redacts_well_known_token_formats_anywhere(self):
        tokens = (OPENAI_STYLE, GOOGLE_STYLE, GROQ_STYLE, NPM_STYLE, PYPI_STYLE, GITHUB_STYLE,
                  AWS_STYLE, AWS_SESSION_STYLE, GITLAB_STYLE, HUGGING_FACE_STYLE, SLACK_WEBHOOK,
                  JWT_STYLE) + STRIPE_STYLES
        for index, token in enumerate(tokens):
            with self.subTest(index=index):
                redacted = redact_secrets(f"see {token} in the logs")
                self.assertHidden(token, redacted)
                self.assertIn("[REDACTED TOKEN]", redacted)

    def test_token_lookalikes_are_untouched(self):
        for line in ("the task-runner flag", "AKIA" + "X" * 17, "eyJhbGciOiJIUzI1NiJ9 alone",
                     "hf_hub_download(repo)", "https://hooks.example.com/services/x"):
            with self.subTest(index=line[:4]):
                self.assertEqual(redact_secrets(line), line)

    def test_redacts_bearer_values_and_keeps_the_prefix(self):
        self.assertEqual(
            redact_secrets(f"Authorization: Bearer {FAKE_VALUE}"),
            "Authorization: Bearer [REDACTED]",
        )


class PrivateKeyTests(RedactionTestCase):
    def test_each_line_of_a_block_is_masked_and_line_count_is_kept(self):
        content = f"before\n{PEM_BLOCK}\nafter"
        lines = redacted_lines(content)
        self.assertEqual(len(lines), len(content.splitlines()))
        self.assertEqual(lines[0], "before")
        self.assertEqual(lines[1:5], ["[REDACTED PRIVATE KEY]"] * 4)
        self.assertEqual(lines[5], "after")

    def test_redact_secrets_keeps_crlf_line_endings(self):
        content = f"before\r\n{PEM_BLOCK.replace(chr(10), chr(13) + chr(10))}\r\nafter\r\n"
        redacted = redact_secrets(content)
        self.assertHidden(KEY_MATERIAL, redacted)
        self.assertEqual(redacted.count("\r\n"), content.count("\r\n"))

    def test_a_block_without_end_is_masked_to_the_end_of_the_text(self):
        content = f"x = 1\n-----BEGIN PRIVATE KEY-----\n{KEY_MATERIAL}\n{KEY_MATERIAL}"
        lines = redacted_lines(content)
        self.assertEqual(lines[0], "x = 1")
        self.assertEqual(lines[1:], ["[REDACTED PRIVATE KEY]"] * 3)

    def test_an_escaped_single_line_key_is_masked(self):
        line = f'"private_pem": "-----BEGIN PRIVATE KEY-----\\n{KEY_MATERIAL}\\n-----END PRIVATE KEY-----"'
        self.assertHidden(KEY_MATERIAL, redact_secrets(line))


class ToolRedactionTests(RedactionTestCase):
    def setUp(self):
        settings = default_settings()
        address = RepositoryAddress.parse("https://github.com/example/project", default_settings().github)
        tree = {"tree": [{"type": "blob", "path": "deploy/key.pem.txt", "size": 400}]}
        with patch("auditor.repository._request_json", return_value=tree):
            source = GitHubApiSource(address, RepositoryMetadata("main", 1), settings)
        self.repository = GitHubRepository(address, settings, source=source)
        self.content = f"# deploy key\n{PEM_BLOCK}\nend = true\n"

    @patch("auditor.repository._request_bytes")
    def test_read_file_masks_a_multi_line_private_key(self, request_bytes):
        request_bytes.return_value = self.content.encode("utf-8")
        result = self.repository.read_file("deploy/key.pem.txt")
        self.assertHidden(KEY_MATERIAL, result["content"])
        rendered = result["content"].splitlines()
        self.assertEqual(result["line_count"], 6)
        self.assertEqual(rendered[2], "3: [REDACTED PRIVATE KEY]")
        self.assertEqual(rendered[5], "6: end = true")

    @patch("auditor.repository._request_bytes")
    def test_search_matches_redacted_text_so_key_material_cannot_be_probed(self, request_bytes):
        request_bytes.return_value = self.content.encode("utf-8")
        self.assertEqual(self.repository.search_code(KEY_MATERIAL[:12])["matches"], [])
        result = self.repository.search_code("REDACTED PRIVATE KEY")
        self.assertEqual([match["line"] for match in result["matches"]], [2, 3, 4, 5])
        for match in result["matches"]:
            self.assertEqual(match["text"], "[REDACTED PRIVATE KEY]")


class CredentialCheckTests(unittest.TestCase):
    def _flagged_lines(self, content):
        findings = audit_inspected_files({"settings.py": content})["findings"]
        self.assertTrue(all(item["rule_id"] == "SEC-001" for item in findings))
        return [item["line"] for item in findings]

    def test_flags_prefixed_names_the_old_pattern_missed(self):
        content = f'DB_PASSWORD = "{"hunter2" * 2}"\nOPENAI_API_KEY = "{OPENAI_STYLE}"\n'
        self.assertEqual(self._flagged_lines(content), [1, 2])

    def test_evidence_is_redacted(self):
        content = f'OPENAI_API_KEY = "{OPENAI_STYLE}"\n'
        finding = audit_inspected_files({"settings.py": content})["findings"][0]
        self.assertFalse(OPENAI_STYLE in finding["evidence"], "secret value in evidence")
        self.assertEqual(finding["evidence"], "OPENAI_API_KEY = [REDACTED]")

    def test_ignores_placeholders_references_and_short_values(self):
        content = "\n".join((
            'DB_PASSWORD = "changeme"',
            'API_KEY = "your-api-key-here"',
            'GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")',
            "client_secret: ${CLIENT_SECRET}",
            "password: ${{ secrets.DB_PASSWORD }}",
            'password = "tiny"',
            "max_tokens = 409600000",
        ))
        self.assertEqual(self._flagged_lines(content), [])

    def test_a_placeholder_does_not_hide_a_real_value_on_the_same_line(self):
        content = f'user_password = "changeme"; ADMIN_PASSWORD = "{FAKE_VALUE}"'
        self.assertEqual(self._flagged_lines(content), [1])

    def test_evidence_inside_a_private_key_block_matches_whole_file_redaction(self):
        content = f"{PEM_BLOCK[:PEM_BLOCK.index(chr(10))]}\napi_key = \"{KEY_MATERIAL}\"\n-----END RSA PRIVATE KEY-----\n"
        files = {"settings.py": content}
        findings = audit_inspected_files(files, rule_ids=["SEC-001"])["findings"]
        self.assertEqual([item["line"] for item in findings], [2])
        self.assertEqual(findings[0]["evidence"], "[REDACTED PRIVATE KEY]")
        self.assertTrue(supports_finding("SEC-001", "settings.py", 2, findings[0]["evidence"], files))


class SettingValueTests(RedactionTestCase):
    def test_setting_values_of_sensitive_keys_stay_visible(self):
        for line in ("permissions:\n  id-token: write", "secret: false", "token: null",
                     "auth: true", 'password: "None"', "client_secret: ~", "API_TOKEN=no"):
            with self.subTest(line=line):
                self.assertEqual(redact_secrets(line), line)

    def test_values_that_only_start_like_a_setting_are_redacted(self):
        for index, value in enumerate((f"write{FAKE_VALUE}", f"true-{FAKE_VALUE}", f"'no {FAKE_VALUE}'")):
            with self.subTest(index=index):
                self.assertHidden(FAKE_VALUE, redact_secrets(f"token: {value}"))

    def test_setting_values_are_not_reported(self):
        files = {"ci.yml": "permissions:\n  id-token: write\nsecret: false\n"}
        self.assertEqual(audit_inspected_files(files)["findings"], [])


class UrlUserinfoPrecisionTests(RedactionTestCase):
    def test_digest_pins_and_ports_are_untouched(self):
        for line in ("uses: docker://alpine:3.18@sha256:" + "a" * 64,
                     "http://localhost:8080@x", "image: docker://repo/app:1.2.3@sha512:abc"):
            with self.subTest(line=line[:30]):
                self.assertEqual(redact_secrets(line), line)

    def test_real_passwords_are_still_redacted(self):
        cases = (
            (f"postgres://user:{FAKE_VALUE}@host/db", "postgres://user:[REDACTED]@host/db"),
            ("postgres://user:12ab34@host/db", "postgres://user:[REDACTED]@host/db"),
            (f"https://bot:{FAKE_VALUE}@sha.example.com/x", "https://bot:[REDACTED]@sha.example.com/x"),
        )
        for index, (line, expected) in enumerate(cases):
            with self.subTest(index=index):
                self.assertEqual(redact_secrets(line), expected)


class DollarValueTests(RedactionTestCase):
    def test_values_starting_with_a_dollar_are_redacted_unless_env_style(self):
        for line in ('password = "$ecretP4ss99"', "PASSWORD=$uperSecret123",
                     "DB_PASSWORD=$abcDEF123xyz"):
            with self.subTest(index=line[:3]):
                redacted = redact_secrets(line)
                self.assertNotEqual(redacted, line)
                self.assertTrue(redacted.endswith("[REDACTED]"))

    def test_env_style_and_braced_references_stay_visible(self):
        for line in ("DB_PASSWORD=$DB_PASSWORD", "password: $_PW_2", "token=${apiToken}"):
            with self.subTest(line=line):
                self.assertEqual(redact_secrets(line), line)


class ExpressionReferenceTests(RedactionTestCase):
    def test_expressions_holding_quoted_literals_are_redacted(self):
        lines = (
            "password: ${{ secrets.DB || 'fallbackSecret123' }}",
            "token: ${{ 'literalSecretValue1' }}",
            'token: "${{ format(\'{0}\', \'literalSecretValue2\') }}"',
        )
        for index, line in enumerate(lines):
            with self.subTest(index=index):
                redacted = redact_secrets(line)
                self.assertFalse("Secret" in redacted.split(":", 1)[1], "literal survived")
                self.assertIn("[REDACTED]", redacted)

    def test_plain_expressions_stay_whole(self):
        for line in ("password: ${{ secrets.DB }}", "token: ${{ github.token }}",
                     "api_key: ${{ secrets.API_KEY || secrets.FALLBACK }}"):
            with self.subTest(line=line):
                self.assertEqual(redact_secrets(line), line)


class PgpKeyTests(RedactionTestCase):
    def test_pgp_private_key_blocks_are_masked(self):
        block = f"-----BEGIN PGP PRIVATE KEY BLOCK-----\n\n{KEY_MATERIAL}\n-----END PGP PRIVATE KEY BLOCK-----"
        lines = redacted_lines(f"before\n{block}\nafter")
        self.assertHidden(KEY_MATERIAL, "\n".join(lines))
        self.assertEqual(lines[0], "before")
        self.assertEqual(lines[-1], "after")
        self.assertEqual(len(lines), 6)

    def test_pgp_public_key_blocks_are_untouched(self):
        block = f"-----BEGIN PGP PUBLIC KEY BLOCK-----\n{KEY_MATERIAL}\n-----END PGP PUBLIC KEY BLOCK-----"
        self.assertEqual(redact_secrets(block), block)


class PathologicalInputTests(unittest.TestCase):
    def assertFast(self, text):
        started = time.perf_counter()
        redact_secrets(text)
        self.assertLess(time.perf_counter() - started, TIME_LIMIT_SECONDS)

    def test_long_lines_of_harmless_assignments_redact_quickly(self):
        for unit in ("k=v&", "a=", "ab: cd, ", "--x ", "curl ", "x://:", "-eyJ", "<a>"):
            with self.subTest(unit=unit):
                self.assertFast(unit * (PATHOLOGICAL_LINE_CHARS // len(unit)))

    def test_repeated_private_key_headers_redact_quickly(self):
        self.assertFast("-----BEGIN PRIVATE KEY-----" * (PATHOLOGICAL_LINE_CHARS // 27))

    def test_an_over_long_line_fails_closed_after_a_sensitive_key(self):
        filler = "k=v&" * (PATHOLOGICAL_LINE_CHARS // 4)
        redacted = redact_secrets(f"{filler}password={FAKE_VALUE}&{filler}")
        self.assertFalse(FAKE_VALUE in redacted, "secret value survived on a long line")
        self.assertTrue(redacted.endswith("password=[REDACTED]"))

    def test_an_over_long_line_still_gets_token_redaction(self):
        filler = "k=v&" * (PATHOLOGICAL_LINE_CHARS // 4)
        redacted = redact_secrets(f"{filler} {GITHUB_STYLE} {filler}")
        self.assertFalse(GITHUB_STYLE in redacted, "token survived on a long line")


class RulesFileTests(unittest.TestCase):
    def setUp(self):
        self.original = RULES_PATH.read_text(encoding="utf-8")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "rules.toml"

    def _load(self, text):
        self.path.write_text(text, encoding="utf-8")
        return load_secret_rules(self.path)

    def test_packaged_rules_load(self):
        self.assertEqual(load_secret_rules().min_secret_length, 8)

    def test_missing_key_is_rejected(self):
        text = self.original.replace("min_secret_length = 8\n", "")
        with self.assertRaisesRegex(ConfigError, "secrets.min_secret_length"):
            self._load(text)

    def test_unknown_key_is_rejected(self):
        text = self.original.replace("[redaction]\n", "[redaction]\nextra = 1\n")
        with self.assertRaisesRegex(ConfigError, "unknown setting redaction.extra"):
            self._load(text)

    def test_missing_section_is_rejected(self):
        text = self.original.replace("[redaction]", "[redactions]")
        with self.assertRaisesRegex(ConfigError, r"missing section \[redaction\]"):
            self._load(text)

    def test_invalid_regex_is_rejected(self):
        text = self.original.replace("(?:AKIA|ASIA)[0-9A-Z]{16}", "(?:AKIA|ASIA[0-9A-Z]{16}")
        self.assertNotEqual(text, self.original)
        with self.assertRaisesRegex(ConfigError, r"token_patterns\[1\]"):
            self._load(text)

    def test_wrong_types_are_rejected(self):
        cases = (
            ("min_secret_length = 8", "min_secret_length = true"),
            ("max_line_chars = 4096", "max_line_chars = 0"),
            ('value_marker = "[REDACTED]"', "value_marker = 3"),
            ('"password", "passwd", "pwd",', '"password", "password", "pwd",'),
        )
        for old, new in cases:
            self.assertIn(old, self.original)
            with self.subTest(new=new), self.assertRaises(ConfigError):
                self._load(self.original.replace(old, new))

    def test_key_patterns_need_a_key_group(self):
        old = "'(?<![\\w:]):(?P<key>[A-Za-z_]\\w*)[ \\t]*=>[ \\t]*',"
        self.assertIn(old, self.original)
        text = self.original.replace(old, "':(\\w+)=>',")
        with self.assertRaisesRegex(ConfigError, r"key_patterns\[2\] needs named groups key"):
            self._load(text)

    def test_prefixed_patterns_need_exactly_one_group(self):
        old = "'(?i)((?<![A-Za-z0-9])Bearer[ \\t]+)[A-Za-z0-9._~+/=-]{8,}',"
        self.assertIn(old, self.original)
        text = self.original.replace(old, "'(?i)Bearer[ \\t]+\\S+',")
        with self.assertRaisesRegex(ConfigError, r"prefixed_patterns\[0\]"):
            self._load(text)

    def test_unreadable_toml_is_rejected(self):
        with self.assertRaises(ConfigError):
            self._load("[secrets\n")

    def test_cli_exits_2_when_the_rules_file_is_broken(self):
        self.path.write_text(self.original.replace("min_secret_length = 8\n", ""), encoding="utf-8")
        redaction.packaged_secret_rules.cache_clear()
        self.addCleanup(redaction.packaged_secret_rules.cache_clear)
        with (
            patch.object(redaction, "RULES_PATH", self.path),
            patch.object(agent, "run_agent") as run_agent,
            patch("agent.sys.stderr", new_callable=io.StringIO) as stderr,
            patch("urllib.request.urlopen", side_effect=AssertionError("network used")) as urlopen,
            tempfile.TemporaryDirectory() as directory,
            contextlib.chdir(directory),
        ):
            result = agent.main(["--repo", "https://github.com/example/project", "--output", "r.json"])
        urlopen.assert_not_called()
        self.assertEqual(result, 2)
        self.assertTrue(stderr.getvalue().startswith("Error: "))
        self.assertIn("secrets.min_secret_length", stderr.getvalue())
        run_agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
