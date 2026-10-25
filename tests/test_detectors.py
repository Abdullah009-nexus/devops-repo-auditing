import unittest

from auditor.checks import audit_inspected_files, supports_finding
from auditor.rules import get_rule
from auditor.redaction import redact_secrets


def run(rule_id, files=None, paths=None):
    result = audit_inspected_files(files or {}, paths=paths, rule_ids=[rule_id])
    return [f for f in result["findings"] if f["rule_id"] == rule_id]


def lines_of(findings):
    return [f["line"] for f in findings]


WORKFLOW = ".github/workflows/ci.yml"
AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"
PASSWORD = "hunter2" + "hunter2"


class CredentialTests(unittest.TestCase):
    def test_quoted_literal_in_code_is_flagged_as_critical_with_redacted_evidence(self):
        found = run("SEC-001", {"app.py": f'DB_PASSWORD = "{PASSWORD}"\n'})
        self.assertEqual(lines_of(found), [1])
        self.assertEqual(found[0]["severity"], "critical")
        self.assertNotIn(PASSWORD, repr(found))

    def test_code_expressions_and_masks_are_not_flagged(self):
        source = "\n".join(
            [
                "password = settings.DB_PASSWORD",
                "password = os.environ['DB_PASSWORD']",
                "password: str = None",
                'password = "********"',
                'password = "{{ vault_password }}"',
                'password = "process.env.DB_PASSWORD"',
                "self.password = password",
            ]
        )
        self.assertEqual(run("SEC-001", {"app.py": source}), [])

    def test_unquoted_value_counts_in_env_style_files(self):
        self.assertEqual(lines_of(run("SEC-001", {"prod.env": f"DB_PASSWORD={PASSWORD}\n"})), [1])
        self.assertEqual(run("SEC-001", {"prod.env": "DB_PASSWORD=${DB_PASSWORD}\n"}), [])

    def test_name_and_value_on_separate_lines(self):
        manifest = f"env:\n  - name: DB_PASSWORD\n    value: {PASSWORD}\n"
        found = run("SEC-001", {"deploy.yaml": manifest})
        self.assertEqual(lines_of(found), [3])
        self.assertNotIn(PASSWORD, found[0]["evidence"])
        self.assertIn("DB_PASSWORD", found[0]["evidence"])
        self.assertEqual(run("SEC-001", {"deploy.yaml": "- name: DB_PASSWORD\n  value: ${DB_PASSWORD}\n"}), [])
        self.assertEqual(run("SEC-001", {"deploy.yaml": f"- name: LOG_LEVEL\n  value: {PASSWORD}\n"}), [])

    def test_block_scalar_secret(self):
        found = run("SEC-001", {"values.yml": f"password: |\n  {PASSWORD}\nother: 1\n"})
        self.assertEqual(lines_of(found), [2])
        self.assertNotIn(PASSWORD, repr(found))

    def test_redaction_hides_the_multiline_shapes(self):
        text = f"- name: API_TOKEN\n  value: {PASSWORD}\nsecret: >\n  {PASSWORD}\n"
        self.assertNotIn(PASSWORD, redact_secrets(text))
        self.assertEqual(redact_secrets("- name: API_TOKEN\n  value: ${{ secrets.T }}\n").count("secrets.T"), 1)

    def test_strict_evidence_contract(self):
        files = {"app.py": f'API_TOKEN = "{PASSWORD}"\n'}
        finding = run("SEC-001", files)[0]
        self.assertTrue(supports_finding("SEC-001", "app.py", 1, finding["evidence"], files))
        self.assertFalse(supports_finding("SEC-001", "app.py", 1, "other", files))


class TokenAndFileTests(unittest.TestCase):
    def test_known_token_format_is_flagged_and_redacted(self):
        found = run("SEC-002", {"notes.txt": f"key {AWS_KEY}\n"})
        self.assertEqual(lines_of(found), [1])
        self.assertNotIn(AWS_KEY, repr(found))

    def test_documented_fake_and_public_key_are_not_flagged(self):
        files = {"a.txt": f"{AWS_KEY} example only\n", "b.pem": "-----BEGIN PUBLIC KEY-----\nabc\n"}
        self.assertEqual(run("SEC-002", files), [])

    def test_private_key_block_is_flagged(self):
        begin = "-----BEGIN RSA " + "PRIVATE KEY-----"
        found = run("SEC-002", {"k.txt": f"{begin}\nabcdef\n"})
        self.assertEqual(lines_of(found), [1])
        self.assertNotIn("abcdef", repr(found))

    def test_secret_files_by_path_only(self):
        found = run("SEC-003", paths=[".env", "config/.env.production", "keys/id_rsa", "app.py"])
        self.assertEqual(sorted(f["file"] for f in found), [".env", "config/.env.production", "keys/id_rsa"])
        self.assertTrue(all(f["line"] is None and f["evidence"] == f["file"] for f in found))

    def test_example_env_and_fixtures_are_not_flagged(self):
        self.assertEqual(run("SEC-003", paths=[".env.example", ".env.sample", "tests/fixtures/x/.env"]), [])

    def test_npmrc_needs_a_literal_token_and_pem_needs_a_private_key(self):
        token = "npm" + "tokenvalue123"
        self.assertEqual(len(run("SEC-003", {".npmrc": f"//r/:_authToken={token}\n"})), 1)
        self.assertEqual(run("SEC-003", {".npmrc": "//r/:_authToken=${NPM_TOKEN}\n"}), [])
        self.assertEqual(run("SEC-003", paths=[".npmrc"]), [])
        self.assertEqual(run("SEC-003", {"ca.pem": "-----BEGIN CERTIFICATE-----\n"}), [])


class WorkflowTests(unittest.TestCase):
    def test_write_all_and_scoped_permissions(self):
        self.assertEqual(lines_of(run("CI-001", {WORKFLOW: "permissions: write-all\n"})), [1])
        self.assertEqual(run("CI-001", {WORKFLOW: "permissions:\n  contents: read\n"}), [])

    def test_pull_request_target_checkout(self):
        bad = (
            "on:\n  pull_request_target:\njobs:\n  t:\n    steps:\n"
            "      - uses: actions/checkout@v4\n        with:\n"
            "          ref: ${{ github.event.pull_request.head.sha }}\n      - run: pytest\n"
        )
        found = run("CI-002", {WORKFLOW: bad})
        self.assertEqual((lines_of(found), found[0]["severity"]), ([8], "critical"))
        self.assertEqual(run("CI-002", {WORKFLOW: bad.replace("_target", "")}), [])

    def test_gate_that_cannot_fail(self):
        steps = (
            "jobs:\n  t:\n    steps:\n"
            "      - run: pytest || true\n"
            "      - name: lint\n        continue-on-error: true\n        run: ruff check .\n"
            "      - run: |\n          flake8 --exit-zero\n"
        )
        self.assertEqual(lines_of(run("CI-004", {WORKFLOW: steps})), [4, 6, 9])

    def test_optional_steps_are_not_gates(self):
        steps = (
            "jobs:\n  t:\n    steps:\n"
            "      - run: pytest\n"
            "      - name: pytest report\n        continue-on-error: true\n        run: echo done\n"
            "      - continue-on-error: ${{ matrix.experimental }}\n        run: pytest\n"
            "      - run: echo ok || true\n"
        )
        self.assertEqual(run("CI-004", {WORKFLOW: steps}), [])

    def test_script_injection(self):
        bad = "steps:\n  - run: echo ${{ github.event.pull_request.title }}\n"
        self.assertEqual(lines_of(run("CI-007", {WORKFLOW: bad})), [2])
        safe = (
            "steps:\n  - run: echo \"$TITLE\"\n    env:\n"
            "      TITLE: ${{ github.event.pull_request.title }}\n"
        )
        self.assertEqual(run("CI-007", {WORKFLOW: safe}), [])

    def test_actions_pinned_to_branches(self):
        uses = (
            "steps:\n  - uses: acme/tool@main\n  - uses: acme/other\n"
            "  - uses: acme/ok@v1\n  - uses: acme/sha@" + "a" * 40 + "\n"
            "  - uses: ./local\n  - uses: docker://alpine\n"
        )
        self.assertEqual(lines_of(run("CI-008", {WORKFLOW: uses})), [2, 3])

    def test_workflow_rules_ignore_other_files(self):
        self.assertEqual(run("CI-001", {"notes.yml": "permissions: write-all\n"}), [])


class PathAndConfigTests(unittest.TestCase):
    def test_generated_files_are_grouped_by_directory(self):
        paths = ["node_modules/a/b.js", "node_modules/c.js", "pkg/__pycache__/x.pyc", "run.log", "src/app.py"]
        found = run("HYG-004", paths=paths)
        self.assertEqual(sorted(f["evidence"] for f in found), ["node_modules/", "pkg/__pycache__/", "run.log"])

    def test_fixtures_and_sources_are_not_generated(self):
        self.assertEqual(run("HYG-004", paths=["tests/fixtures/x/a.pyc", "src/build_tools.py"]), [])

    def test_machine_specific_paths(self):
        windows = "C:" + "\\Users\\carol\\proj\\data.csv"
        files = {"conf.py": f'ROOT = r"{windows}"\nHOME = "/home/dmitri/app"\n'}
        self.assertEqual(lines_of(run("CFG-001", files)), [1, 2])

    def test_escaped_windows_paths_in_ordinary_strings(self):
        escaped = "C:" + "\\\\Users\\\\carol\\\\proj"
        files = {"conf.py": f'ROOT = "{escaped}"\n'}
        self.assertEqual(lines_of(run("CFG-001", files)), [1])

    def test_documented_and_container_paths_are_not_flagged(self):
        files = {
            "README.md": "see /home/alice/x\n",
            "docs/setup.txt": "/Users/alice/x\n",
            "run.py": 'A = "/home/runner/work"\nB = "/app/data"\nC = "/home/<user>/x"\n',
        }
        self.assertEqual(run("CFG-001", files), [])

    def test_dockerfile_floating_base_and_pipe_to_shell(self):
        text = (
            "FROM python:latest\nFROM node\nFROM alpine:3.19\nRUN curl -sSL https://x.sh | sh\n"
        )
        self.assertEqual(lines_of(run("BLD-003", {"Dockerfile": text})), [1, 2, 4])

    def test_pinned_and_dynamic_bases_are_not_flagged(self):
        text = (
            "ARG BASE=python:3.12\nFROM ${BASE}\nFROM scratch\nFROM python:3.12 AS build\n"
            "FROM build AS final\nFROM python@sha256:" + "0" * 64 + "\nRUN curl -o f https://x\n"
        )
        self.assertEqual(run("BLD-003", {"Dockerfile": text}), [])


class GateFixTests(unittest.TestCase):
    def test_injection_flags_only_attacker_controlled_fields(self):
        def injected(expression):
            return run("CI-007", {WORKFLOW: "steps:\n  - run: echo ${{ " + expression + " }}\n"})

        for expression in (
            "github.event.issue.body",
            "github.event.comment.body",
            "github.event.pull_request.head.ref",
            "github.event.head_commit.author.email",
            "github.event.commits[0].message",
            "github.head_ref",
        ):
            self.assertEqual(len(injected(expression)), 1, expression)
        for expression in (
            "github.event.pull_request.number",
            "github.event.pull_request.head.sha",
            "github.event.pull_request.user.login",
            "github.event.issue.id",
        ):
            self.assertEqual(injected(expression), [], expression)

    def test_pr_target_run_after_untrusted_checkout_is_code_execution(self):
        def workflow(ref_line, step="      - run: ./build.sh\n"):
            return (
                "on:\n  pull_request_target:\njobs:\n  t:\n    steps:\n"
                "      - uses: actions/checkout@v4\n        with:\n"
                f"          {ref_line}\n{step}"
            )

        for ref_line in (
            "ref: refs/pull/${{ github.event.pull_request.number }}/merge",
            "ref: ${{ github.head_ref }}",
            "repository: ${{ github.event.pull_request.head.repo.full_name }}",
        ):
            self.assertEqual(lines_of(run("CI-002", {WORKFLOW: workflow(ref_line)})), [8], ref_line)
        self.assertEqual(run("CI-002", {WORKFLOW: workflow("ref: main")}), [])
        no_run = workflow("ref: ${{ github.head_ref }}", step="      - uses: acme/lint@v1\n")
        self.assertEqual(run("CI-002", {WORKFLOW: no_run}), [])

    def test_short_quoted_literal_in_code_counts_but_not_in_config(self):
        self.assertEqual(lines_of(run("SEC-001", {"app.py": 'password = "admin"\n'})), [1])
        self.assertEqual(lines_of(run("SEC-001", {"Main.kt": 'val password = "123456"\n'})), [1])
        self.assertEqual(run("SEC-001", {"app.py": 'password = "abcd"\n'}), [])
        self.assertEqual(run("SEC-001", {"app.yml": "password: admin\n"}), [])

    def test_service_container_defaults_and_json_prose_are_not_credentials(self):
        compose = (
            "services:\n  db:\n    environment:\n"
            "      POSTGRES_PASSWORD: postgres\n      MYSQL_ROOT_PASSWORD: rootpassword\n"
        )
        self.assertEqual(run("SEC-001", {"docker-compose.yml": compose}), [])
        locale = '{"password": "password", "forgot_password": "Forgot your password?"}\n'
        self.assertEqual(run("SEC-001", {"en.json": locale}), [])
        self.assertEqual(len(run("SEC-001", {"cfg.json": '{"password": "' + PASSWORD + '"}\n'})), 1)

    def test_build_and_dist_count_only_at_the_root_for_artefacts(self):
        found = run("HYG-004", paths=["dist/app.js", "dist/bundle.whl", "build/lib.so"])
        self.assertEqual(sorted(f["evidence"] for f in found), ["build/", "dist/"])
        kept = [
            "build/Dockerfile",
            "build/build_test.go",
            "dist/README.md",
            "src/dist/app.js",
            ".github/actions/build/action.yml",
        ]
        self.assertEqual(run("HYG-004", paths=kept), [])

    def test_env_templates_and_keys_need_evidence(self):
        safe = [".env.test", ".env.development", "example.env", ".env.defaults"]
        self.assertEqual(run("SEC-003", paths=safe), [])
        self.assertEqual(len(run("SEC-003", paths=["prod.env"])), 1)
        self.assertEqual(run("SEC-003", paths=["server.key"]), [])
        private = "-----BEGIN RSA " + "PRIVATE KEY-----\nabc\n"
        self.assertEqual(len(run("SEC-003", {"server.key": private})), 1)
        self.assertEqual(run("SEC-003", {"server.key": "not a key\n"}), [])
        self.assertEqual(run("SEC-003", {".env.production": "VITE_API_URL=https://api.dev\n"}), [])
        self.assertEqual(len(run("SEC-003", {".pypirc": "password = pypi-" + PASSWORD + "\n"})), 1)
        self.assertEqual(run("SEC-003", paths=[".pypirc"]), [])

    def test_comments_tests_and_sandbox_users_skip_absolute_paths(self):
        self.assertEqual(run("CFG-001", {"conf.py": "# see /home/dmitri/app\n"}), [])
        self.assertEqual(run("CFG-001", {"tests/test_conf.py": 'P = "/home/dmitri/app"\n'}), [])
        self.assertEqual(run("CFG-001", {"lib/conf_test.go": 'P = "/home/dmitri/app"\n'}), [])
        self.assertEqual(run("CFG-001", {"dev.json": '"/home/vscode/app"\n'}), [])
        self.assertEqual(lines_of(run("CFG-001", {"conf.py": 'P = "/home/dmitri/app"\n'})), [1])

    def test_ungated_matches_commands_at_segment_start_only(self):
        def gates(step):
            return lines_of(run("CI-004", {WORKFLOW: "jobs:\n  t:\n    steps:\n" + step}))

        self.assertEqual(gates("      - run: cd app && pytest || :\n"), [4])
        self.assertEqual(gates("      - run: python -m pytest || true\n"), [4])
        self.assertEqual(gates("      - run: pip install pytest || true\n"), [])
        self.assertEqual(gates("      - run: npm install eslint || true\n"), [])
        self.assertEqual(gates("      - run: sudo apt-get install -y ruff || true\n"), [])
        self.assertEqual(gates("      - run: echo run pytest next || true\n"), [])

    def test_job_level_continue_on_error(self):
        def job(job_id, name=""):
            label = f"    name: {name}\n" if name else ""
            return (
                f"jobs:\n  {job_id}:\n{label}    continue-on-error: true\n"
                "    steps:\n      - run: pytest\n"
            )

        self.assertEqual(lines_of(run("CI-004", {WORKFLOW: job("build")})), [3])
        self.assertEqual(run("CI-004", {WORKFLOW: job("experimental")}), [])
        self.assertEqual(run("CI-004", {WORKFLOW: job("t", "Nightly build")}), [])
        matrix = job("t") + "      - if: matrix.experimental\n        run: echo\n"
        self.assertEqual(run("CI-004", {WORKFLOW: matrix}), [])


class ResultShapeTests(unittest.TestCase):
    def test_findings_use_catalog_fields_and_rule_filter(self):
        files = {WORKFLOW: "permissions: write-all\nsteps:\n  - uses: acme/tool@main\n"}
        everything = audit_inspected_files(files)
        only = audit_inspected_files(files, rule_ids=["CI-008"])
        self.assertEqual({f["rule_id"] for f in everything["findings"]}, {"CI-001", "CI-008"})
        self.assertEqual({f["rule_id"] for f in only["findings"]}, {"CI-008"})
        finding = only["findings"][0]
        rule = get_rule("CI-008")
        self.assertEqual(
            (finding["category"], finding["severity"], finding["description"], finding["recommendation"]),
            (rule.category, rule.default_severity, rule.title, rule.recommendation),
        )


if __name__ == "__main__":
    unittest.main()
