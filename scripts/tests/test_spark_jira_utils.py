#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check JIRA versions and umbrella links of dev/spark_jira_utils.py against a fake JIRA."""

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[2] / "dev" / "spark_jira_utils.py"
SPEC = importlib.util.spec_from_file_location("spark_jira_utils", SCRIPT)
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


def version(name, released=False, archived=False):
    return SimpleNamespace(name=name, raw={"released": released, "archived": archived})


class FakeIssue:
    def __init__(self, key, summary="", fields=None, fail_update=False):
        self.key = key
        self.fields = SimpleNamespace(summary=summary)
        self.created_fields = fields
        self.updates = []
        self.fail_update = fail_update

    def update(self, fields):
        if self.fail_update:
            raise RuntimeError("no permission to set the Fix Version")
        self.updates.append(fields)


class FakeJira:
    def __init__(self, versions, epics=(), fail_update=False, fail_search=False, fail_link=False):
        self.versions = versions
        self.epics = list(epics)
        self.created = []
        self.links = []
        self.fail_update, self.fail_search, self.fail_link = fail_update, fail_search, fail_link

    def project_versions(self, project):
        assert project == "SPARK"
        return self.versions

    def search_issues(self, jql):
        # JIRA's `summary ~` text search is fuzzy, so it can return other Epics too.
        assert "issuetype = Epic" in jql
        if self.fail_search:
            raise RuntimeError("search failed")
        return self.epics

    def create_issue(self, fields):
        issue = FakeIssue(
            f"SPARK-{100 + len(self.created)}",
            fields["summary"],
            fields,
            fail_update=self.fail_update,
        )
        self.created.append(issue)
        return issue

    def create_issue_link(self, link_type, inwardIssue, outwardIssue):
        if self.fail_link:
            raise RuntimeError("link failed")
        self.links.append((link_type, inwardIssue, outwardIssue))


UMBRELLA = FakeIssue("SPARK-59102", "spark-connect-rust connect-rust-4.10.0 release")
VERSIONS = [
    version("5.0.0"),
    version("connect-rust-4.9.0"),
    version("connect-rust-4.10.0"),
    version("connect-rust-9.0.0", released=True),
    version("connect-rust-8.0.0", archived=True),
]


def create(jira, **kwargs):
    """Create an issue and return its key with what was printed to stderr."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        key = utils.create_jira_issue(jira, "Title", "Connect", **kwargs)
    return key, stderr.getvalue()


class VersionTests(unittest.TestCase):
    def test_unreleased_connect_rust_versions_newest_first(self):
        names = [v.name for v in utils.unreleased_connect_rust_versions(FakeJira(VERSIONS))]
        self.assertEqual(names, ["connect-rust-4.10.0", "connect-rust-4.9.0"])

    def test_newest_unreleased_connect_rust_version_compared_numerically(self):
        self.assertEqual(utils.detect_affected_version(FakeJira(VERSIONS)), "connect-rust-4.10.0")

    def test_core_spark_versions_are_not_candidates(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            utils.detect_affected_version(FakeJira([version("5.0.0"), version("4.1.2")]))

    def test_an_explicit_version_must_be_an_unreleased_connect_rust_version(self):
        for name in ["4.10.0", "connect-rust-9.0.0", "connect-rust-8.0.0", "connect-rust-1.0.0"]:
            jira = FakeJira(VERSIONS, epics=[UMBRELLA])
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                utils.create_jira_issue(jira, "Title", "Connect", version=name)
            self.assertEqual(jira.created, [], name)


class CreateJiraIssueTests(unittest.TestCase):
    def test_both_versions_and_the_release_umbrella_link(self):
        jira = FakeJira(VERSIONS, epics=[UMBRELLA])
        key, stderr = create(jira, issue_type="Bug")
        (issue,) = jira.created
        self.assertEqual(key, issue.key)
        self.assertEqual(issue.created_fields["versions"], [{"name": "connect-rust-4.10.0"}])
        self.assertEqual(issue.created_fields["issuetype"], {"name": "Bug"})
        self.assertEqual(issue.updates, [{"fixVersions": [{"name": "connect-rust-4.10.0"}]}])
        self.assertEqual(jira.links, [("Incorporates", "SPARK-59102", key)])
        self.assertEqual(stderr, "")

    def test_explicit_version_and_its_own_umbrella(self):
        other = FakeIssue("SPARK-59035", "spark-connect-rust connect-rust-4.9.0 release")
        jira = FakeJira(VERSIONS, epics=[UMBRELLA, other])
        key, _ = create(jira, version="connect-rust-4.9.0")
        (issue,) = jira.created
        self.assertEqual(issue.created_fields["versions"], [{"name": "connect-rust-4.9.0"}])
        self.assertEqual(issue.updates, [{"fixVersions": [{"name": "connect-rust-4.9.0"}]}])
        self.assertEqual(jira.links, [("Incorporates", "SPARK-59035", key)])

    def test_subtasks_are_not_linked_to_the_umbrella(self):
        jira = FakeJira(VERSIONS, epics=[UMBRELLA])
        create(jira, parent="SPARK-1")
        (issue,) = jira.created
        self.assertEqual(issue.created_fields["issuetype"], {"name": "Sub-task"})
        self.assertEqual(issue.updates, [{"fixVersions": [{"name": "connect-rust-4.10.0"}]}])
        self.assertEqual(jira.links, [])

    def test_only_a_single_umbrella_with_the_exact_summary_is_linked(self):
        near = FakeIssue("SPARK-2", "spark-connect-rust connect-rust-4.10.0 release notes")
        twin = FakeIssue("SPARK-3", UMBRELLA.fields.summary)
        for epics in ([], [near], [UMBRELLA, twin]):
            jira = FakeJira(VERSIONS, epics=epics)
            key, stderr = create(jira)
            self.assertEqual(jira.links, [], epics)
            self.assertIn("No single release umbrella Epic for connect-rust-4.10.0", stderr)
            self.assertIn(key, stderr)
        jira = FakeJira(VERSIONS, epics=[near, UMBRELLA])
        key, _ = create(jira)
        self.assertEqual(jira.links, [("Incorporates", "SPARK-59102", key)])

    def test_steps_after_the_issue_exists_warn_and_keep_the_key(self):
        jira = FakeJira(VERSIONS, epics=[UMBRELLA], fail_update=True)
        key, stderr = create(jira)
        self.assertIn("could not set its Fix Version", stderr)
        self.assertIn(key, stderr)
        self.assertEqual(jira.links, [("Incorporates", "SPARK-59102", key)])
        for failure in ("fail_search", "fail_link"):
            jira = FakeJira(VERSIONS, epics=[UMBRELLA], **{failure: True})
            key, stderr = create(jira)
            self.assertEqual(key, jira.created[0].key)
            self.assertIn("could not link it to the connect-rust-4.10.0 release umbrella", stderr)
            self.assertIn(key, stderr)


if __name__ == "__main__":
    unittest.main()
