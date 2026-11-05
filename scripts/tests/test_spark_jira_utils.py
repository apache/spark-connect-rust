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
    def __init__(self, key, summary="", fields=None):
        self.key = key
        self.fields = SimpleNamespace(summary=summary)
        self.created_fields = fields
        self.updates = []

    def update(self, fields):
        self.updates.append(fields)


class FakeJira:
    def __init__(self, versions, epics=()):
        self.versions = versions
        self.epics = list(epics)
        self.created = []
        self.links = []

    def project_versions(self, project):
        assert project == "SPARK"
        return self.versions

    def search_issues(self, jql):
        assert "issuetype = Epic" in jql
        return [epic for epic in self.epics if epic.fields.summary in jql]

    def create_issue(self, fields):
        issue = FakeIssue(f"SPARK-{100 + len(self.created)}", fields["summary"], fields)
        self.created.append(issue)
        return issue

    def issue_link_types(self):
        return [
            SimpleNamespace(name="Reference", outward="relates to", inward="relates to"),
            SimpleNamespace(name="Incorporates", outward="incorporates", inward="is part of"),
        ]

    def create_issue_link(self, link_type, inwardIssue, outwardIssue):
        self.links.append((link_type, inwardIssue, outwardIssue))


UMBRELLA = FakeIssue("SPARK-59102", "spark-connect-rust connect-rust-4.10.0 release")
VERSIONS = [
    version("5.0.0"),
    version("connect-rust-4.9.0"),
    version("connect-rust-4.10.0"),
    version("connect-rust-9.0.0", released=True),
    version("connect-rust-8.0.0", archived=True),
]


class DetectAffectedVersionTests(unittest.TestCase):
    def test_newest_unreleased_connect_rust_version_compared_numerically(self):
        self.assertEqual(utils.detect_affected_version(FakeJira(VERSIONS)), "connect-rust-4.10.0")

    def test_core_spark_versions_are_not_candidates(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            utils.detect_affected_version(FakeJira([version("5.0.0"), version("4.1.2")]))


class CreateJiraIssueTests(unittest.TestCase):
    def test_both_versions_and_the_release_umbrella_link(self):
        jira = FakeJira(VERSIONS, epics=[UMBRELLA])
        key = utils.create_jira_issue(jira, "Title", "Connect", issue_type="Bug")
        (issue,) = jira.created
        self.assertEqual(key, issue.key)
        self.assertEqual(issue.created_fields["versions"], [{"name": "connect-rust-4.10.0"}])
        self.assertEqual(issue.created_fields["issuetype"], {"name": "Bug"})
        self.assertEqual(issue.updates, [{"fixVersions": [{"name": "connect-rust-4.10.0"}]}])
        self.assertEqual(jira.links, [("Incorporates", "SPARK-59102", key)])

    def test_explicit_version_and_its_own_umbrella(self):
        other = FakeIssue("SPARK-59035", "spark-connect-rust connect-rust-4.9.0 release")
        jira = FakeJira(VERSIONS, epics=[UMBRELLA, other])
        key = utils.create_jira_issue(jira, "Title", "Connect", version="connect-rust-4.9.0")
        (issue,) = jira.created
        self.assertEqual(issue.created_fields["versions"], [{"name": "connect-rust-4.9.0"}])
        self.assertEqual(issue.updates, [{"fixVersions": [{"name": "connect-rust-4.9.0"}]}])
        self.assertEqual(jira.links, [("Incorporates", "SPARK-59035", key)])

    def test_subtasks_are_not_linked_to_the_umbrella(self):
        jira = FakeJira(VERSIONS, epics=[UMBRELLA])
        utils.create_jira_issue(jira, "Title", "Connect", parent="SPARK-1")
        (issue,) = jira.created
        self.assertEqual(issue.created_fields["issuetype"], {"name": "Sub-task"})
        self.assertEqual(issue.updates, [{"fixVersions": [{"name": "connect-rust-4.10.0"}]}])
        self.assertEqual(jira.links, [])

    def test_a_missing_umbrella_is_reported_but_not_fatal(self):
        jira = FakeJira(VERSIONS)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            key = utils.create_jira_issue(jira, "Title", "Connect")
        self.assertEqual(jira.links, [])
        self.assertIn(key, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
