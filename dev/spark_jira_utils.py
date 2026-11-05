#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import re
import sys

try:
    import jira.client

    JIRA_IMPORTED = True
except ImportError:
    JIRA_IMPORTED = False

# ASF JIRA access token
JIRA_ACCESS_TOKEN = os.environ.get("JIRA_ACCESS_TOKEN")
JIRA_API_BASE = "https://issues.apache.org/jira"


def fail(msg):
    print(msg)
    sys.exit(-1)


def get_jira_client():
    """Create and return a JIRA client, or exit with a helpful message."""
    errors = []
    if not JIRA_IMPORTED:
        errors.append("jira-python library not installed, run 'pip install jira'")
    if not JIRA_ACCESS_TOKEN:
        errors.append("JIRA_ACCESS_TOKEN env-var not set")
    if errors:
        fail(
            "Cannot create JIRA ticket automatically (%s). "
            "Please create the ticket manually at %s" % ("; ".join(errors), JIRA_API_BASE)
        )
    return jira.client.JIRA(
        {"server": JIRA_API_BASE}, token_auth=JIRA_ACCESS_TOKEN, timeout=(3.05, 30)
    )


# spark-connect-rust ships on its own connect-rust-x.y.z version line, so that its JIRAs
# appear in its release notes rather than in core Spark's.
CONNECT_RUST_VERSION = re.compile(r"connect-rust-(\d+)\.(\d+)\.(\d+)")


def warn(msg):
    print(msg, file=sys.stderr)


def unreleased_connect_rust_versions(asf_jira):
    """Return the unreleased, unarchived connect-rust-x.y.z versions, newest first.

    Versions compare numerically, so connect-rust-4.10.0 is newer than connect-rust-4.9.0.
    """
    versions = []
    for x in asf_jira.project_versions("SPARK"):
        match = CONNECT_RUST_VERSION.fullmatch(x.name)
        if match and not x.raw["released"] and not x.raw["archived"]:
            versions.append((tuple(int(n) for n in match.groups()), x))
    return [x for _, x in sorted(versions, key=lambda pair: pair[0], reverse=True)]


def detect_affected_version(asf_jira):
    """Return the newest unreleased connect-rust-x.y.z version, or exit."""
    versions = unreleased_connect_rust_versions(asf_jira)
    if not versions:
        fail(
            "Cannot detect an unreleased connect-rust-x.y.z version. "
            "Please create the ticket manually at %s" % JIRA_API_BASE
        )
    return versions[0].name


def find_release_umbrella(asf_jira, version):
    """Return the key of the release umbrella Epic of `version`, or None unless exactly one.

    The summary search is a fuzzy text match, so only an Epic with exactly the umbrella
    summary counts.
    """
    summary = "spark-connect-rust %s release" % version
    issues = asf_jira.search_issues(
        'project = SPARK AND issuetype = Epic AND summary ~ "\\"%s\\""' % summary
    )
    keys = [i.key for i in issues if i.fields.summary == summary]
    return keys[0] if len(keys) == 1 else None


def list_components(asf_jira):
    """Print all non-archived Spark JIRA components."""
    components = asf_jira.project_components("SPARK")
    components = [c for c in components if not c.raw.get("archived", False)]
    for c in sorted(components, key=lambda x: x.name):
        print(c.name)


def create_jira_issue(
    asf_jira, title, component, parent=None, issue_type=None, version=None, description=""
):
    """Create a JIRA issue and return the issue key (e.g. SPARK-12345).

    Affects and Fix Version/s are both `version`, an unreleased connect-rust-x.y.z version,
    by default the newest one. Issues other than subtasks are linked as "is part of" that
    version's release umbrella Epic, when there is exactly one. Once the issue exists, a
    failing later step only warns, so that its key is still returned.
    """
    if version:
        if version not in [v.name for v in unreleased_connect_rust_versions(asf_jira)]:
            fail("%s is not an unreleased connect-rust-x.y.z version." % version)
    else:
        version = detect_affected_version(asf_jira)

    issue_dict = {
        "project": {"key": "SPARK"},
        "summary": title,
        "description": description or "",
        "versions": [{"name": version}],
        "components": [{"name": component}],
    }

    if parent:
        issue_dict["issuetype"] = {"name": "Sub-task"}
        issue_dict["parent"] = {"key": parent}
    else:
        issue_dict["issuetype"] = {"name": issue_type if issue_type else "Improvement"}

    try:
        new_issue = asf_jira.create_issue(fields=issue_dict)
    except Exception as e:
        fail("Failed to create JIRA issue: %s" % e)

    # Setting a Fix Version can need a permission that contributors lack, and the create
    # screen omits the field for them, so set it separately and only warn on failure.
    try:
        new_issue.update(fields={"fixVersions": [{"name": version}]})
    except Exception as e:
        warn(
            "Created %s, but could not set its Fix Version to %s: %s" % (new_issue.key, version, e)
        )

    if not parent:
        try:
            umbrella = find_release_umbrella(asf_jira, version)
            if umbrella:
                # The umbrella incorporates the new issue, which "is part of" the umbrella.
                asf_jira.create_issue_link(
                    "Incorporates", inwardIssue=umbrella, outwardIssue=new_issue.key
                )
            else:
                warn(
                    "No single release umbrella Epic for %s; link %s to it by hand."
                    % (version, new_issue.key)
                )
        except Exception as e:
            warn(
                "Created %s, but could not link it to the %s release umbrella: %s"
                % (new_issue.key, version, e)
            )
    return new_issue.key
