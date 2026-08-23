#!/usr/bin/env bash
# No-stub audit - enforces the "100% coverage, no followups" rule.
#
# Fails (non-zero) if any placeholder/deferral markers remain in the Rust
# sources. "Done" means implemented and tested, never stubbed. Run in CI and
# before claiming any phase complete.
set -uo pipefail
cd "$(dirname "$0")/.."

# Markers that unambiguously indicate deferred/incomplete work. We deliberately
# do NOT match bare "stub"/"placeholder" (they appear in legitimate identifiers
# like a tonic client field or in `<param_name>` substitution comments); real
# deferrals always carry one of the markers below.
PATTERN='unimplemented!\(|todo!\(|unreachable!\("(stub|todo|not)|TODO|FIXME|\bXXX\b|not[[:space:]]+yet[[:space:]]+implemented|stretch goal|for[[:space:]]+now|deferred|come back later|to be implemented|[Ff]uture implementation|[Ff]uture work|will be implemented|no-op placeholder'

hits=$(grep -rInE "$PATTERN" crates/*/src crates/*/build.rs 2>/dev/null)

if [[ -n "$hits" ]]; then
  echo "NO-STUB AUDIT FAILED - remove these placeholders (implement fully):"
  echo "$hits"
  echo
  echo "count: $(echo "$hits" | wc -l | tr -d ' ')"
  exit 1
fi
echo "no-stub audit passed: no placeholders in crates/*/src"
