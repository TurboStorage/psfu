# Manufacturer RMA Summary

## Unit

- Make/model: ExampleCo EX-1000
- Wattage: 1000W
- Serial: SYNTH-0001
- Lot/date code: LOT-DEMO
- Test date: 2026-01-01
- Ambient temp: 22.00C

## Bench Isolation Result

- Known-good bench: synthetic-bench
- POST with returned PSU: pass
- Known-good PSU substitution performed: no
- Known-good PSU substitution result: not-done
- Coverage confidence: 0.40

## Defensible Statement

This run captured warning-level behavior. Retest with a known-good PSU substitution before making an RMA claim.

## Storage Workload Characterization

- Observed sustained-write behavior matched `cliff stable` with cliff ratio 0.238.

## Directly Observed Facts

- [power] 12V drift reached warning band (12.10V → 11.54V).

## Evidence Bundle

- Summary: summary.md
- Verdict: verdict.json
- Telemetry samples: samples.csv
- Event log: events.csv

## Scope Limitation

This statement is intentionally limited to what the bench test directly supports. If no multimeter readings were recorded, this package does not claim absolute rail-spec compliance, ripple compliance, or transient-response compliance.

