#!/usr/bin/env python3
"""
Compare CSLR training results: Baseline vs Hungarian 0.5
Only comparing first 13 epochs for fair comparison
"""

import json

# Read baseline log
baseline_results = []
with open('out/cslr_baseline/log.txt', 'r') as f:
    for line in f:
        if line.strip():
            baseline_results.append(json.loads(line))

# Read hungarian log
hungarian_results = []
with open('out/cslr_hungarian_0.5/log.txt', 'r') as f:
    for line in f:
        if line.strip():
            hungarian_results.append(json.loads(line))

# Compare first 13 epochs
print("="*80)
print("CSLR Results Comparison: Baseline vs Hungarian 0.5 (First 13 Epochs)")
print("="*80)
print()

print(f"{'Epoch':<8} {'Baseline WER':<15} {'Hungarian WER':<15} {'Diff':<10} {'Winner':<10}")
print("-"*80)

for i in range(min(13, len(baseline_results), len(hungarian_results))):
    b_wer = baseline_results[i]['test_wer']
    h_wer = hungarian_results[i]['test_wer']
    diff = h_wer - b_wer
    winner = "Baseline" if b_wer < h_wer else "Hungarian"

    print(f"{i:<8} {b_wer:<15.2f} {h_wer:<15.2f} {diff:<10.2f} {winner:<10}")

print()
print("="*80)
print("Summary Statistics (First 13 Epochs)")
print("="*80)

# Best WER in first 13 epochs
baseline_13 = baseline_results[:13]
hungarian_13 = hungarian_results[:13]

b_best_wer = min(epoch['test_wer'] for epoch in baseline_13)
b_best_epoch = [i for i, e in enumerate(baseline_13) if e['test_wer'] == b_best_wer][0]

h_best_wer = min(epoch['test_wer'] for epoch in hungarian_13)
h_best_epoch = [i for i, e in enumerate(hungarian_13) if e['test_wer'] == h_best_wer][0]

print(f"\nBaseline:")
print(f"  Best WER: {b_best_wer:.2f}% (epoch {b_best_epoch})")
print(f"  Final WER (epoch 12): {baseline_13[12]['test_wer']:.2f}%")
print(f"  Del Rate: {baseline_13[12]['test_del_rate']:.2f}%")
print(f"  Ins Rate: {baseline_13[12]['test_ins_rate']:.2f}%")
print(f"  Sub Rate: {baseline_13[12]['test_sub_rate']:.2f}%")

print(f"\nHungarian 0.5:")
print(f"  Best WER: {h_best_wer:.2f}% (epoch {h_best_epoch})")
print(f"  Final WER (epoch 12): {hungarian_13[12]['test_wer']:.2f}%")
print(f"  Del Rate: {hungarian_13[12]['test_del_rate']:.2f}%")
print(f"  Ins Rate: {hungarian_13[12]['test_ins_rate']:.2f}%")
print(f"  Sub Rate: {hungarian_13[12]['test_sub_rate']:.2f}%")

print(f"\nDifference (Hungarian - Baseline):")
print(f"  Best WER: {h_best_wer - b_best_wer:+.2f}%")
print(f"  Final WER: {hungarian_13[12]['test_wer'] - baseline_13[12]['test_wer']:+.2f}%")

# Count wins
baseline_wins = sum(1 for i in range(13) if baseline_13[i]['test_wer'] < hungarian_13[i]['test_wer'])
hungarian_wins = sum(1 for i in range(13) if hungarian_13[i]['test_wer'] < baseline_13[i]['test_wer'])

print(f"\nEpoch Wins (Lower WER):")
print(f"  Baseline: {baseline_wins}/13 epochs")
print(f"  Hungarian: {hungarian_wins}/13 epochs")

print()
print("="*80)
