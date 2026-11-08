# Saved results

This folder contains the completed experiment results, including improvements, ties and worse outcomes.

| Status | Count |
| :--- | ---: |
| Configurations in the original plan | 893 |
| Configurations completed for their full training length | 705 |
| Configurations deferred after the budget change | 188 |
| Completed task groups with test evaluation after parameter selection | 6 |

## Files

| Path | Contents |
| :--- | :--- |
| `revision/ID/result.json` | Metrics, training settings, final-epoch results and validation-selected checkpoint results when available |
| `revision/ID/config.json` | The saved settings for that run |
| `revision/ID/deferred.json` | The decision to defer a configuration; this is not a completed or failed training run |
| `revision/_extension_selection/` | Test results after each task's 30-run search and validation-based selection |
| `historical/` | Earlier experiments that used different settings from the revised benchmark |
| [`paper/evidence/completed_results.jsonl`](../paper/evidence/completed_results.jsonl) | All completed result records in one file |

The metric values are unchanged from the original records. Computer-specific administrative fields have been removed. Queue priority changed for 25 mechanism runs; it only determines run order. The result check therefore ignores that field and compares every training and evaluation setting exactly.

## Check the records

From the repository root, run:

```bash
python scripts/summarize_results.py
```

The command checks completion status, training lengths and configuration matches. It then prints counts by task family. It does not train models or download data.

Full training logs and model checkpoints are created when you run the experiments and are not stored in Git. The exact data for all three paper loss curves are in [`paper/figure_data/`](../paper/figure_data/). The paper appendix includes the complete comparisons and explains which configurations were completed or deferred.
