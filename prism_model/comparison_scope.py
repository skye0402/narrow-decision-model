"""Primary-family category coverage, separate from aggregate quality targets."""


def category_coverage(rows, config):
    if any(not r['primary'] for r in rows) or len({r['family_id'] for r in rows}) != len(rows):
        raise ValueError('Category qualification requires distinct primary families')
    result = {}
    for group in config['default_policies']:
        selected = [r for r in rows if r['expected']['group'] == group]
        ready = [r for r in selected if r['prediction']['status'] == 'ready_for_buyer_review']
        truly = sum(r['expected']['status'] == 'ready_for_buyer_review' for r in selected)
        correct = sum(r['expected']['status'] == 'ready_for_buyer_review'
            and r['prediction']['group'] == group
            and r['prediction']['identification'] == r['expected']['identification'] for r in ready)
        result[group] = {'families': len(selected), 'truly_ready': truly,
            'predicted_ready': len(ready), 'correct_ready': correct, 'false_ready': len(ready)-correct,
            'ready_recall': correct/truly if truly else None}
    return result


def covered(coverage):
    """Do not substitute a service that disables entire declared categories."""
    return bool(coverage) and all(v['truly_ready'] > 0 and v['correct_ready'] > 0 for v in coverage.values())


def annotate(summary, rows, config):
    coverage = category_coverage(rows, config)
    aggregate = summary['eligible']
    return {**summary, 'aggregate_targets_eligible': aggregate, 'category_coverage': coverage,
            'supported_category_scope_covered': covered(coverage),
            'eligible': aggregate and covered(coverage)}
