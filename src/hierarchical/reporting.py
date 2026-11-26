"""Offline summaries of existing evaluation reports; never executes policies."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

COLORS = ('#777777', '#2a9d8f', '#457b9d', '#e76f51', '#e9c46a', '#9b5de5')


def records(report):
    return [episode['race_record'] for episode in report['episode_results']]


def summarize(paths):
    """Rows remain separated by method, map and immutable skill-bank identity."""
    matrix, segments, switches = [], [], []
    expected_bank = None
    usage = defaultdict(int)
    transitions = defaultdict(int)
    for path in paths:
        report = json.loads(Path(path).read_text())
        for record in records(report):
            hierarchy = record.get('hierarchical', {})
            bank = tuple(entry['sha256'] for entry in hierarchy.get('skills', []))
            if bank:
                if expected_bank is not None and bank != expected_bank:
                    raise ValueError('Reports use different adapter banks; summarize them separately')
                expected_bank = bank
            method = str(Path(path).parent)
            matrix.append({'method': method, 'map': record['map_id'],
                'seed': record['seed'], 'bank': '|'.join(bank),
                'clean_completion': int(record['strict_clean_finish']),
                'duration_s': record['duration_s'],
                'clean_finish_time_s': record.get('mean_clean_finish_time_s'), 'progress': record['mean_net_progress_laps'],
                'crash': int(record['any_learner_collision_dnf'])})
            for i, row in enumerate(hierarchy.get('transition_counts', [])):
                for j, count in enumerate(row):
                    transitions[(method, i, j)] += count
            trace = hierarchy.get('trajectory', [])
            visits = []
            for position, row in enumerate(trace):
                usage[(method, row.get('geometry', 'unknown'), row['skill'])] += 1
                if row['switched']:
                    future = trace[position:position+20]
                    switches.append({'method': method, 'map': record['map_id'], **row,
                        'post_switch_crash_20ll': int(any(r.get('collision', False) for r in future)),
                        'post_switch_max_lateral_error_20ll': max(abs(r['lateral_error']) for r in future),
                        'post_switch_forward_centerline_progress_m': (future[-1]['forward_centerline_progress_m'] - row['before_forward_centerline_progress_m']
                            if 'before_forward_centerline_progress_m' in row else None),
                        'observed_post_switch_ll': len(future),
                        'observed_post_switch_seconds': sum(r.get('decision_dt', 0.) for r in future),
                        'post_switch_window_complete': len(future) == 20,
                        'post_switch_window_ended_episode': bool(len(future) < 20 and (future[-1].get('terminated') or future[-1].get('truncated')))})
                key = row.get('maneuver_id', 'unknown')
                if not visits or visits[-1][0] != key:
                    visits.append((key, []))
                visits[-1][1].append(row)
            for key, rows in visits:
                segments.append({'method': method, 'map': record['map_id'], 'segment': key,
                    'duration_s': (sum(row['decision_dt'] for row in rows) if all('decision_dt' in row for row in rows)
                                   else rows[-1]['timestamp'] - rows[0]['timestamp']),
                    'entry_speed': rows[0]['speed'], 'exit_speed': rows[-1]['speed'],
                    'mean_abs_lateral_error': statistics.mean(abs(row['lateral_error']) for row in rows),
                    'mean_command_change': statistics.mean(row['command_jump'] for row in rows),
                    'partial_visit': rows is visits[0][1] or rows is visits[-1][1]})
    return {'matrix': matrix, 'segments': segments, 'switches': switches,
        'utilization': [dict(method=m, geometry=g, skill=s, ll_decisions=n) for (m,g,s),n in usage.items()],
        'transitions': [dict(method=m, source=i, target=j, count=n) for (m,i,j),n in transitions.items()]}


def trajectory_svg(trace, path):
    """Color-coded skill trajectory using stdlib only; suitable for offline viewing."""
    if not trace:
        raise ValueError('Enable hierarchical.record_trajectory for this evaluation')
    xs, ys = [r['x'] for r in trace], [r['y'] for r in trace]
    scale = 740 / max(max(xs)-min(xs), max(ys)-min(ys), 1.)
    points = [(30+(x-min(xs))*scale, 770-(y-min(ys))*scale) for x,y in zip(xs,ys)]
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800">', '<rect width="800" height="800" fill="white"/>']
    for i in range(1, len(points)):
        x,y = points[i-1]; u,v = points[i]
        lines.append(f'<path d="M{x},{y} L{u},{v}" stroke="{COLORS[trace[i]["skill"] % len(COLORS)]}" stroke-width="3"/>')
        if trace[i]['switched']:
            lines.append(f'<circle cx="{u}" cy="{v}" r="4" fill="black"/>')
    lines.append('</svg>')
    Path(path).write_text('\n'.join(lines))


def write_reports(paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = summarize(paths)
    for name, rows in result.items():
        if rows:
            with (output / f'{name}.csv').open('w', newline='') as handle:
                # CSV retains nested diagnostic actions as compact list representations.
                writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
                writer.writeheader()
                writer.writerows(rows)
    for index, path in enumerate(paths):
        report = json.loads(Path(path).read_text())
        for episode, record in enumerate(records(report)):
            trace = record.get('hierarchical', {}).get('trajectory')
            if trace:
                trajectory_svg(trace, output / f'trajectory_{index}_{episode}.svg')
    return result


def validation_best_fixed(paths):
    """Reuse checkpoint ranking; refuse test results, mixed banks or unmatched seeds."""
    from training.hooks import EvaluationCheckpointHook
    candidates = []
    reference = None
    for path in paths:
        report = json.loads(Path(path).read_text())
        if report.get('protocol') != 'selection':
            raise ValueError('Best fixed skill selection requires validation protocol reports')
        episodes = records(report)
        if any('validation_composite_' not in row['map_id'] for row in episodes):
            raise ValueError('Best fixed skill must be selected on composite validation tracks only')
        if not episodes:
            raise ValueError('Fixed comparison report contains no episodes')
        settings = report.get('seeds'), [r['map_id'] for r in episodes], episodes[0]['hierarchical']['skills']
        if reference is not None and settings != reference:
            raise ValueError('Fixed comparisons require identical banks, maps and seeds')
        reference = settings
        selector = (report.get('actor_identity') or {}).get('selector', {})
        if selector.get('type') != 'fixed' or type(selector.get('skill')) is not int:
            raise ValueError('Provide reports from an explicitly configured fixed selector')
        active = {selector['skill']}
        score = EvaluationCheckpointHook.selection_score(report['summary'], 'completion_progress')
        candidates.append((score, -next(iter(active)), str(path)))
    if not candidates:
        raise ValueError('No fixed-selector reports')
    score, negative_skill, path = max(candidates)
    return {'skill': -negative_skill, 'validation_report': path, 'selection_score': score}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--best-fixed', action='store_true', help='Select a fixed skill using composite validation reports only')
    args = parser.parse_args()
    if args.best_fixed:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'best_fixed.json').write_text(json.dumps(validation_best_fixed(args.reports), indent=2))
    else:
        write_reports(args.reports, args.output)
