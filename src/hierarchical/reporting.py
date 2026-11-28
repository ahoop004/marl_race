"""Offline summaries of existing evaluation reports; never executes policies."""
from collections import defaultdict
import csv
from html import escape
import json
from math import sqrt
from pathlib import Path
import statistics

import numpy as np

COLORS = ('#777777', '#2a9d8f', '#457b9d', '#e76f51', '#e9c46a', '#9b5de5')
ACTIVE_THRESHOLD = 1e-6
COLLISION_WINDOW_LL = 20
F110_MAPS = {'Budapest_map', 'Hockenheim_map', 'Melbourne_map', 'Montreal_map',
             'Shanghai_map', 'Silverstone_map', 'Spa_map', 'Spielberg_map'}


def records(report):
    return [episode.get('race_record', episode) for episode in report['episode_results']]


def _is_composition(hierarchy):
    trace = hierarchy.get('trajectory') or []
    return (hierarchy.get('routing_type') == 'composition' or 'composition_trajectory' in hierarchy
            or any(key.startswith('composition/') for key in hierarchy.get('metrics', {}))
            or bool(trace and 'dominant_option' in trace[0]))


def _map_domain(report, record):
    map_id = record['map_id']
    provenance = report.get('evaluation_provenance') or {}
    protocol = provenance.get('map_protocols', {}).get(map_id, {})
    domain = (record.get('map_domain') or record.get('domain')
              or report.get('map_domains', {}).get(map_id) or protocol.get('domain'))
    if isinstance(domain, str) and domain:
        return domain.lower()
    path = Path(protocol.get('yaml_path') or map_id or '')
    if 'maps_generated' in path.parts or ('tracks' in path.parts and path.stem.startswith('research_')):
        return 'generated'
    if path.stem in F110_MAPS:
        return 'f110'
    return None


def _active(coefficients):
    return np.abs(coefficients) > ACTIVE_THRESHOLD


def _utilization(coefficients, names, context):
    for i, name in enumerate(names):
        values = coefficients[:, i]
        yield {**context, 'adapter': name, 'll_decisions': len(values),
               'mean_coefficient': float(values.mean()), 'mean_absolute_coefficient': float(np.abs(values).mean()),
               'active_fraction': float(_active(values).mean()), 'active_threshold': ACTIVE_THRESHOLD}


def _composition_trace(hierarchy, names, context, tables, geometry, geometry_counts):
    macros = hierarchy.get('composition_trajectory') or []
    if not macros or not names:
        return {}
    vectors = np.asarray([row['coefficients'] for row in macros], dtype=float)
    indices = {row['hl_decision_index']: i for i, row in enumerate(macros)}
    active = _active(vectors)
    residual_active = active[:, 1:]
    pairs = residual_active.astype(np.int64).T @ residual_active / len(vectors)
    metrics = {f'composition/active_fraction/{name}': float(active[:, i].mean())
               for i, name in enumerate(names)}
    metrics.update({f'composition/coactivation/{a}/{b}': float(pairs[i, j])
                    for i, a in enumerate(names[1:]) for j, b in enumerate(names[1:])})
    metrics['composition/active_residual_adapters'] = float(residual_active.sum(axis=1).mean())
    if residual_active.size:
        metrics['composition/inactive_residual_fraction'] = 1 - float(residual_active.mean())

    trace = hierarchy.get('trajectory') or []
    for i, row in enumerate(trace):
        decision = indices.get(row.get('hl_decision_index'))
        if decision is None:
            continue
        label = row.get('geometry')
        if label not in (None, '', 'unknown'):
            key = context['method'], context['distribution'], label
            sums = geometry.setdefault(key, np.zeros((3, len(names))))
            sums += (vectors[decision], np.abs(vectors[decision]), active[decision])
            geometry_counts[key] += 1
        previous = trace[i - 1] if i else {}
        previous_index = indices.get(previous.get('hl_decision_index'))
        before, after = previous.get('maneuver_id'), row.get('maneuver_id')
        if (previous_index is not None and before not in (None, '', 'unknown')
                and after not in (None, '', 'unknown') and before != after):
            successor = macros[decision + 1] if decision + 1 < len(macros) else {}
            tables['composition_maneuver_transitions'].append(
                {**context, 'll_decision_index': i, 'hl_decision_index': row['hl_decision_index'],
                 'forward_centerline_progress_m': row.get('forward_centerline_progress_m'),
                 'from_maneuver': before, 'to_maneuver': after, 'from_geometry': previous.get('geometry'),
                 'to_geometry': label, 'coefficients_before': macros[previous_index]['coefficients'],
                 'coefficients_at_transition': macros[decision]['coefficients'],
                 'l1_change_at_transition': float(np.abs(vectors[decision] - vectors[previous_index]).sum()),
                 'l1_change_at_previous_hl': macros[previous_index].get('coefficient_l1_change'),
                 'next_hl_decision_index': successor.get('hl_decision_index'),
                 'coefficients_at_next_hl': successor.get('coefficients'),
                 'l1_change_at_next_hl': successor.get('coefficient_l1_change')})
        if row.get('collision') and not previous.get('collision'):
            start = max(0, i - COLLISION_WINDOW_LL)
            prior = [(j, indices.get(trace[j].get('hl_decision_index'))) for j in range(start, i)]
            covered = [hl for _, hl in prior if hl is not None]
            changes = [macros[hl]['coefficient_l1_change'] for j, hl in prior
                       if hl is not None and j and trace[j].get('hl_decision_index') != trace[j-1].get('hl_decision_index')
                       and macros[hl].get('coefficient_l1_change') is not None]
            collision = {**context, 'collision_ll_decision_index': i,
                         'collision_hl_decision_index': row['hl_decision_index'], 'geometry': label,
                         'pre_collision_window_ll': COLLISION_WINDOW_LL, 'recorded_pre_collision_ll': len(covered),
                         'observed_hl_changes': len(changes),
                         'mean_pre_collision_l1_change': statistics.fmean(changes) if changes else None,
                         'collision_macro_l1_change': macros[decision].get('coefficient_l1_change'),
                         'collision_coefficients': macros[decision]['coefficients']}
            tables['composition_collisions'].append(collision)
            if covered:
                tables['composition_pre_collision_utilization'].extend(_utilization(vectors[covered], names, collision))
    return metrics


def _weighted(rows, key, weight=lambda metrics: metrics.get('hierarchical/hl_decisions', 1)):
    samples = [(m[key], weight(m)) for m, _ in rows if m.get(key) is not None]
    samples = [(value, n) for value, n in samples if n > 0]
    if samples:
        values, weights = zip(*samples)
        return statistics.fmean(values, weights)


def _composition_summaries(groups, tables):
    for (scope, method, map_id, domain, distribution), rows in groups.items():
        context = dict(scope=scope, method=method, map=map_id, domain=domain, distribution=distribution)
        names = list(dict.fromkeys(entry['name'] for _, h in rows for entry in h.get('skills', [])))
        stability = {}
        for key in ('effective_residual_adapters', 'zero_residual_fraction', 'active_residual_adapters',
                    'inactive_residual_fraction',
                    'negative_fraction', 'total_absolute_residual_weight', 'dominant_mean_dwell_ll_decisions',
                    'dominant_transitions_per_metre', 'dominant_transitions_per_second', 'mean_l1_change'):
            metric = f'composition/{key}'
            if key == 'mean_l1_change':
                value = _weighted(rows, metric, lambda m: max(0, m.get('hierarchical/hl_decisions', 2) - 1))
            elif key == 'effective_residual_adapters':
                value = _weighted(rows, metric, lambda m: m.get('hierarchical/hl_decisions', 1)
                                  * (1 - m.get('composition/zero_residual_fraction', 0)))
                if value is None and any(m.get(metric) == 0 for m, _ in rows):
                    value = 0.
            else:
                value = _weighted(rows, metric)
            if value is not None:
                stability[key] = value
        transitions = [m['composition/dominant_transitions'] for m, _ in rows if 'composition/dominant_transitions' in m]
        if transitions:
            stability['dominant_transitions'] = sum(transitions)
        if stability:
            tables['composition_stability'].append({**context, 'episodes': len(rows), **stability})

        for i, a in enumerate(names):
            values = {key: _weighted(rows, f'composition/{key}/{a}')
                      for key in ('mean', 'mean_absolute', 'active_fraction', 'dominant_selection_fraction')}
            for stat, reduce in (('min', min), ('max', max)):
                key = f'composition/{stat}/{a}'
                values[stat] = reduce((m[key] for m, _ in rows if key in m), default=None)
            for j, b in enumerate(names):
                pair = {**context, 'source': a, 'target': b}
                key = f'composition/covariance/{a}/{b}'
                mean_a, mean_b = f'composition/mean/{a}', f'composition/mean/{b}'
                samples = [(m[key], m[mean_a], m[mean_b], m.get('hierarchical/hl_decisions'))
                           for m, _ in rows if {key, mean_a, mean_b} <= m.keys()
                           and m.get('hierarchical/hl_decisions', 1) > 0]
                if samples:
                    cov, x, y, counts = zip(*samples)
                    weights = counts if None not in counts else None
                    cross = statistics.fmean((c + u*v for c, u, v in zip(cov, x, y)), weights)
                    value = cross - statistics.fmean(x, weights) * statistics.fmean(y, weights)
                    if i == j:
                        value = max(value, 0.)
                        values['std'] = sqrt(value)
                    tables['composition_covariance'].append({**pair, 'covariance': value,
                        'episodes': len(samples), 'hl_decisions': sum(weights) if weights is not None else None})
                coactivation = _weighted(rows, f'composition/coactivation/{a}/{b}')
                if coactivation is not None:
                    tables['composition_coactivation'].append({**pair, 'coactivation_fraction': coactivation,
                                                               'active_threshold': ACTIVE_THRESHOLD})
                if i != j:
                    count = sum(counts[i][j] for _, h in rows if (counts := h.get('transition_counts')))
                    if count:
                        tables['composition_dominant_transitions'].append({**pair, 'count': count})
            if any(value is not None for value in values.values()):
                tables['composition_coefficients'].append({**context, 'adapter': a, 'episodes': len(rows), **values})

        prefix = 'composition/influence/'
        metrics = dict.fromkeys(key for m, _ in rows for key in m if key.startswith(prefix))
        for key in metrics:
            samples = sum(m.get('composition/influence_samples', 0) for m, _ in rows if key in m)
            if samples:
                layer, measurement = key.removeprefix(prefix).split('/', 1)
                tables['composition_layer_influence'].append(
                    {**context, 'layer': layer, 'measurement': measurement, 'samples': samples,
                     'mean': _weighted(rows, key, lambda m: m.get('composition/influence_samples', 0))})


def summarize(paths):
    """Rows remain separated by method, map and immutable skill-bank identity."""
    matrix, segments, switches = [], [], []
    expected_bank = None
    usage = defaultdict(int)
    transitions = defaultdict(int)
    groups, domains, tables = defaultdict(list), defaultdict(list), defaultdict(list)
    geometry, geometry_counts = {}, defaultdict(int)
    skill_order = {}
    for path in paths:
        report = json.loads(Path(path).read_text())
        for record in records(report):
            hierarchy = record.get('hierarchical', {})
            trace = hierarchy.get('trajectory') or []
            composition = _is_composition(hierarchy)
            bank = tuple(entry['sha256'] for entry in hierarchy.get('skills', []) if 'sha256' in entry)
            if bank:
                if expected_bank is not None and bank != expected_bank:
                    raise ValueError('Reports use different adapter banks; summarize them separately')
                expected_bank = bank
            method = (report.get('method') if composition else None) or str(Path(path).parent)
            map_id = record['map_id']
            row = {'method': method, 'map': map_id,
                'seed': record['seed'], 'bank': '|'.join(bank),
                'clean_completion': int(record['strict_clean_finish']),
                'duration_s': record['duration_s'],
                'clean_finish_time_s': record.get('mean_clean_finish_time_s'), 'progress': record['mean_net_progress_laps'],
                'crash': int(record['any_learner_collision_dnf'])}
            matrix.append(row)
            if composition:
                domain, distribution = _map_domain(report, record), hierarchy.get('distribution')
                row.update(routing_type='composition', domain=domain, distribution=distribution)
                context = {key: row[key] for key in ('method', 'map', 'seed', 'domain', 'distribution')}
                context['episode'] = record.get('environment_episode')
                names = [entry['name'] for entry in hierarchy.get('skills', [])]
                if names:
                    skill_order[method, distribution] = names
                metrics = {**hierarchy.get('metrics', {}), **_composition_trace(
                    hierarchy, names, context, tables, geometry, geometry_counts)}
                if hierarchy.get('composition_trajectory'):
                    metrics.setdefault('hierarchical/hl_decisions', len(hierarchy['composition_trajectory']))
                sample = (metrics, {key: hierarchy[key] for key in ('skills', 'transition_counts') if key in hierarchy})
                groups['method', method, None, None, distribution].append(sample)
                groups['map', method, map_id, domain, distribution].append(sample)
                if domain:
                    groups['domain', method, None, domain, distribution].append(sample)
                    domains[method, distribution, domain].append(row)
                continue
            for i, row in enumerate(hierarchy.get('transition_counts', [])):
                for j, count in enumerate(row):
                    transitions[(method, i, j)] += count
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
    _composition_summaries(groups, tables)
    for (method, distribution, label), sums in geometry.items():
        n = geometry_counts[method, distribution, label]
        for name, (mean, magnitude, active) in zip(skill_order[method, distribution], sums.T / n):
            tables['composition_geometry'].append(dict(method=method, distribution=distribution,
                geometry=label, adapter=name, ll_decisions=n, mean_coefficient=mean,
                mean_absolute_coefficient=magnitude, active_fraction=active, active_threshold=ACTIVE_THRESHOLD))
    for (method, distribution, domain), races in domains.items():
        means = {key: statistics.fmean(r[key] for r in races if r[key] is not None)
                 for key in ('clean_completion', 'crash', 'progress', 'duration_s')
                 if any(r[key] is not None for r in races)}
        tables['composition_domains'].append(dict(method=method, distribution=distribution, domain=domain,
            episodes=len(races), **{f'mean_{key}': value for key, value in means.items()}))
    return {'matrix': matrix, 'segments': segments, 'switches': switches,
        'utilization': [dict(method=m, geometry=g, skill=s, ll_decisions=n) for (m,g,s),n in usage.items()],
        'transitions': [dict(method=m, source=i, target=j, count=n) for (m,i,j),n in transitions.items()],
        **tables}


def _track_points(trace, size):
    xs, ys = [r['x'] for r in trace], [r['y'] for r in trace]
    x0, y0 = min(xs), min(ys)
    scale = (size - 60) / max(max(xs) - x0, max(ys) - y0, 1.)
    return [(30 + (x-x0)*scale, size-30 - (y-y0)*scale) for x, y in zip(xs, ys)]


def _svg(path, width, height, lines):
    Path(path).write_text('\n'.join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="13">',
        f'<rect width="{width}" height="{height}" fill="white"/>', *lines, '</svg>']))


def trajectory_svg(trace, path):
    """Color-coded skill trajectory using stdlib only."""
    if not trace:
        raise ValueError('Enable hierarchical.record_trajectory for this evaluation')
    points = _track_points(trace, 800)
    lines = []
    for i in range(1, len(points)):
        x,y = points[i-1]; u,v = points[i]
        lines.append(f'<path d="M{x},{y} L{u},{v}" stroke="{COLORS[trace[i]["skill"] % len(COLORS)]}" stroke-width="3"/>')
        if trace[i]['switched']:
            lines.append(f'<circle cx="{u}" cy="{v}" r="4" fill="black"/>')
    _svg(path, 800, 800, lines)


def _color(value, limit):
    if value is None:
        return '#e5e5e5'
    weight = min(abs(value) / limit, 1.)
    target = (177, 52, 45) if value < 0 else (34, 101, 170)
    return 'rgb({},{},{})'.format(*(round(247 + weight * (channel - 247)) for channel in target))


def _composition_svgs(hierarchy, output, suffix):
    macros = hierarchy.get('composition_trajectory') or []
    names = [entry['name'] for entry in hierarchy.get('skills', [])]
    if not macros or not names:
        return
    low = 0. if hierarchy.get('coefficient_constraint') in ('convex', 'independent_positive') else -1.
    end = macros[-1]['end_forward_centerline_progress_m']
    axis_end = end if end > 0 else 1.
    scale = 620 / axis_end
    lines = ['<text x="60" y="21">Execution coefficient by forward track progress</text>',
             '<path d="M54,40 V250 H674" fill="none" stroke="#555"/>',
             '<text x="315" y="298">Forward centerline progress (m)</text>',
             f'<text x="53" y="270">0</text><text x="674" y="270">{axis_end:.2g}</text>']
    for level in sorted({low, 0., 1.}):
        y = 250 - (level-low) * 210 / (1-low)
        lines.append(f'<text x="23" y="{y+4:.1f}">{level:g}</text>')
        lines.append(f'<path d="M54,{y:.1f} H674" stroke="#ddd"/>')
    vectors = {row['hl_decision_index']: row['coefficients'] for row in macros}
    for i, name in enumerate(names):
        color = COLORS[i % len(COLORS)]
        points = []
        for row in macros:
            y = 250 - (row['coefficients'][i]-low) * 210 / (1-low)
            points.extend(f'{54 + scale * row[f"{side}_forward_centerline_progress_m"]:.2f},{y:.2f}'
                          for side in ('start', 'end'))
        lines.extend((f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round"/>',
                      f'<text x="{58 + i*101}" y="315" fill="{color}">{escape(name)}</text>'))
    _svg(output / f'composition_progress_{suffix}.svg', 720, 330, lines)

    trace = hierarchy.get('trajectory') or []
    if not trace:
        return
    points = _track_points(trace, 240)
    lines = ['<text x="12" y="18">Track coefficients · red −1 · neutral 0 · blue +1</text>']
    for i, name in enumerate(names):
        lines.append(f'<g transform="translate({i % 3 * 240},{22 + i // 3 * 240})">'
                     f'<text x="12" y="15">{escape(name)}</text>')
        for j, row in enumerate(trace):
            values = vectors.get(row.get('hl_decision_index'))
            if values is None:
                continue
            x, y = points[j]
            color = _color(values[i], 1.)
            if j:
                u, v = points[j-1]
                lines.append(f'<path d="M{u:.2f},{v:.2f} L{x:.2f},{y:.2f}" stroke="{color}" stroke-width="3"/>')
            else:
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{color}"/>')
        lines.append('</g>')
    _svg(output / f'composition_track_{suffix}.svg', 720, 22 + 240*((len(names)+2)//3), lines)


def _heatmaps(output, name, title, rows, row_key, column_key, value_key, *, limit=None):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row['method'], row.get('distribution')][row[row_key], row[column_key]] = row[value_key]
    for index, ((method, distribution), cells) in enumerate(grouped.items()):
        labels = list(dict.fromkeys(y for y, _ in cells))
        columns = list(dict.fromkeys(x for _, x in cells))
        left = max(115, 9 * max(map(len, labels)) + 12)
        cell_width = max(87, 9 * max(map(len, columns)) + 12)
        scale = limit or max(abs(value) for value in cells.values()) or 1.
        heading = f'{title} · {method}' + (f' · {distribution}' if distribution else '')
        lines = [f'<text x="12" y="20">{escape(heading)}</text>',
                 f'<text x="12" y="40">Color limit ±{scale:.3g}. Blank cells have no data.</text>']
        for j, column in enumerate(columns):
            x = left + j * cell_width
            lines.append(f'<text x="{x+6}" y="64">{escape(column)}</text>')
        for i, label in enumerate(labels):
            y = 72 + i * 29
            lines.append(f'<text x="12" y="{y+18}">{escape(label)}</text>')
            for j, column in enumerate(columns):
                x = left + j * cell_width
                value = cells.get((label, column))
                lines.append(f'<rect x="{x}" y="{y}" width="{cell_width-2}" height="27" fill="{_color(value, scale)}"/>')
                if value is not None:
                    ink = 'white' if abs(value) / scale > .65 else '#222'
                    lines.append(f'<text x="{x+6}" y="{y+18}" fill="{ink}">{value:.3g}</text>')
        width = max(left + len(columns)*cell_width + 10, 8*len(heading) + 25)
        _svg(output / f'{name}_{index}.svg', width, 86 + len(labels)*29, lines)


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
            hierarchy = record.get('hierarchical', {})
            if _is_composition(hierarchy):
                _composition_svgs(hierarchy, output, f'{index}_{episode}')
            elif hierarchy.get('trajectory'):
                trajectory_svg(hierarchy['trajectory'], output / f'trajectory_{index}_{episode}.svg')
    _heatmaps(output, 'composition_geometry', 'Mean absolute coefficient by geometry',
              result.get('composition_geometry', []), 'geometry', 'adapter', 'mean_absolute_coefficient', limit=1.)
    _heatmaps(output, 'composition_covariance', 'Coefficient covariance',
              (row for row in result.get('composition_covariance', []) if row['scope'] == 'method'),
              'source', 'target', 'covariance')
    influence = [row for row in result.get('composition_layer_influence', []) if row['scope'] == 'method']
    _heatmaps(output, 'composition_layer_influence', 'Mean residual norms by layer',
              ({**row, 'measurement': row['measurement'].removeprefix('weighted_norm/')}
               for row in influence if row['measurement'] != 'relative_norm'),
              'layer', 'measurement', 'mean')
    _heatmaps(output, 'composition_relative_influence', 'Mean residual norm relative to base output',
              (row for row in influence if row['measurement'] == 'relative_norm'),
              'layer', 'measurement', 'mean')
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
