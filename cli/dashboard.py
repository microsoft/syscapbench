"""Generate an interactive dashboard for SysCapBench runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # Plotly is optional for the SWE-Bench style leaderboard
    go = None  # type: ignore
    make_subplots = None  # type: ignore

from logger import logger


@dataclass
class BenchmarkEntry:
    """Light-weight container for a single benchmark run."""

    name: str
    model: str
    agent: str
    directory: Path
    avg_score: Optional[Dict[str, Any]] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    final_score: Optional[float] = None
    result_count: int = 0
    question_types: Dict[str, int] = field(default_factory=dict)
    score_by_test_paper: List[Dict[str, Any]] = field(default_factory=list)
    timestamp_label: Optional[str] = None
    variant: Optional[str] = None
    category: str = 'general'
    timestamp: Optional[datetime] = None

    @property
    def display_name(self) -> str:
        suffix = f' | {self.variant}' if self.variant else ''
        ts = f' ({self.timestamp_label})' if self.timestamp_label else ''
        return f'{self.name} | {self.model} | {self.agent}{suffix}{ts}'


class BenchmarkDashboard:
    """Parse benchmark outputs and generate Plotly figures."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.outputs_dir = self.results_dir
        self.benchmarks: List[BenchmarkEntry] = []
        self._validate_inputs()
        self.load_results()

    def _validate_inputs(self) -> None:
        if not self.results_dir.exists():
            raise FileNotFoundError(f'Results directory not found: {self.results_dir}')
        if not self.outputs_dir.exists():
            raise FileNotFoundError(f'Expected outputs directory missing: {self.outputs_dir}')

    def load_results(self) -> None:
        for bench_dir in sorted(self.outputs_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            entry = self._load_entry(bench_dir)
            if entry:
                self.benchmarks.append(entry)

    def _load_entry(self, bench_dir: Path) -> Optional[BenchmarkEntry]:
        metadata = self._parse_directory_metadata(bench_dir.name)
        avg_score = self._safe_load_json(bench_dir / 'avg_score.json')
        results = self._safe_load_json_lines(bench_dir / 'result.jsonl')

        metrics = self._flatten_metrics(avg_score) if avg_score else {}
        final_score = None
        if avg_score:
            final_score = avg_score.get('final_score')
            if final_score is None and isinstance(avg_score.get('score'), dict):
                score_dict = avg_score['score']
                llm_score = score_dict.get('llm_score')
                full_score = score_dict.get('full_score')
                if isinstance(llm_score, (int, float)) and isinstance(full_score, (int, float)) and full_score:
                    final_score = llm_score / full_score

        question_types: Dict[str, int] = {}
        for record in results:
            qtype = record.get('type')
            if qtype:
                question_types[qtype] = question_types.get(qtype, 0) + 1

        category = 'general'
        bench_name_lower = metadata['benchmark'].lower()
        if 'course' in bench_name_lower:
            category = 'course'
        elif 'cache' in bench_name_lower:
            category = 'cache'
        elif 'example' in bench_name_lower:
            category = 'example'

        return BenchmarkEntry(
            name=metadata['benchmark'],
            model=metadata['model'],
            agent=metadata['agent'],
            directory=bench_dir,
            avg_score=avg_score,
            results=results,
            metrics=metrics,
            final_score=final_score,
            result_count=len(results),
            question_types=question_types,
            score_by_test_paper=avg_score.get('score_by_test_paper', []) if isinstance(avg_score, dict) else [],
            timestamp_label=metadata.get('timestamp'),
            variant=metadata.get('variant'),
            category=category,
            timestamp=metadata.get('timestamp_dt'),
        )

    def _parse_directory_metadata(self, name: str) -> Dict[str, Optional[str]]:
        parts = name.split('__')
        benchmark = parts[0] if parts else 'unknown'
        model = parts[1] if len(parts) > 1 else 'unknown'
        agent = parts[2] if len(parts) > 2 else 'unknown'

        timestamp = None
        timestamp_dt: Optional[datetime] = None
        variant = None
        remainder: List[str] = []
        if len(parts) > 3:
            candidate = parts[-1]
            try:
                timestamp_dt = datetime.strptime(candidate, '%Y-%m-%d_%H-%M-%S')
                timestamp = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
                remainder = parts[3:-1]
            except ValueError:
                remainder = parts[3:]
        variant = '__'.join(remainder) if remainder else None
        return {
            'benchmark': benchmark,
            'model': model,
            'agent': agent,
            'timestamp': timestamp,
            'timestamp_dt': timestamp_dt,
            'variant': variant,
        }

    def _safe_load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)

    def _safe_load_json_lines(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _flatten_metrics(self, avg_score: Dict[str, Any]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for key, value in avg_score.items():
            if key == 'score_by_test_paper':
                continue
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (int, float)):
                        metrics[f'{key}.{sub_key}'] = float(sub_val)
        return metrics

    def _format_metric(self, value: Any) -> str:
        if isinstance(value, float):
            if abs(value) >= 1:
                return f'{value:.3f}'.rstrip('0').rstrip('.')
            return f'{value:.4f}'.rstrip('0').rstrip('.')
        if isinstance(value, int):
            return str(value)
        return str(value)

    def create_overview_table(self) -> Optional[go.Figure]:
        if go is None:
            return None
        if not self.benchmarks:
            return None

        headers = ['Benchmark', 'Model', 'Agent', 'Final Score', 'Key Metrics']
        rows: List[List[str]] = []

        for bench in self.benchmarks:
            metrics_summary = self._summarise_metrics(bench.metrics)
            final_score = 'N/A'
            if bench.final_score is not None:
                final_score = self._format_metric(bench.final_score)
            rows.append(
                [
                    bench.name,
                    bench.model,
                    bench.agent,
                    final_score,
                    metrics_summary or '—',
                ]
            )

        table = go.Table(
            header=dict(values=headers, fill_color='#d0f0ff', align='left', font=dict(size=14, color='black')),
            cells=dict(values=list(zip(*rows)), fill_color='#f5f8fa', align='left', font=dict(size=12)),
        )
        fig = go.Figure(data=[table])
        fig.update_layout(title='Benchmark Overview', height=280 + 24 * len(rows))
        return fig

    def _summarise_metrics(self, metrics: Dict[str, float], limit: int = 4) -> str:
        if not metrics:
            return ''
        pairs = []
        for key in sorted(metrics):
            if key == 'final_score':
                continue
            pairs.append(f'{key}: {self._format_metric(metrics[key])}')
        return ', '.join(pairs[:limit])

    def create_exam_benchmark_chart(self) -> Optional[go.Figure]:
        if go is None or make_subplots is None:
            return None
        exam_entries = [b for b in self.benchmarks if b.category == 'exam' and b.avg_score]
        if not exam_entries:
            return None

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=('Overall Score', 'Score by Test Paper', 'Question Type Distribution', 'Score Comparison'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'pie'}, {'type': 'bar'}]],
        )

        for bench in exam_entries:
            avg_score = bench.avg_score or {}
            score_block = avg_score.get('score', {})
            reference_block = avg_score.get('reference', {})

            llm_score = score_block.get('llm_score')
            full_score = score_block.get('full_score')
            reference_avg = reference_block.get('avg_score')

            if llm_score is not None:
                fig.add_trace(
                    go.Bar(
                        name=f'{bench.model} — LLM',
                        x=[bench.name],
                        y=[llm_score],
                        text=[llm_score],
                        textposition='auto',
                    ),
                    row=1,
                    col=1,
                )
            if reference_avg is not None:
                fig.add_trace(
                    go.Bar(
                        name=f'{bench.name} Ref Avg',
                        x=[bench.name],
                        y=[reference_avg],
                        text=[self._format_metric(reference_avg)],
                        textposition='auto',
                    ),
                    row=1,
                    col=1,
                )
            if full_score is not None:
                fig.add_trace(
                    go.Bar(
                        name=f'{bench.name} Full',
                        x=[bench.name],
                        y=[full_score],
                        text=[full_score],
                        textposition='auto',
                    ),
                    row=1,
                    col=1,
                )

            if bench.score_by_test_paper:
                paper_names = [self._shorten(tp.get('test_paper_name', ''), 36) for tp in bench.score_by_test_paper]
                llm_scores = [tp.get('llm_score', 0) for tp in bench.score_by_test_paper]
                ref_scores = [tp.get('reference_score_avg', 0) for tp in bench.score_by_test_paper]

                fig.add_trace(
                    go.Bar(
                        name=f'{bench.model} LLM', x=paper_names, y=llm_scores, text=llm_scores, textposition='auto'
                    ),
                    row=1,
                    col=2,
                )
                fig.add_trace(
                    go.Bar(
                        name=f'{bench.name} Ref Avg',
                        x=paper_names,
                        y=ref_scores,
                        text=[self._format_metric(v) for v in ref_scores],
                        textposition='auto',
                    ),
                    row=1,
                    col=2,
                )

            if bench.question_types:
                labels = list(bench.question_types.keys())
                values = list(bench.question_types.values())
                fig.add_trace(go.Pie(labels=labels, values=values, name=bench.name), row=2, col=1)

            if bench.results:
                type_scores: Dict[str, List[float]] = {}
                type_max: Dict[str, List[float]] = {}
                for record in bench.results:
                    qtype = record.get('type', 'Unknown')
                    type_scores.setdefault(qtype, []).append(record.get('llm_score', 0))
                    type_max.setdefault(qtype, []).append(record.get('points', 1))

                types = list(type_scores.keys())
                avg_scores = [sum(scores) / len(scores) for scores in type_scores.values()]
                avg_max = [sum(max_scores) / len(max_scores) for max_scores in type_max.values()]

                fig.add_trace(
                    go.Bar(
                        name=f'{bench.model} Avg Score',
                        x=types,
                        y=avg_scores,
                        text=[self._format_metric(v) for v in avg_scores],
                        textposition='auto',
                    ),
                    row=2,
                    col=2,
                )
                fig.add_trace(
                    go.Bar(
                        name=f'{bench.name} Avg Max',
                        x=types,
                        y=avg_max,
                        text=[self._format_metric(v) for v in avg_max],
                        textposition='auto',
                    ),
                    row=2,
                    col=2,
                )

        fig.update_layout(height=920, title_text='Exam Benchmark Analysis', barmode='group', showlegend=True)
        return fig

    def _shorten(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + '...'

    def create_cache_benchmark_chart(self) -> Optional[go.Figure]:
        if go is None:
            return None
        cache_entries = [b for b in self.benchmarks if b.category == 'cache' and b.avg_score]
        if not cache_entries:
            return None

        fig = go.Figure()
        metrics_of_interest = ['miss_rate', 'time_cost']
        for bench in cache_entries:
            values = [bench.metrics.get(metric, 0.0) for metric in metrics_of_interest]
            fig.add_trace(
                go.Bar(
                    name=bench.display_name,
                    x=metrics_of_interest,
                    y=values,
                    text=[self._format_metric(val) for val in values],
                    textposition='auto',
                )
            )

        fig.update_layout(title='Cache Benchmark Metrics', xaxis_title='Metric', yaxis_title='Value', height=520)
        return fig

    def create_similarity_metrics_chart(self) -> Optional[go.Figure]:
        if go is None:
            return None
        metrics = [
            'syntax_acc',
            'exact_match',
            'jaccard_similarity',
            'cosine_similarity',
            'embeddings_similarity',
            'llmjudger_rating',
        ]
        candidates = [b for b in self.benchmarks if any(metric in b.metrics for metric in metrics)]
        if not candidates:
            return None

        fig = go.Figure()
        for bench in candidates:
            values = [bench.metrics.get(metric, 0.0) for metric in metrics]
            fig.add_trace(go.Scatterpolar(r=values, theta=metrics, fill='toself', name=bench.display_name))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-0.5, 1])), title='Similarity Metrics', height=620
        )
        return fig

    def _render_source_links(self, entries: List[BenchmarkEntry], title: str, indent: str = '') -> str:
        if not entries:
            return ''

        rows = []
        for bench in entries:
            rel_dir = bench.directory.relative_to(self.results_dir)
            avg_path = rel_dir / 'avg_score.json'
            result_path = rel_dir / 'result.jsonl'
            link_avg = f'<a href="{avg_path.as_posix()}" target="_blank">avg_score.json</a>'
            link_result = f'<a href="{result_path.as_posix()}" target="_blank">result.jsonl</a>'
            rows.append(
                indent + '        <tr>'
                f'<td>{bench.name}</td>'
                f'<td>{bench.model}</td>'
                f'<td>{bench.agent}</td>'
                f'<td>{link_avg}</td>'
                f'<td>{link_result}</td>'
                '</tr>'
            )

        table_html = [
            f'{indent}<div class="card section-table">',
            f'{indent}    <h3>{title}</h3>',
            f'{indent}    <table class="metrics-table compact">',
            f'{indent}        <tr><th>Benchmark</th><th>Model</th><th>Agent</th><th>Avg Score</th><th>Results</th></tr>',
        ]
        table_html.extend(rows)
        table_html.append(f'{indent}    </table>')
        table_html.append(f'{indent}</div>')
        return '\n'.join(table_html)

    def _sorted_benchmarks(self) -> List[BenchmarkEntry]:
        return sorted(
            self.benchmarks,
            key=lambda b: (b.final_score is None, -b.final_score if b.final_score is not None else 0.0),
        )

    def _format_score_value(self, score: Optional[float]) -> str:
        if score is None:
            return '—'
        if -1.0 <= score <= 1.0:
            return f'{score * 100:.1f}%'
        return self._format_metric(score)

    def _format_timestamp(self, bench: BenchmarkEntry) -> str:
        if bench.timestamp:
            return bench.timestamp.strftime('%Y-%m-%d %H:%M')
        if bench.timestamp_label:
            return bench.timestamp_label
        return '—'

    def _primary_metric_label(self, bench: BenchmarkEntry) -> str:
        if bench.category == 'exam' and bench.avg_score:
            score_block = bench.avg_score.get('score', {}) if isinstance(bench.avg_score, dict) else {}
            llm_score = score_block.get('llm_score')
            full_score = score_block.get('full_score')
            if isinstance(llm_score, (int, float)) and isinstance(full_score, (int, float)) and full_score:
                return f'{int(llm_score)}/{int(full_score)} pts'
            ref_block = bench.avg_score.get('reference', {}) if isinstance(bench.avg_score, dict) else {}
            ref_avg = ref_block.get('avg_score')
            if isinstance(ref_avg, (int, float)):
                return f'Reference avg {self._format_metric(ref_avg)}'
        if bench.category == 'cache':
            miss_rate = bench.metrics.get('miss_rate')
            if miss_rate is not None:
                return f'Miss {self._format_score_value(miss_rate)}'
            time_cost = bench.metrics.get('time_cost')
            if time_cost is not None:
                return f'Time {self._format_metric(time_cost)}'
        for key in sorted(bench.metrics):
            if key == 'final_score':
                continue
            return f'{key}: {self._format_metric(bench.metrics[key])}'
        if bench.result_count:
            return f'{bench.result_count} records'
        return '—'

    def _artifact_links(self, bench: BenchmarkEntry) -> str:
        rel_dir = bench.directory.relative_to(self.results_dir)
        avg_path = rel_dir / 'avg_score.json'
        result_path = rel_dir / 'result.jsonl'
        return (
            f'<a href="{avg_path.as_posix()}" target="_blank">avg_score</a>'
            '<span class="artifact-separator">·</span>'
            f'<a href="{result_path.as_posix()}" target="_blank">results</a>'
        )

    def _metrics_for_card(self, bench: BenchmarkEntry, limit: int = 5) -> List[tuple[str, str]]:
        entries: List[tuple[str, str]] = []
        if bench.category == 'exam' and bench.avg_score:
            score_block = bench.avg_score.get('score', {}) if isinstance(bench.avg_score, dict) else {}
            ref_block = bench.avg_score.get('reference', {}) if isinstance(bench.avg_score, dict) else {}
            llm_score = score_block.get('llm_score')
            full_score = score_block.get('full_score')
            question_count = score_block.get('question_count')
            ref_avg = ref_block.get('avg_score')
            if isinstance(llm_score, (int, float)) and isinstance(full_score, (int, float)):
                entries.append(('LLM Score', f'{llm_score}/{full_score}'))
            if isinstance(question_count, (int, float)):
                entries.append(('Questions', str(int(question_count))))
            if isinstance(ref_avg, (int, float)):
                entries.append(('Ref Avg', self._format_metric(ref_avg)))
        elif bench.category == 'cache':
            miss_rate = bench.metrics.get('miss_rate')
            time_cost = bench.metrics.get('time_cost')
            if miss_rate is not None:
                entries.append(('Miss Rate', self._format_score_value(miss_rate)))
            if time_cost is not None:
                entries.append(('Time Cost', self._format_metric(time_cost)))
        if not entries:
            for key in sorted(bench.metrics):
                if key == 'final_score':
                    continue
                entries.append((key, self._format_metric(bench.metrics[key])))
        if bench.result_count:
            entries.append(('Records', str(bench.result_count)))
        return entries[:limit]

    def _compute_summary_stats(self) -> Dict[str, str]:
        total_runs = len(self.benchmarks)
        categories = sorted({bench.category for bench in self.benchmarks})
        scored_entries = [b for b in self.benchmarks if b.final_score is not None]
        best_entry = max(scored_entries, key=lambda b: b.final_score, default=None)
        avg_score = sum(b.final_score for b in scored_entries) / len(scored_entries) if scored_entries else None
        latest_entry = max(
            (b for b in self.benchmarks if b.timestamp),
            key=lambda b: b.timestamp,
            default=None,
        )

        stats = {
            'total_runs': str(total_runs),
            'category_count': str(len(categories)),
            'categories': ', '.join(cat.title() for cat in categories) if categories else '—',
            'avg_score': self._format_score_value(avg_score),
            'best_score': '—',
            'latest_run': '—',
        }
        if best_entry:
            stats['best_score'] = f'{self._format_score_value(best_entry.final_score)} · {best_entry.name}'
        if latest_entry:
            stats['latest_run'] = f'{self._format_timestamp(latest_entry)} · {latest_entry.name}'
        return stats

    def _render_hero_section(self, generated_at: str) -> str:
        return '\n'.join(
            [
                '<header class="hero">',
                '    <div class="hero-badge">SysCapBench</div>',
                '    <h1>System Intelligence Leaderboard</h1>',
                '    <p>SysCapBench is a comprehensive benchmarking framework for evaluating the performance of Large Language Models (LLMs) and AI systems across critical system-related tasks. This leaderboard showcases the results of various benchmarks conducted using this framework.</p>',
                '    <div class="hero-meta">',
                f'        <span>Results directory: <strong>{self.results_dir}</strong></span>',
                f'        <span>Generated: <strong>{generated_at}</strong></span>',
                '    </div>',
                '</header>',
            ]
        )

    def _render_summary_section(self, stats: Dict[str, str]) -> str:
        return '\n'.join(
            [
                '<section class="summary-grid">',
                '    <div class="summary-card">',
                '        <h3>Total Runs</h3>',
                f'        <p>{stats["total_runs"]}</p>',
                '    </div>',
                '    <div class="summary-card">',
                '        <h3>Categories</h3>',
                f'        <p>{stats["category_count"]}</p>',
                f'        <span class="summary-muted">{stats["categories"]}</span>',
                '    </div>',
                '    <div class="summary-card">',
                '        <h3>Average Score</h3>',
                f'        <p>{stats["avg_score"]}</p>',
                '    </div>',
                '    <div class="summary-card">',
                '        <h3>Latest Run</h3>',
                f'        <p>{stats["latest_run"]}</p>',
                '    </div>',
                '    <div class="summary-card">',
                '        <h3>Best Result</h3>',
                f'        <p>{stats["best_score"]}</p>',
                '    </div>',
                '</section>',
            ]
        )

    def _render_leaderboard_section(self) -> str:
        benches = self._sorted_benchmarks()
        if not benches:
            return '<section class="leaderboard-card"><h2>Leaderboard</h2><p>No runs found.</p></section>'

        rows = []
        for idx, bench in enumerate(benches, start=1):
            score_text = self._format_score_value(bench.final_score)
            metric_label = self._primary_metric_label(bench)
            timestamp = self._format_timestamp(bench)
            artifacts = self._artifact_links(bench)
            row_classes = 'leaderboard-row leaderboard-row-primary' if idx == 1 else 'leaderboard-row'
            rows.append(
                '        <tr class="' + row_classes + '">'
                f'<td class="col-rank">{idx}</td>'
                '<td class="col-benchmark">'
                f'<div class="bench-title">{bench.name}</div>'
                f'<div class="bench-meta"><span class="badge badge-category badge-{bench.category}">{bench.category.title()}</span>'
                + (f'<span class="badge badge-variant">{bench.variant}</span>' if bench.variant else '')
                + f'<span class="bench-timestamp">{timestamp}</span>'
                '</div></td>'
                '<td class="col-model">'
                f'<div class="model-name">{bench.model}</div>'
                f'<div class="model-agent">{bench.agent}</div>'
                '</td>'
                '<td class="col-score">'
                f'<div class="score-value">{score_text}</div>'
                '</td>'
                '<td class="col-metric">'
                f'<span class="metric-pill">{metric_label}</span>'
                '</td>'
                '<td class="col-records">'
                f'<span class="records-text">{bench.result_count} items</span>'
                f'<div class="artifact-links">{artifacts}</div>'
                '</td>'
                '        </tr>'
            )

        table = [
            '<section class="leaderboard-card">',
            '    <h2>Leaderboard</h2>',
            '    <table class="leaderboard-table">',
            '        <thead>',
            '            <tr><th>#</th><th>Benchmark</th><th>Model</th><th>Score</th><th>Primary Metric</th><th>Artifacts</th></tr>',
            '        </thead>',
            '        <tbody>',
        ]
        table.extend(rows)
        table.extend(['        </tbody>', '    </table>', '</section>'])
        return '\n'.join(table)

    def _render_detail_cards(self) -> str:
        benches = self._sorted_benchmarks()
        if not benches:
            return ''

        cards = [
            '<section class="details-section">',
            '    <h2 class="section-heading">Benchmark Snapshots</h2>',
            '    <div class="card-grid">',
        ]
        for bench in benches:
            metrics = self._metrics_for_card(bench)
            metrics_html = ''.join(
                f'<div class="metric-row"><span>{label}</span><span>{value}</span></div>' for label, value in metrics
            )
            cards.extend(
                [
                    '        <article class="detail-card">',
                    f'            <h3>{bench.name}</h3>',
                    f'            <p class="detail-meta">{bench.model} · {bench.agent}</p>',
                    f'            <div class="detail-score">Score {self._format_score_value(bench.final_score)}</div>',
                    f'            <div class="metric-stack">{metrics_html}</div>',
                    f'            <div class="detail-links">{self._artifact_links(bench)}</div>',
                    '        </article>',
                ]
            )
        cards.extend(['    </div>', '</section>'])
        return '\n'.join(cards)

    def generate_html_dashboard(self, output_file: str = 'dashboard.html') -> Path:
        output_path = self.results_dir / output_file
        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        stats = self._compute_summary_stats()
        hero_html = self._render_hero_section(generated_at)
        summary_html = self._render_summary_section(stats)
        leaderboard_html = self._render_leaderboard_section()
        detail_cards_html = self._render_detail_cards()

        exam_fig = self.create_exam_benchmark_chart()
        cache_fig = self.create_cache_benchmark_chart()
        similarity_fig = self.create_similarity_metrics_chart()
        chart_blocks = []
        if exam_fig:
            chart_blocks.append(('Exam Benchmark Breakdown', exam_fig))
        if cache_fig:
            chart_blocks.append(('Cache Benchmark Metrics', cache_fig))
        if similarity_fig:
            chart_blocks.append(('Similarity Metrics', similarity_fig))

        include_plotly = bool(chart_blocks)

        css = (
            ':root { color-scheme: dark; }\n'
            '* { box-sizing: border-box; }\n'
            "body { margin: 0; font-family: 'Inter', 'Segoe UI', sans-serif; background: radial-gradient(circle at top left, rgba(56,189,248,0.18), rgba(14,23,42,0.95) 46%), #03070f; color: #dbeafe; line-height: 1.6; }\n"
            'a { color: #7dd3fc; text-decoration: none; transition: color 0.2s ease; }\n'
            'a:hover { color: #bae6fd; }\n'
            'main.layout { max-width: 1100px; margin: 0 auto; padding: 64px 24px 96px; }\n'
            '@media (min-width: 1280px) { main.layout { padding: 72px 0 104px; } }\n'
            '.hero { position: relative; overflow: hidden; background: linear-gradient(135deg, rgba(15,23,42,0.92) 0%, rgba(30,64,175,0.55) 60%, rgba(14,165,233,0.45) 100%); border: 1px solid rgba(59,130,246,0.35); border-radius: 28px; padding: 48px; margin-bottom: 48px; box-shadow: 0 40px 120px -60px rgba(59,130,246,0.65); }\n'
            ".hero::after { content: ''; position: absolute; width: 340px; height: 340px; background: radial-gradient(circle, rgba(56,189,248,0.42), transparent 68%); right: -120px; top: -140px; opacity: 0.6; }\n"
            '.hero-badge { position: relative; display: inline-flex; align-items: center; padding: 8px 20px; border-radius: 999px; border: 1px solid rgba(125,211,252,0.45); background: rgba(56,189,248,0.18); color: #7dd3fc; letter-spacing: 0.28em; text-transform: uppercase; font-size: 0.72rem; font-weight: 600; }\n'
            '.hero h1 { position: relative; margin: 22px 0 16px; font-size: 2.9rem; letter-spacing: 0.03em; color: #f8fafc; }\n'
            '.hero p { position: relative; margin: 0; max-width: 560px; color: #bfdbfe; font-size: 1.05rem; }\n'
            '.hero-meta { position: relative; margin-top: 30px; display: flex; flex-wrap: wrap; gap: 16px; color: #cbd5f5; font-size: 0.9rem; }\n'
            '.hero-meta strong { color: #f8fafc; }\n'
            '.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 48px; }\n'
            '.summary-card { background: rgba(10,16,28,0.92); border: 1px solid rgba(148,163,184,0.22); border-radius: 20px; padding: 18px 22px; box-shadow: 0 18px 50px rgba(8,12,24,0.35); }\n'
            '.summary-card h3 { margin: 0; font-size: 0.75rem; letter-spacing: 0.14em; color: #94a3b8; text-transform: uppercase; }\n'
            '.summary-card p { margin: 12px 0 0; font-size: 1.32rem; font-weight: 600; color: #e2e8f0; }\n'
            '.summary-muted { display: block; margin-top: 6px; font-size: 0.75rem; color: #64748b; letter-spacing: 0.06em; }\n'
            '.leaderboard-card { background: rgba(11,18,32,0.94); border: 1px solid rgba(37,99,235,0.28); border-radius: 26px; padding: 34px; box-shadow: 0 28px 80px -40px rgba(30,64,175,0.75); }\n'
            '.leaderboard-card h2 { margin: 0 0 26px; font-size: 1.55rem; letter-spacing: 0.08em; text-transform: uppercase; color: #e0f2fe; }\n'
            '.leaderboard-table { width: 100%; border-collapse: collapse; }\n'
            '.leaderboard-table thead th { font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.18em; text-transform: uppercase; padding: 12px 16px; border-bottom: 1px solid rgba(148,163,184,0.25); text-align: left; }\n'
            '.leaderboard-table tbody td { padding: 18px 16px; border-bottom: 1px solid rgba(148,163,184,0.12); vertical-align: middle; }\n'
            '.leaderboard-row:hover { background: rgba(30,64,175,0.22); }\n'
            '.leaderboard-row-primary { background: rgba(37,99,235,0.24); box-shadow: inset 0 0 0 1px rgba(56,189,248,0.35); }\n'
            '.col-rank { width: 62px; font-weight: 600; color: #7c91c3; font-size: 0.85rem; text-align: center; }\n'
            '.bench-title { font-size: 1.05rem; font-weight: 600; color: #f8fafc; }\n'
            '.bench-meta { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; color: #94a3b8; font-size: 0.78rem; }\n'
            '.badge { display: inline-flex; align-items: center; justify-content: center; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.26); font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; }\n'
            '.badge-general { background: rgba(56,189,248,0.12); border-color: rgba(56,189,248,0.38); color: #7dd3fc; }\n'
            '.badge-exam { background: rgba(249,115,22,0.15); border-color: rgba(249,115,22,0.42); color: #fb923c; }\n'
            '.badge-cache { background: rgba(168,85,247,0.15); border-color: rgba(168,85,247,0.45); color: #c4b5fd; }\n'
            '.badge-variant { background: rgba(148,163,184,0.18); border-color: rgba(148,163,184,0.4); color: #cbd5f5; letter-spacing: 0.06em; text-transform: none; }\n'
            '.bench-timestamp { color: #64748b; font-size: 0.72rem; letter-spacing: 0.03em; }\n'
            '.model-name { font-weight: 600; color: #e2e8f0; font-size: 1rem; }\n'
            '.model-agent { margin-top: 6px; color: #94a3b8; font-size: 0.82rem; letter-spacing: 0.04em; }\n'
            '.score-value { font-size: 1.22rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.06em; }\n'
            '.metric-pill { display: inline-flex; align-items: center; padding: 6px 16px; border-radius: 999px; border: 1px solid rgba(94,234,212,0.28); background: rgba(15,118,110,0.22); color: #5eead4; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }\n'
            '.records-text { display: block; font-size: 0.78rem; color: #94a3b8; letter-spacing: 0.05em; text-transform: uppercase; }\n'
            '.artifact-links { margin-top: 10px; display: flex; gap: 12px; align-items: center; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }\n'
            '.artifact-links a { color: #a5b4fc; }\n'
            '.artifact-links a:hover { color: #c084fc; }\n'
            '.artifact-separator { color: #475569; }\n'
            '.details-section { margin-top: 52px; }\n'
            '.section-heading { font-size: 1.28rem; letter-spacing: 0.1em; text-transform: uppercase; color: #cbd5f5; margin-bottom: 22px; }\n'
            '.card-grid { display: grid; gap: 22px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }\n'
            '.detail-card { background: rgba(10,16,30,0.9); border: 1px solid rgba(148,163,184,0.2); border-radius: 22px; padding: 24px; box-shadow: 0 18px 60px -32px rgba(14,23,42,0.6); }\n'
            '.detail-card h3 { margin: 0; font-size: 1.08rem; color: #f8fafc; }\n'
            '.detail-meta { margin: 8px 0 18px; color: #94a3b8; font-size: 0.82rem; letter-spacing: 0.05em; }\n'
            '.detail-score { font-size: 1.18rem; font-weight: 600; color: #38bdf8; margin-bottom: 18px; }\n'
            '.metric-stack { display: grid; gap: 10px; }\n'
            '.metric-row { display: flex; justify-content: space-between; font-size: 0.84rem; color: #cbd5f5; }\n'
            '.metric-row span:last-child { color: #e0f2fe; font-weight: 500; }\n'
            '.detail-links { margin-top: 20px; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; display: flex; gap: 12px; }\n'
            '.insights-section { margin-top: 64px; }\n'
            '.chart-card { background: rgba(11,18,32,0.94); border: 1px solid rgba(37,99,235,0.26); border-radius: 24px; padding: 28px; margin-top: 24px; box-shadow: 0 28px 80px -46px rgba(30,64,175,0.65); }\n'
            '.chart-card h3 { margin: 0 0 16px; font-size: 1.1rem; color: #e0f2fe; letter-spacing: 0.06em; text-transform: uppercase; }\n'
            '@media (max-width: 720px) { .leaderboard-table thead { display: none; } .leaderboard-table, .leaderboard-table tbody, .leaderboard-table tr, .leaderboard-table td { display: block; width: 100%; } .leaderboard-table tr { margin-bottom: 18px; padding: 18px; border: 1px solid rgba(148,163,184,0.18); border-radius: 18px; } .leaderboard-table td { border: 0; padding: 8px 0; } .col-rank { text-align: left; } .metric-pill { margin-top: 8px; } .artifact-links { margin-top: 14px; } }\n'
        )

        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '    <meta charset="utf-8">',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '    <title>System Capability Leaderboard</title>',
        ]
        if include_plotly:
            html_parts.append('    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
        html_parts.append('    <style>')
        html_parts.append(css)
        html_parts.append('    </style>')
        html_parts.append('</head>')
        html_parts.append('<body>')
        html_parts.append('    <main class="layout">')
        html_parts.append(hero_html)
        html_parts.append(summary_html)
        html_parts.append(leaderboard_html)
        if detail_cards_html:
            html_parts.append(detail_cards_html)
        if chart_blocks:
            html_parts.append('    <section class="insights-section">')
            html_parts.append('        <h2 class="section-heading">Analytics</h2>')
            for title, figure in chart_blocks:
                html_parts.append('        <div class="chart-card">')
                html_parts.append(f'            <h3>{title}</h3>')
                html_parts.append(figure.to_html(full_html=False, include_plotlyjs=False))
                html_parts.append('        </div>')
            html_parts.append('    </section>')
        html_parts.append('    </main>')
        html_parts.append('</body>')
        html_parts.append('</html>')

        output_path.write_text('\n'.join(html_parts), encoding='utf-8')
        return output_path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate dashboard from SysCapBench results')
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to the results directory (e.g. all_results_2025-10-02_15-57-15)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dashboard.html',
        help='Name of the output HTML file to create',
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    dashboard = BenchmarkDashboard(args.results_dir)
    output_path = dashboard.generate_html_dashboard(args.output)
    logger.info('Dashboard generated: %s', output_path)


if __name__ == '__main__':
    main()
