"""Generate interactive expected-vs-actual plots for the drivetrain test CSVs.

Usage: python plot_tests.py [data_dir] [out_dir]   (defaults: data, docs)

Files (searched in data_dir and its subfolders) must be named
angle_test_<wheel angles>_<commanded angle>_<trial>.csv; anything else is skipped.
If a file holds several runs, only the first is used. Output:
  - one page per CSV: XY path, heading over time, speed over time
  - one overlay page per commanded angle with every trial on the same plots
  - index.html linking to everything
"""
import csv
import html
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
import plotly.offline

# Also accepts a browser duplicate-download suffix, e.g. "angle_test_2540_0_1 (1).csv".
FILENAME_RE = re.compile(r"^angle_test_(\d+)_(-?\d+)_(\d+)(?: \(\d+\))?\.csv$")

EXPECTED_COLOR = "#2a78d6"
ACTUAL_COLOR = "#eb6834"
GRID_COLOR = "#e4e3dc"
TEXT_COLOR = "#3d3d3a"


def to_float(value):
    return float(value) if value not in ("", None) else None


def relative_heading_deg(values, ref):
    """Heading relative to ref, unwrapped so it never jumps by 360°, in degrees."""
    out, prev = [], None
    for v in values:
        if v is None:
            out.append(None)
            continue
        rel = math.atan2(math.sin(v - ref), math.cos(v - ref))
        if prev is not None:
            rel += 2 * math.pi * round((prev - rel) / (2 * math.pi))
        prev = rel
        out.append(math.degrees(rel))
    return out


def load_runs(path):
    """Load a CSV, splitting it into separate runs when it contains several run_start values."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    runs = defaultdict(list)
    for r in rows:
        runs[r["run_start"]].append(r)
    return [load_run(run_rows) for run_rows in runs.values()]


def load_run(rows):
    t0 = float(rows[0]["timestamp"])
    d = {"t": [float(r["timestamp"]) - t0 for r in rows]}
    for name in ("expected_x", "expected_y", "expected_orientation", "expected_speed",
                 "vision_x", "vision_y", "vision_orientation", "vision_speed"):
        d[name] = [to_float(r[name]) for r in rows]

    # Convert to the robot's starting frame: origin at the expected start position,
    # starting heading pointing up (+y), matching the robot's convention that 0° is up.
    # In this frame the expected path points along the commanded angle. Expected and
    # actual share one transform so their error is unchanged.
    x0, y0, h0 = d["expected_x"][0], d["expected_y"][0], d["expected_orientation"][0]
    c, s = math.cos(h0), math.sin(h0)
    for prefix in ("expected", "vision"):
        xs, ys = d[f"{prefix}_x"], d[f"{prefix}_y"]
        forward = [c * (x - x0) + s * (y - y0) if x is not None else None for x, y in zip(xs, ys)]
        left = [-s * (x - x0) + c * (y - y0) if x is not None else None for x, y in zip(xs, ys)]
        d[f"{prefix}_x"] = [-v if v is not None else None for v in left]
        d[f"{prefix}_y"] = forward
        d[f"{prefix}_heading"] = relative_heading_deg(d[f"{prefix}_orientation"], h0)
    d["run_start"] = rows[0]["run_start"]
    d["n_actual"] = sum(v is not None for v in d["vision_x"])
    # Older runs command an instant step to full speed; newer ones ramp up over ~0.5 s.
    sp = d["expected_speed"]
    d["step_profile"] = sp[0] >= 0.1 or (len(sp) > 4 and sp[4] >= 0.4)
    return d


def base_layout(fig, title, x_title, y_title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        template="plotly_white",
        font=dict(color=TEXT_COLOR),
        height=480,
        margin=dict(l=70, r=20, t=60, b=60),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(title=y_title, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)


def last_point(x, y):
    for xv, yv in zip(reversed(x), reversed(y)):
        if xv is not None and yv is not None:
            return xv, yv
    return None


# A "line" is a dict: d (loaded data), prefix ("expected"/"vision"), name, color, dash,
# group (legend group), and optionally width, opacity, start_marker (default True).

def add_line(fig, x, y, line, hover, customdata=None):
    fig.add_trace(go.Scatter(
        x=x, y=y, name=line["name"], mode="lines", connectgaps=False,
        legendgroup=line["group"], customdata=customdata, opacity=line.get("opacity", 1),
        line=dict(color=line["color"], width=line.get("width", 2), dash=line["dash"]),
        hovertemplate=hover))


def xy_figure(lines):
    fig = go.Figure()
    hover = "x %{x:.3f} m<br>y %{y:.3f} m<br>t %{customdata:.2f} s<extra>%{fullData.name}</extra>"
    for line in lines:
        d, p = line["d"], line["prefix"]
        x, y = d[f"{p}_x"], d[f"{p}_y"]
        add_line(fig, x, y, line, hover, customdata=d["t"])
        # Mark where each path starts.
        first = next((i for i, v in enumerate(x) if v is not None), None)
        if line.get("start_marker", True) and first is not None:
            fig.add_trace(go.Scatter(
                x=[x[first]], y=[y[first]], mode="markers", showlegend=False,
                legendgroup=line["group"],
                marker=dict(size=10, color=line["color"], line=dict(color="white", width=2)),
                hovertemplate=f"{line['name']} start<br>x %{{x:.3f}} m<br>y %{{y:.3f}} m<extra></extra>"))
    base_layout(fig, "Position (robot start frame): expected vs actual",
                "x to the right of starting heading (m)", "y along starting heading (m)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def time_figure(lines, key, title, y_title, unit, unified):
    fig = go.Figure()
    hover = f"%{{y:.2f}}{unit}<extra>%{{fullData.name}}</extra>"
    for line in lines:
        d = line["d"]
        add_line(fig, d["t"], d[f"{line['prefix']}_{key}"], line, hover)
    base_layout(fig, title, "Time since start (s)", y_title)
    # With many overlaid trials a unified tooltip lists every line; "closest" names one.
    fig.update_layout(hovermode="x unified" if unified else "closest")
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1,
                     spikecolor="#9b9a92", spikedash="solid")
    return fig


def figures(lines, unified=True):
    return [
        xy_figure(lines),
        time_figure(lines, "heading", "Heading change from start: expected vs actual",
                    "Heading change (°)", "°", unified),
        time_figure(lines, "speed", "Speed over time: expected vs actual", "Speed", "", unified),
    ]


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="plotly.min.js"></script>
<style>
body {{ font-family: system-ui, sans-serif; color: {text}; background: #fff;
       max-width: 1000px; margin: 0 auto; padding: 24px 16px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 24px 0 8px; }}
.meta {{ color: #6b6a63; margin: 0 0 16px; }}
.note {{ background: #fdf3e7; border-left: 3px solid #eda100; padding: 8px 12px; margin: 0 0 16px; }}
a {{ color: {blue}; }}
</style></head><body>
{body}
</body></html>"""


def write_page(path, title, body):
    path.write_text(PAGE.format(title=html.escape(title), body=body,
                                text=TEXT_COLOR, blue=EXPECTED_COLOR), encoding="utf-8")


FRAME_NOTE = ('<p class="meta">Plots use the robot\'s starting frame: the start position is the '
              'origin and the starting heading points up (0°), so the expected path runs at '
              'the commanded angle. Heading is the change from the starting heading.</p>')


def plots_html(figs):
    return FRAME_NOTE + "".join(f.to_html(full_html=False, include_plotlyjs=False) for f in figs)


def missing_note(d):
    n = len(d["t"])
    if d["n_actual"] == 0:
        return "no actual (vision) data"
    if d["n_actual"] < n:
        return f"actual (vision) data missing for {n - d['n_actual']} of {n} samples"
    return ""


SERIES = ("expected_x", "expected_y", "expected_heading", "expected_speed",
          "vision_x", "vision_y", "vision_heading", "vision_speed")


def fill_gaps(values):
    """Linearly interpolate missing samples between two known ones; leave the ends empty."""
    known = [i for i, v in enumerate(values) if v is not None]
    out = list(values)
    for a, b in zip(known, known[1:]):
        for i in range(a + 1, b):
            out[i] = values[a] + (values[b] - values[a]) * (i - a) / (b - a)
    return out


MEAN_COVERAGE = 0.75  # plot the mean only where at least this share of trials have data


def mean_run(runs):
    """Average of several runs, sample by sample from each run's start (all log at ~20 Hz).

    Short vision dropouts inside a trial are interpolated first. The mean starts as a plain
    average, then advances by the average step of the trials present at both samples, so it
    stays continuous when a trial's data ends. It stops once fewer than MEAN_COVERAGE of
    the trials have data.
    """
    length = max(len(d["t"]) for d in runs)
    mean = {"t": [], "n_actual": 1}
    for i in range(length):
        ts = [d["t"][i] for d in runs if i < len(d["t"])]
        mean["t"].append(sum(ts) / len(ts))

    def at(f, i):
        return f[i] if i < len(f) else None

    for key in SERIES:
        filled = [fill_gaps(d[key]) for d in runs if any(v is not None for v in d[key])]
        need = max(2, math.ceil(MEAN_COVERAGE * len(filled)))
        col, current = [None] * length, None
        for i in range(length):
            present = [at(f, i) for f in filled if at(f, i) is not None]
            if len(present) < need:
                if current is not None:
                    break  # coverage has dropped for good; stop the mean here
                continue
            if current is None:
                current = sum(present) / len(present)
            else:
                steps = [at(f, i) - at(f, i - 1) for f in filled
                         if at(f, i) is not None and at(f, i - 1) is not None]
                current += sum(steps) / len(steps) if steps else 0
            col[i] = current
        mean[key] = col
    return mean


def moved(d):
    speeds = [v for v in d["vision_speed"] if v is not None]
    return bool(speeds) and max(speeds) >= STILL_SPEED


def overlay_lines(runs):
    """Every trial's actual path as a faint line, plus the mean actual and mean expected."""
    lines = []
    for label, d in runs:
        if d["n_actual"]:
            lines.append(dict(d=d, prefix="vision", name=f"Trial {label}", color=ACTUAL_COLOR,
                              dash="solid", group=f"trial{label}", width=1.5, opacity=0.35,
                              start_marker=False))
    # Runs where the robot never moved are drawn but left out of the mean.
    mean = mean_run([d for _, d in runs if moved(d)] or [d for _, d in runs])
    lines.append(dict(d=mean, prefix="vision", name="Mean actual", color=ACTUAL_COLOR,
                      dash="solid", group="mean", width=4))
    lines.append(dict(d=mean, prefix="expected", name="Expected (mean)", color=EXPECTED_COLOR,
                      dash="dash", group="expected", width=2.5))
    return lines


STEP_NOTE = ("commanded speed jumps straight to full speed (older step profile) instead of "
             "ramping up over ~0.5 s like the other runs")


def notes_html(runs):
    notes = [f"Trial {label}: {missing_note(d)}" for label, d in runs if missing_note(d)]
    html_out = ""
    if notes:
        html_out += ('<p class="note">' + "<br>".join(html.escape(n) for n in notes)
                     + "<br>Gaps are left unconnected.</p>")
    steps = [label for label, d in runs if d["step_profile"]]
    if steps:
        who = "This run's" if len(runs) == 1 else f"Trial{'s' * (len(steps) > 1)} {', '.join(steps)}:"
        html_out += f'<p class="note">{who} {STEP_NOTE}.</p>'
    return html_out


def extra_runs_note(n_runs):
    if n_runs <= 1:
        return ""
    return (f'<p class="note">This file contains {n_runs} runs; only the first is used, as '
            f'agreed. The other {n_runs - 1} are ignored.</p>')


def trial_page(out_dir, wheels, angle, trial, path, d, n_runs, overlay_name):
    title = f"Wheel angles {wheels} · commanded angle {angle}° · trial {trial}"
    lines = [
        dict(d=d, prefix="expected", name="Expected", color=EXPECTED_COLOR, dash="dash",
             group="expected"),
        dict(d=d, prefix="vision", name="Actual", color=ACTUAL_COLOR, dash="solid",
             group="actual"),
    ]
    body = (f'<p><a href="index.html">&larr; All tests</a> · '
            f'<a href="{overlay_name}">All trials at {angle}° ({wheels})</a></p>'
            f'<h1>{html.escape(title)}</h1>'
            f'<p class="meta">{html.escape(path.name)} · run start {html.escape(d["run_start"])}</p>'
            + extra_runs_note(n_runs) + notes_html([(str(trial), d)]) + plots_html(figures(lines)))
    out_name = f"angle_test_{wheels}_{angle}_{trial}.html"  # clean name even for "X (1).csv"
    write_page(out_dir / out_name, title, body)
    return out_name


def overlay_page(out_dir, wheels, angle, runs, links, overlay_name):
    """runs: list of (label, data); links: list of (trial number, trial page filename)."""
    title = f"Wheel angles {wheels} · commanded angle {angle}° · all trials"
    trial_links = " · ".join(f'<a href="{name}">Trial {t}</a>' for t, name in links)
    body = (f'<p><a href="index.html">&larr; All tests</a></p>'
            f'<h1>{html.escape(title)}</h1>'
            f'<p class="meta">Faint orange lines are individual trials ({len(runs)}); the bold '
            f'orange line is their mean (trials where the robot never moved are left out of it) '
            f'and the dashed blue line is the expected path. Hover a '
            f'line to see which trial it is. Click a legend entry to hide it; double-click to '
            f'show only that one. Individual trials: {trial_links}</p>'
            + notes_html(runs) + plots_html(figures(overlay_lines(runs), unified=False)))
    write_page(out_dir / overlay_name, title, body)


# One fixed colour per commanded direction, the same in every wheel configuration.
# Neighbouring directions take neighbouring palette slots (validated for colour-blind
# separation); any other angle is drawn in neutral grey rather than a made-up hue.
DIRECTION_COLORS = {0: "#2a78d6", 45: "#eb6834", 90: "#1baf7a", 135: "#eda100",
                    180: "#e87ba4", 225: "#008300", 270: "#6250d6", 315: "#e34948"}
OTHER_DIRECTION_COLOR = "#8a8980"


def config_page(out_dir, wheels, by_angle, page_name):
    """by_angle: list of (angle, runs, overlay page name) for one wheel configuration."""
    fig = go.Figure()
    for angle, runs, _ in by_angle:
        color = DIRECTION_COLORS.get(angle % 360, OTHER_DIRECTION_COLOR)
        group = f"dir{angle}"
        for label, d in runs:
            if d["n_actual"]:
                fig.add_trace(go.Scatter(
                    x=d["vision_x"], y=d["vision_y"], mode="lines", connectgaps=False,
                    name=f"{angle}° trial {label}", legendgroup=group, showlegend=False,
                    opacity=0.3, line=dict(color=color, width=1.2),
                    hovertemplate=f"{angle}° trial {label}<br>x %{{x:.2f}} m<br>y %{{y:.2f}} m<extra></extra>"))
        mean = mean_run([d for _, d in runs if moved(d)] or [d for _, d in runs])
        fig.add_trace(go.Scatter(
            x=mean["expected_x"], y=mean["expected_y"], mode="lines", legendgroup=group,
            showlegend=False, line=dict(color=color, width=2, dash="dash"),
            hovertemplate=f"{angle}° expected<br>x %{{x:.2f}} m<br>y %{{y:.2f}} m<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=mean["vision_x"], y=mean["vision_y"], mode="lines", name=f"{angle}°",
            legendgroup=group, line=dict(color=color, width=4),
            hovertemplate=f"{angle}° mean actual<br>x %{{x:.2f}} m<br>y %{{y:.2f}} m<extra></extra>"))
        # Label the end of the expected path with its direction (identity is never colour alone).
        end = last_point(mean["expected_x"], mean["expected_y"])
        if end:
            reach = math.hypot(*end) or 1
            fig.add_trace(go.Scatter(
                x=[end[0] * (1 + 0.12 / reach)], y=[end[1] * (1 + 0.12 / reach)], mode="text",
                text=[f"{angle}°"], textfont=dict(color=TEXT_COLOR, size=13),
                legendgroup=group, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(size=10, color=TEXT_COLOR, line=dict(color="white", width=2))))
    base_layout(fig, f"Wheel angles {wheels}: every trial, every direction",
                "x to the right of starting heading (m)", "y along starting heading (m)")
    fig.update_layout(height=760, legend=dict(title=dict(text="Commanded")))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    overlay_links = " · ".join(f'<a href="{name}">{angle}°</a>' for angle, _, name in by_angle)
    n_trials = sum(len(runs) for _, runs, _ in by_angle)
    title = f"Wheel angles {wheels} · all directions"
    body = (f'<p><a href="index.html">&larr; All tests</a></p><h1>{html.escape(title)}</h1>'
            f'<p class="meta">All {n_trials} trials, coloured by commanded direction. Faint lines '
            f'are individual trials, bold lines are the mean for each direction, dashed lines are '
            f'the expected paths (labelled with their direction). Every run starts at the black dot, '
            f'facing up. Click a direction in the legend to hide it; double-click to show only that '
            f'one. Per-direction overlays: {overlay_links}</p>'
            + FRAME_NOTE + fig.to_html(full_html=False, include_plotlyjs=False))
    write_page(out_dir / page_name, title, body)


STILL_SPEED = 0.05  # a run whose actual speed never exceeds this is treated as "never moved"


def distance_to_path(px, py, path):
    """Shortest distance from a point to a polyline given as [(x, y), ...]."""
    if len(path) == 1:
        return math.dist((px, py), path[0])
    best = math.inf
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        dx, dy = x2 - x1, y2 - y1
        seg_sq = dx * dx + dy * dy
        t = 0.0 if seg_sq == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_sq))
        best = min(best, math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))
    return best


def wrap_deg(angle):
    return (angle + 180) % 360 - 180


def run_errors(d):
    """Error measures for one run, or a reason string when it can't be scored.

    Runs with no vision data, or where the robot never moved (e.g. disabled), are
    excluded since they don't measure tracking.
    """
    if d["n_actual"] == 0:
        return "no vision data"
    if max(v for v in d["vision_speed"] if v is not None) < STILL_SPEED:
        return "robot never moved"
    path = list(zip(d["expected_x"], d["expected_y"]))
    pos_sq, head_sq, xt_sq = [], [], []
    last = None
    for i, (ex, ey, ax, ay, eh, ah) in enumerate(zip(
            d["expected_x"], d["expected_y"], d["vision_x"], d["vision_y"],
            d["expected_heading"], d["vision_heading"])):
        if ax is None:
            continue
        pos_sq.append((ax - ex) ** 2 + (ay - ey) ** 2)
        head_sq.append(wrap_deg(ah - eh) ** 2)
        xt_sq.append(distance_to_path(ax, ay, path) ** 2)
        last = i
    # Final errors only count if vision saw the robot come to rest after the commanded stop.
    stop_t = max(t for t, s in zip(d["t"], d["expected_speed"]) if s > 0)
    final_pos = final_head = None
    if d["t"][last] > stop_t and d["vision_speed"][last] < STILL_SPEED:
        final_pos = math.dist((d["vision_x"][last], d["vision_y"][last]),
                              (d["expected_x"][-1], d["expected_y"][-1]))
        final_head = abs(wrap_deg(d["vision_heading"][last] - d["expected_heading"][-1]))
    return dict(n=len(pos_sq), pos=pos_sq, head=head_sq, xt=xt_sq, step=d["step_profile"],
                final_pos=final_pos, final_head=final_head)


def rms(squares):
    return math.sqrt(sum(squares) / len(squares))


def variance(values):
    """Sample variance (n - 1); None with fewer than two values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def fmt(value, digits):
    return "—" if value is None else f"{value:.{digits}f}"


def sigma_band(value, mean, std):
    """How many σ a trial's value sits from its angle's mean, e.g. '+1.3σ (within 2σ)'."""
    if value is None or not std:
        return "—"
    z = (value - mean) / std
    z = 0.0 if abs(z) < 0.05 else z  # avoid printing "-0.0σ"
    band = max(1, math.ceil(abs(z)))
    return f"{z:+.1f}σ (within {band}σ)"


# (key, column label, unit, decimals, kind). "rms" metrics hold per-sample squared errors
# and the per-angle value pools every sample; "final" metrics are one number per trial and
# the per-angle value is the mean across trials.
TIMED_METRICS = [("pos", "Position RMS", "m", 3, "rms"), ("head", "Heading RMS", "°", 1, "rms")]
PATH_METRICS = [("xt", "Cross-track RMS", "m", 3, "rms"),
                ("final_pos", "Final position error", "m", 3, "final"),
                ("final_head", "Final heading error", "°", 1, "final")]


def metric_stats(scored, key, kind):
    """Per-trial values plus the per-angle headline, variance and σ for one metric."""
    if kind == "rms":
        trial_vals = [rms(e[key]) for _, e in scored]
        pooled = [v for _, e in scored for v in e[key]]
        headline = rms(pooled) if pooled else None
    else:
        trial_vals = [e[key] for _, e in scored]
        headline = None
    present = [v for v in trial_vals if v is not None]
    mean = sum(present) / len(present) if present else None
    if kind == "final":
        headline = mean
    var = variance(present)
    return dict(trial_vals=trial_vals, headline=headline, var=var, mean=mean,
                std=math.sqrt(var) if var is not None else None)


def sq_unit(unit):
    return "m²" if unit == "m" else "°²"


def angle_table(full, metrics):
    head1 = ('<tr><th rowspan="2">Wheel angles</th><th rowspan="2">Commanded</th>'
             '<th rowspan="2">Trials</th>')
    head2 = "<tr>"
    for _, label, unit, _, kind in metrics:
        head1 += f'<th colspan="3">{label if kind == "rms" else label + " (mean)"}</th>'
        head2 += f"<th>Value ({unit})</th><th>Variance ({sq_unit(unit)})</th><th>σ ({unit})</th>"
    rows = ""
    for wheels, angle, scored, excluded, ms in full:
        rows += f"<tr><td>{wheels}</td><td>{angle}°</td><td>{len(scored)}{'*' if excluded else ''}</td>"
        for key, _, _, digits, _ in metrics:
            m = ms[key]
            rows += (f"<td>{fmt(m['headline'], digits)}</td><td>{fmt(m['var'], digits + 1)}</td>"
                     f"<td>{fmt(m['std'], digits)}</td>")
        rows += "</tr>"
    return (f'<div class="scroll"><table><thead>{head1}</tr>{head2}</tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")


def trial_table(full, metrics):
    head = "<tr><th>Wheel angles</th><th>Commanded</th><th>Trial</th><th>Samples</th>"
    for _, label, unit, _, _ in metrics:
        head += f"<th>{label} ({unit})</th><th>σ</th>"
    rows = ""
    for wheels, angle, scored, _, ms in full:
        for i, (label, e) in enumerate(scored):
            rows += f"<tr><td>{wheels}</td><td>{angle}°</td><td>{label}</td><td>{e['n']}</td>"
            for key, _, _, digits, _ in metrics:
                m = ms[key]
                v = m["trial_vals"][i]
                rows += f"<td>{fmt(v, digits)}</td><td>{sigma_band(v, m['mean'], m['std'])}</td>"
            rows += "</tr>"
    return (f'<div class="scroll"><table><thead>{head}</tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")


def summary_page(out_dir, stats):
    """stats: list of (wheels, angle, [(label, errors dict)], [excluded descriptions])."""
    full, csv_rows, excluded_notes, missing_final, step_trials = [], [], [], [], []
    for wheels, angle, scored, excluded in stats:
        ms = {key: metric_stats(scored, key, kind) for key, *_, kind in TIMED_METRICS + PATH_METRICS}
        full.append((wheels, angle, scored, excluded, ms))
        excluded_notes += [f"{wheels} {angle}° trial {e}" for e in excluded]
        missing_final += [f"{wheels} {angle}° trial {label}" for label, e in scored
                          if e["final_pos"] is None]
        step_trials += [f"{wheels} {angle}° trial {label}" for label, e in scored if e["step"]]
        row = dict(wheels=wheels, angle=angle, trials=len(scored))
        for key, *_ in TIMED_METRICS + PATH_METRICS:
            row.update({key: ms[key]["headline"], f"{key}_var": ms[key]["var"],
                        f"{key}_std": ms[key]["std"]})
        csv_rows.append(row)

    with open(out_dir / "rms_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    excluded_html = ('<p class="muted">* Excluded runs: '
                     + "; ".join(html.escape(e) for e in excluded_notes) + "</p>"
                     ) if excluded_notes else ""
    missing_html = ('<p class="muted">Final errors not available (vision lost the robot before '
                    'it stopped): ' + "; ".join(html.escape(m) for m in missing_final) + "</p>"
                    ) if missing_final else ""
    if step_trials:
        missing_html += ('<p class="muted">Included but using the older step speed profile '
                         '(instant jump to full speed, no ramp): '
                         + "; ".join(html.escape(s) for s in step_trials) + "</p>")
    sigma_note = ('<p class="meta">σ columns: how far each trial is from the average trial for '
                  'its commanded angle, in standard deviations. + means worse than average, '
                  '− means better. "Within 1σ" means typical; beyond 2σ means unusual.</p>')
    body = (
        '<p><a href="index.html">&larr; All tests</a></p><h1>Error summary</h1>'
        '<p class="meta">Variance and σ are across trials (of each trial\'s value) — how '
        'repeatable the result is. Runs with no vision data, or where the robot never moved '
        '(actual speed always below 0.05), are excluded. Also saved as '
        '<code>rms_summary.csv</code>.</p>'
        '<h2>Path shape (timing ignored)</h2>'
        '<p class="meta"><strong>Cross-track RMS:</strong> for every actual sample, the distance '
        'to the nearest point anywhere on the expected path — how far off the line the robot '
        'was, regardless of how late it got there. <strong>Final position / heading error:'
        '</strong> where the robot came to rest compared with the end of the expected path '
        '(heading as an absolute difference). Per-angle values: cross-track pools every sample; '
        'final errors are the mean across trials. <strong>Read cross-track together with final '
        'position error:</strong> a robot that barely moves stays near the start of the path, '
        'so it scores a low cross-track error but a large final error (e.g. 2540 at 90°/270°, '
        'where the robot mostly spins in place).</p>'
        + angle_table(full, PATH_METRICS) + excluded_html + missing_html +
        '<h2>Time-matched</h2>'
        '<p class="meta">Actual compared with expected at the same timestamp. Any delay in '
        'the robot getting up to speed shows up here as position error, even if it drives in '
        'the right direction. Heading error is the difference wrapped to '
        '±180°. Per-angle RMS pools every sample.</p>'
        + angle_table(full, TIMED_METRICS) +
        '<h2>Per trial — path shape</h2>' + sigma_note + trial_table(full, PATH_METRICS) +
        '<h2>Per trial — time-matched</h2>' + trial_table(full, TIMED_METRICS) +
        '<style>.scroll{overflow-x:auto} table{border-collapse:collapse;margin-bottom:16px}'
        'th,td{padding:6px 12px;border-bottom:1px solid #e4e3dc;text-align:right;white-space:nowrap}'
        'th{font-weight:600} .muted{color:#6b6a63;font-size:12px}</style>')
    write_page(out_dir / "summary.html", "Error summary", body)
    return csv_rows


def main():
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "docs")  # docs/ is served by GitHub Pages
    out_dir.mkdir(exist_ok=True)
    # Clear pages from previous runs so tests that are no longer in the data don't linger.
    for old in out_dir.glob("*.html"):
        old.unlink()
    (out_dir / "plotly.min.js").write_text(plotly.offline.get_plotlyjs(), encoding="utf-8")
    (out_dir / ".nojekyll").touch()  # tell GitHub Pages to serve the files as-is

    groups, skipped, seen = defaultdict(list), [], {}
    for path in sorted(data_dir.rglob("*.csv")):
        m = FILENAME_RE.match(path.name)
        if not m:
            skipped.append(path.name)
            continue
        wheels, angle, trial = m.group(1), int(m.group(2)), int(m.group(3))
        key = (wheels, angle, trial)
        if key in seen:
            skipped.append(f"{path.name} (same test as {seen[key].name})")
            continue
        seen[key] = path
        groups[(wheels, angle)].append((trial, path))

    index_rows, config_runs, stats = defaultdict(list), defaultdict(list), []
    n_files = n_multi = 0
    for (wheels, angle) in sorted(groups):
        overlay_name = f"angle_test_{wheels}_{angle}_all_trials.html"
        runs, links = [], []
        scored, excluded = [], []
        for trial, path in sorted(groups[(wheels, angle)]):
            loaded = load_runs(path)
            d = loaded[0]  # only the first run in each file counts
            n_multi += len(loaded) > 1
            name = trial_page(out_dir, wheels, angle, trial, path, d, len(loaded), overlay_name)
            runs.append((str(trial), d))
            links.append((trial, name))
            n_files += 1
            errors = run_errors(d)
            if isinstance(errors, str):
                excluded.append(f"{trial} — {errors}")
            else:
                scored.append((str(trial), errors))
        stats.append((wheels, angle, scored, excluded))
        overlay_page(out_dir, wheels, angle, runs, links, overlay_name)
        config_runs[wheels].append((angle, runs, overlay_name))
        trial_links = " ".join(f'<a href="{name}">{t}</a>' for t, name in links)
        index_rows[wheels].append(
            f'<tr><td>{angle}°</td><td><a href="{overlay_name}">All {len(links)} trials</a></td>'
            f'<td class="trials">{trial_links}</td></tr>')

    skipped_html = ""
    if skipped:
        skipped_html = ("<h2>Skipped (filename not in expected format)</h2><ul>"
                        + "".join(f"<li>{html.escape(s)}</li>" for s in skipped) + "</ul>")
    for wheels, by_angle in config_runs.items():
        config_page(out_dir, wheels, by_angle, f"angle_test_{wheels}_all_directions.html")
    sections = "".join(
        f'<h2>Wheel angles {wheels}</h2><p><a href="angle_test_{wheels}_all_directions.html">'
        f'<strong>All directions on one plot</strong></a></p>'
        f'<div class="scroll"><table><thead><tr><th>Commanded</th>'
        f'<th>Overlay</th><th>Individual trials</th></tr></thead><tbody>{"".join(rows)}'
        f'</tbody></table></div>' for wheels, rows in sorted(index_rows.items()))
    rows = summary_page(out_dir, stats)
    write_page(out_dir / "index.html", "Drivetrain angle tests",
               '<h1>Drivetrain angle tests</h1><p><a href="summary.html"><strong>Error summary'
               '</strong></a></p>' + sections + skipped_html +
               '<style>.scroll{overflow-x:auto} table{border-collapse:collapse}'
               'th,td{padding:6px 12px;border-bottom:1px solid #e4e3dc;text-align:left}'
               'td.trials a{margin-right:8px}</style>')
    print(f"{n_files} trial pages ({n_multi} files had extra runs, first run used), "
          f"{len(groups)} overlay pages, {len(skipped)} skipped: {skipped}\n")
    metrics = TIMED_METRICS + PATH_METRICS
    print(f"{'wheels':>6} {'cmd':>4} {'n':>2} | " + " | ".join(
        f"{label} ({unit}) ±σ".rjust(26) for _, label, unit, _, _ in metrics))
    for r in rows:
        print(f"{r['wheels']:>6} {r['angle']:>4} {r['trials']:>2} | " + " | ".join(
            f"{fmt(r[key], digits)} ± {fmt(r[key + '_std'], digits)}".rjust(26)
            for key, _, _, digits, _ in metrics))


if __name__ == "__main__":
    main()
