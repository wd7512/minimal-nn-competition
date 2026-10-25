# Leaderboard design

The generated leaderboard is a static, self-contained page. It groups related
benchmarks instead of listing them as flat sections. The index stays compact:
each submission links to its own detail page, which carries the replay and the
full result metadata. Pass/Fail language and the version footer are preserved.

## Layout

**Benchmark Atlas** is the selected direction.

- A sticky left rail lists the benchmark groups and total submission counts.
- The main column starts with a short masthead, then one section per group.
- Each group has a numbered eyebrow and a large title.
- The group title cycles through versions when JavaScript is available. A
  version row underneath offers direct selection.
- The table is the primary content: neutral rules, generous spacing, a sticky
  rank column on narrow screens, and horizontal scrolling instead of compressed
  cells. It keeps rank, submission (with its score), parameters, and one
  benchmark-specific column: Game length, Error, or Reward.
- Submission names are links. The link opens a detail page in the same tab; the
  secondary line shows the raw submission id when it differs from the name.

The page is deliberately quiet. Colour marks group identity and status; it does
not fill surfaces or decorate every control.

## Groups and versions

| Group | Versions, in order | Default |
| --- | --- | --- |
| Snake | Simple 8×8, Simple 16×16, Obstacles 8×8 | Simple 8×8 |
| Image compression | MNIST, Fashion-MNIST | MNIST |
| Chess evaluation | Stockfish 10k, Stockfish 100k | Stockfish 10k |

Unknown benchmarks fall back to their own single-version group after the known
groups, so a new benchmark is never dropped from the page.

## Palette

Neutral paper and ink carry the layout. Each group owns one muted accent. Green
and rose are reserved for Pass/Fail.

| Token | Light | Dark | Use |
| --- | --- | --- | --- |
| Background | `#f4f6f4` | `#0d1513` | Page |
| Surface | `#fbfcfa` | `#131d1a` | Table header |
| Ink | `#17231f` | `#e7efeb` | Primary text |
| Muted | `#63706b` | `#9aaba4` | Secondary text |
| Rule | `#d9e0db` | `#273631` | Row and section rules |
| Rule strong | `#aebbb5` | `#40534c` | Major rules |
| Hover | `#edf2ef` | `#182622` | Row hover |
| Pass | `#1f6b43` | `#74cf9b` | Pass status |
| Fail | `#8a3434` | `#f09b91` | Fail status |
| Snake accent | `#256b66` | `#7fcac1` | Snake identity |
| MNIST accent | `#8a5b16` | `#e0b76b` | MNIST identity |
| Chess accent | `#3f5f86` | `#9bb9df` | Chess identity |

Accents appear only on the group number, the rail marker, the active version
underline, the cycle glyph, the rail and title hover underlines, and the three
small brand dots. They are not used as card backgrounds, gradients, or table
fills.

`color-scheme: light dark` follows the operating system. There is no manual
theme toggle.

## Typography

- Display: system serif stack (`ui-serif`, Georgia, Times New Roman).
- Body: system sans stack.
- Labels, ranks, counts, scores, and metadata: system monospace stack.
- Headings are tight and large; everything operational is small, uppercase, and
  tracked. No external fonts are loaded.

## Interaction

- **Without JavaScript:** every version panel is rendered and visible. The
  switcher row and cycle glyph stay hidden, so there are no dead controls.
- **With JavaScript:** only the active panel is shown. Version buttons update the
  panel, counts, `aria-pressed`, and the live version label. A hash matching a
  version name opens that version directly.
- **Title cycle:** the group title advances to the next version. It becomes a
  keyboard-operable control (`Enter` or `Space`) only after JavaScript enhances
  it. The accessible label names the current version and position.
- Version selection never reloads the page and never changes table ordering.

## Tables

The index table is intentionally small:

`#`, Name (with score and status), Parameters, and the benchmark column.

- Ranks restart at 1 for each benchmark version.
- Parameter counts use grouped digits.
- Submission links point at `submissions/{stem}.html`.
- Score formatting, near-100% handling, and Pass/Fail semantics are unchanged.
- Name, author, normalisation, game length, samples, tolerance, and replay move
  to the detail page instead of widening the table.

## Submission detail pages

Each result gets one page under `submissions/`, named with the same
`replay_stem(benchmark, submission)` used for replay files. Detail pages are
self-contained and reuse the atlas tokens.

- The header repeats group and version, the submission name, and a back link.
- A two-column body shows the replay on the left and metadata on the right;
  benchmarks without a replay use a single metadata column.
- Snake pages with a replay embed the canvas and controls and write the replay
  JSON to `games/{stem}.json`. Snake pages without a replay say "Not recorded".
- Metadata always includes submission id, name, author, rank, score, status,
  structure (for example `[784,14,14,10]`), and parameters, plus
  benchmark-specific fields when present: normalisation, reward, game length,
  seed game count, landings, and chess tolerance.
- Replay and detail links are relative and open in the same tab.
- Detail filenames use the same collision check as replays, so distinct
  submissions can never overwrite one another.

## Responsive behaviour

- Above 1024px: two-column shell with a sticky rail.
- 701–1024px: one column; the rail becomes a three-column index above the
  content.
- At 700px and below: the rail is a single list, group headers stack, and tables
  scroll horizontally with the rank column pinned.
- The table wrapper is keyboard-focusable and labelled.

## Accessibility

- Sections use `aria-labelledby`; version buttons use `aria-controls` and
  `aria-pressed`.
- The current version is announced through `aria-live="polite"`.
- The cycle trigger is a real keyboard target after enhancement and has a
  descriptive label.
- Status is conveyed by text as well as colour.
- Focus rings use the accent-neutral `--focus` token.
- `prefers-reduced-motion` disables transitions.

## Determinism

`render_html` emits a fixed document for a given ranked result list and replay
map. There are no timestamps, random values, external assets, or runtime
dependencies. Group order, version order, row order, detail-page order, and the
inline CSS/JS are all fixed. `render_site` remains the only writer; it replaces
`submissions/` and `games/` wholesale so stale pages cannot linger. `docs/` is a
build artifact.
