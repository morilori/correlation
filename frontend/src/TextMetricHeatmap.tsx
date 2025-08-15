import { useEffect, useMemo, useState } from 'react';

type SeriesMap = Record<string, (number | null)[]>;

interface Props {
  words: string[];
  effort: number[];
  backendUrl?: string;
  wordColumn?: string;
  group?: Record<string, string>;
}

type MetricsListResponse = { metrics: string[] };
type AlignedSeriesResponse = {
  matched_count: number;
  total_words: number;
  series: Record<string, Array<number | null>>;
  used_columns: string[];
  token_column?: string;
};

export default function TextMetricHeatmap({ words, effort, backendUrl = 'http://localhost:8000', wordColumn, group }: Props) {
  const [availableMetrics, setAvailableMetrics] = useState<string[]>([]);
  const [seriesByMetric, setSeriesByMetric] = useState<SeriesMap>({});
  const [selected, setSelected] = useState<string>('Effort');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // punctuation indices
  const punctuationIndices = useMemo(() => words.map((w, i) => (/^[\p{P}\p{S}]+$/u.test(w) ? i : -1)).filter(i => i !== -1), [words]);

  // Fetch metrics list once
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${backendUrl}/onestop/metrics`);
        if (!res.ok) throw new Error(`Failed to load metrics (${res.status})`);
        const data: MetricsListResponse = await res.json();
        if (!cancelled) setAvailableMetrics(data.metrics || []);
      } catch (e) {
        if (!cancelled) setAvailableMetrics([]);
      }
    })();
    return () => { cancelled = true; };
  }, [backendUrl]);

  // Fetch aligned series for all metrics whenever words/metadata change
  useEffect(() => {
    if (!words.length || !availableMetrics.length) { setSeriesByMetric({}); return; }
    let cancelled = false;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const res = await fetch(`${backendUrl}/onestop/aligned-series`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ words, metrics: availableMetrics, word_column: wordColumn, group })
        });
        if (!res.ok) throw new Error(`Failed to load aligned series (${res.status})`);
        const data: AlignedSeriesResponse = await res.json();
        if (!cancelled) setSeriesByMetric(data.series || {});
      } catch (e) {
        if (!cancelled) {
          setSeriesByMetric({});
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [backendUrl, words, availableMetrics, wordColumn, group]);

  // Build current value array for coloring
  const values: Array<number | null> = useMemo(() => {
    if (selected === 'Effort') {
      const n = Math.min(words.length, effort.length);
      return words.map((_, i) => (i < n ? effort[i] : null));
    }
    const s = seriesByMetric[selected];
    if (!s || s.length !== words.length) return words.map(() => null);
    return s;
  }, [selected, words, effort, seriesByMetric]);

  // Compute color scale on finite, non-punctuation values
  const colorFn = useMemo(() => {
    const vals = values.filter((v, i) => v != null && !Number.isNaN(v) && !punctuationIndices.includes(i)) as number[];
    const min = vals.length ? Math.min(...vals) : 0;
    const max = vals.length ? Math.max(...vals) : 1;
    return (v: number | null, i: number) => {
      if (v == null || Number.isNaN(v) || punctuationIndices.includes(i)) return 'transparent';
      const t = (v - min) / (max - min + 1e-8);
      const r = Math.round(255 - (255 - 0) * t);
      const g = Math.round(255 - (255 - 170) * t);
      const b = 255;
      return `rgb(${r},${g},${b})`;
    };
  }, [values, punctuationIndices]);

  // Metric options
  const options = useMemo(() => ['Effort', ...availableMetrics], [availableMetrics]);

  return (
    <div style={{ margin: '1.5em 0' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <h3 style={{ margin: 0 }}>Text visualization</h3>
        <label style={{ fontSize: 13 }}>
          Color by:
          <select value={selected} onChange={e => setSelected(e.target.value)} style={{ marginLeft: 6 }}>
            {options.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </label>
        {loading && <span style={{ fontSize: 12, color: '#888' }}>loading…</span>}
        {error && <span style={{ fontSize: 12, color: 'var(--color-danger)' }}>{error}</span>}
      </div>
      <div style={{ background: '#fff', padding: 16, borderRadius: 6, lineHeight: 1.8 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px 6px' }}>
          {words.map((w, i) => (
            <span
              key={i}
              title={`${w}${values[i] != null && !Number.isNaN(values[i] as number) ? ` • ${selected}: ${(values[i] as number).toFixed(3)}` : ''}`}
              style={{
                background: colorFn(values[i] as number | null, i),
                borderRadius: 4,
                padding: '2px 6px',
                border: '1px solid #e6f3ff'
              }}
            >
              {w}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
