import { useMemo, useState, useEffect } from 'react';

declare const realData: any[];

type SeriesMap = Record<string, (number | null)[]>;

interface Props {
  words: string[];
  effort: number[];
  normSum?: number[];
  normReceived?: number[];
  normProvided?: number[];
  normRetrieved?: number[];
  normRecalled?: number[];
  backendUrl?: string;
  wordColumn?: string;
  group?: Record<string, string>;
}

export default function TextMetricHeatmap({ words, effort, normSum = [], normReceived = [], normProvided = [], normRetrieved = [], normRecalled = [] }: Props) {
  const [availableMetrics, setAvailableMetrics] = useState<string[]>([]);
  const [seriesByMetric, setSeriesByMetric] = useState<SeriesMap>({});
  const [selected, setSelected] = useState<string>('Effort');

  // punctuation indices
  const punctuationIndices = useMemo(() => words.map((w, i) => (/^[\p{P}\p{S}]+$/u.test(w) ? i : -1)).filter(i => i !== -1), [words]);

  // Load metrics from realData
  useEffect(() => {
    if (!words.length) return;
    // Find matching sentence in realData
    const match = realData.find((d: any) => {
      return d.words && d.words.length === words.length && d.words.every((w: any, i: number) => w.word === words[i]);
    });
    // Collect all metric keys from first word
    let metricKeys: string[] = [];
    let map: SeriesMap = {};
    if (match) {
      metricKeys = Object.keys(match.words[0]).filter(k => k !== 'word');
      for (const key of metricKeys) {
        map[key] = match.words.map((w: any) => typeof w[key] === 'number' ? w[key] : null);
      }
    }
    // Add computed metrics from props
    const computedMetrics = {
      normSum,
      normReceived,
      normProvided,
      normRetrieved,
      normRecalled,
      effort,
    };
    for (const [key, arr] of Object.entries(computedMetrics)) {
      if (arr.length === words.length) {
        if (!metricKeys.includes(key)) metricKeys.push(key);
        map[key] = arr;
      }
    }
    setAvailableMetrics(metricKeys);
    setSeriesByMetric(map);
  }, [words, effort, normSum, normReceived, normProvided, normRetrieved, normRecalled]);

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
            {options.map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </label>
      </div>
      {/* Render word heatmap */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', justifyContent: 'center' }}>
        {words.map((word, i) => (
          <span
            key={i}
            style={{
              background: colorFn(values[i], i),
              padding: '0.5em 0.7em',
              borderRadius: 6,
              margin: '2px 1px',
              border: punctuationIndices.includes(i) ? '1px dashed #aaa' : 'none',
              fontWeight: punctuationIndices.includes(i) ? 400 : 500,
              color: punctuationIndices.includes(i) ? '#888' : undefined,
            }}
            title={values[i] != null ? `${selected}: ${values[i]}` : undefined}
          >
            {word}
          </span>
        ))}
      </div>
    </div>
  );
}
