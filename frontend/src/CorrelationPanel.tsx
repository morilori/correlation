import { useMemo, useState } from 'react';

interface Props {
  words: string[];
  effort: number[];
  backendUrl?: string;
  wordColumn?: string;
  group?: Record<string,string>;
}

type CorrelationResult = {
  matched_count: number;
  total_words: number;
  metric_correlations: Record<string, number>;
  used_columns: string[];
  token_column?: string;
};

export default function CorrelationPanel({ words, effort, backendUrl = 'http://localhost:8000', wordColumn, group }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CorrelationResult | null>(null);

  const disabled = !words.length || !effort.length;
  const top = useMemo(() => {
    if (!result) return [] as Array<[string, number]>;
    const entries = Object.entries(result.metric_correlations);
    return entries.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 10);
  }, [result]);

  const run = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${backendUrl}/effort-correlation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ words, effort, word_column: wordColumn, group }),
      });
      if (!res.ok) throw new Error(`Backend error ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ margin: '1.5em 0' }}>
      <h3>Correlation with OneStop Eye-tracking</h3>
      <button onClick={run} disabled={disabled || loading}>
        {loading ? 'Computing…' : 'Compute Correlations'}
      </button>
      {error && <div style={{ color: 'var(--color-danger)' }}>{error}</div>}
      {result && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: '.9em', color: '#aaa' }}>
            Matched {result.matched_count} of {result.total_words} words
            {result.token_column ? ` • using column: ${result.token_column}` : ''}
          </div>
          <table style={{ margin: '8px auto 0', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ padding: '0.5em', borderBottom: '1px solid #ccc' }}>Metric</th>
                <th style={{ padding: '0.5em', borderBottom: '1px solid #ccc' }}>r</th>
              </tr>
            </thead>
            <tbody>
              {top.map(([k, v]) => (
                <tr key={k}>
                  <td style={{ padding: '0.4em 0.6em', borderBottom: '1px solid #eee' }}>{k}</td>
                  <td style={{ padding: '0.4em 0.6em', borderBottom: '1px solid #eee' }}>{v.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
