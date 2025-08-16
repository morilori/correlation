import { useState } from 'react';

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

export default function CorrelationPanel({ words, effort, backendUrl = 'http://localhost:8000' }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CorrelationResult | null>(null);

  const disabled = !words.length || !effort.length;

  const run = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
  const res = await fetch(`${backendUrl}/effort-correlation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ words, effort }),
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
  <h3>Correlation with external dataset</h3>
      <button onClick={run} disabled={disabled || loading}>
        {loading ? 'Computing…' : 'Compute Correlations'}
      </button>
      {error && <div style={{ color: 'var(--color-danger)' }}>{error}</div>}
      {result && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: '.9em', color: '#aaa' }}>
            Matched {result.matched_count} / {result.total_words} words
            <br />
            Used columns: {result.used_columns.join(', ')}
          </div>
          <table style={{ margin: '1em auto', borderCollapse: 'collapse', minWidth: 320 }}>
            <thead>
              <tr>
                <th style={{ padding: '0.5em', borderBottom: '1px solid #ccc' }}>Metric</th>
                <th style={{ padding: '0.5em', borderBottom: '1px solid #ccc' }}>Correlation</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.metric_correlations).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([metric, corr]) => (
                <tr key={metric}>
                  <td style={{ padding: '0.5em', borderBottom: '1px solid #eee' }}>{metric}</td>
                  <td style={{ padding: '0.5em', borderBottom: '1px solid #eee' }}>{corr.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
