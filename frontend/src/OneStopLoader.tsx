import { useEffect, useState } from 'react';

type ParagraphGroup = {
  group: Record<string, string>;
  size: number;
};

type ParagraphListResponse = {
  groups: ParagraphGroup[];
  word_column: string;
  word_candidates: string[];
  grouping_columns: string[];
  order_columns: string[];
};

interface Props {
  backendUrl?: string;
  onLoad: (text: string, meta?: { wordColumn?: string; group?: Record<string,string> }) => void;
}

export default function OneStopLoader({ backendUrl = 'http://localhost:8000', onLoad }: Props) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [list, setList] = useState<ParagraphListResponse | null>(null);
  const [selected, setSelected] = useState<number>(0);
  const [wordColumn, setWordColumn] = useState<string | null>(null);
  const [wordLimit, setWordLimit] = useState<string>('');

  useEffect(() => {
  if (!open || list) return;
    (async () => {
      setLoading(true);
      setError(null);
      try {
    const query = wordColumn ? `?word_column=${encodeURIComponent(wordColumn)}` : '';
    const res = await fetch(`${backendUrl}/onestop/paragraphs${query}`);
        if (!res.ok) throw new Error(`Backend error ${res.status}`);
        const data: ParagraphListResponse = await res.json();
        setList(data);
    setSelected(0);
    if (!wordColumn) setWordColumn(data.word_column);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, [open, list, backendUrl, wordColumn]);

  const loadParagraph = async () => {
    if (!list || list.groups.length === 0) return;
    const grp = list.groups[selected]?.group ?? {};
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${backendUrl}/onestop/paragraph`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ group: grp, word_column: wordColumn, limit: wordLimit ? Number(wordLimit) : undefined }),
      });
      if (!res.ok) throw new Error(`Backend error ${res.status}`);
      const data: { text: string } = await res.json();
      let text = data.text;
      const n = wordLimit ? Number(wordLimit) : undefined;
      if (n && n > 0) {
        const tokens = text.trim().split(/\s+/);
        if (tokens.length > n) text = tokens.slice(0, n).join(' ');
      }
  onLoad(text, { wordColumn: wordColumn ?? list.word_column, group: grp });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginBottom: 8 }}>
      <button onClick={() => setOpen(o => !o)}>
        {open ? 'Hide OneStop paragraphs' : 'Load OneStop paragraphs'}
      </button>
      {open && (
        <div style={{ marginTop: 8 }}>
          {loading && <div style={{ color: '#aaa' }}>Loading…</div>}
          {error && <div style={{ color: 'var(--color-danger)' }}>{error}</div>}
          {list && list.groups.length > 0 && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', justifyContent: 'center' }}>
              <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                Word column:
                <select
                  value={wordColumn ?? list.word_column}
                  onChange={e => {
                    setWordColumn(e.target.value);
                    // Force refetch of list with new column
                    setList(null);
                  }}
                >
                  {[list.word_column, ...(list.word_candidates || [])]
                    .filter((v, i, a) => a.indexOf(v) === i)
                    .map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                </select>
              </label>
              <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                Word limit:
                <input
                  type="number"
                  min={1}
                  placeholder="all"
                  value={wordLimit}
                  onChange={e => setWordLimit(e.target.value)}
                  style={{ width: 90 }}
                />
              </label>
              <select
                value={selected}
                onChange={e => setSelected(Number(e.target.value))}
                style={{ minWidth: 280 }}
              >
                {list.groups.map((g, i) => (
                  <option value={i} key={i}>
                    {Object.entries(g.group).map(([k, v]) => `${k}:${v}`).join(' • ')} — {g.size} tokens
                  </option>
                ))}
              </select>
              <button onClick={loadParagraph} disabled={loading}>Load paragraph into input</button>
            </div>
          )}
          {/* Preview removed */}
          {list && list.groups.length === 0 && (
            <div style={{ color: '#aaa' }}>No groups detected in dataset.</div>
          )}
        </div>
      )}
    </div>
  );
}
