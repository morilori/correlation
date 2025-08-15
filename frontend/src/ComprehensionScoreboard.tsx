import type { CSSProperties } from 'react';

interface Props {
  words: string[];
  benefit: number[];
  provisionQuality: number[];
  effort: number[];
}

const cell: CSSProperties = { padding: '0.5em', borderBottom: '1px solid #eee' };
const head: CSSProperties = { padding: '0.5em', borderBottom: '1px solid #ccc' };

export const ComprehensionScoreboard = ({ words, benefit, provisionQuality, effort }: Props) => {
  const n = Math.min(words.length, benefit.length, provisionQuality.length, effort.length);
  if (!n) return null;

  const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => effort[a] - effort[b]);
  return (
    <div style={{ margin: '2em 0' }}>
  <h3>Comprehension Scores</h3>
      <table style={{ margin: '0 auto', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={head}>Word</th>
            <th style={head}>Integration</th>
            <th style={head}>Contribution</th>
            <th style={head}>Effort</th>
          </tr>
        </thead>
        <tbody>
          {indices.map((i) => (
            <tr key={i}>
              <td style={cell}>{words[i]}</td>
              <td style={cell}>{(benefit[i] * 100).toFixed(1)}%</td>
              <td style={cell}>{(provisionQuality[i] * 100).toFixed(1)}%</td>
              <td style={cell}>{(effort[i] * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ComprehensionScoreboard;
