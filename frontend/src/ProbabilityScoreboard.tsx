import React from 'react';


interface ProbabilityScoreboardProps {
  words: string[];
  probabilities: number[];
  originalProbabilities?: number[]; // Optional: original probabilities for comparison
}


const ProbabilityScoreboard: React.FC<ProbabilityScoreboardProps> = ({ words, probabilities, originalProbabilities }) => {
  if (!words.length || !probabilities.length) return null;

  return (
    <div style={{ margin: '2em 0' }}>
      <h3>Prediction Probability Scoreboard</h3>
      <table style={{ margin: '0 auto', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ padding: '0.5em', borderBottom: '1px solid #ccc' }}>Word</th>
            {originalProbabilities ? (
              <>
                <th style={{ padding: '0.5em', borderBottom: '1px solid #ccc' }}>Original</th>
                <th style={{ padding: '0.5em', borderBottom: '1px solid #ccc' }}>Changed</th>
                <th style={{ padding: '0.5em', borderBottom: '1px solid #ccc' }}>Δ</th>
              </>
            ) : (
              <th style={{ padding: '0.5em', borderBottom: '1px solid #ccc' }}>Probability</th>
            )}
          </tr>
        </thead>
        <tbody>
          {words.map((word, i) => (
            <tr key={i}>
              <td style={{ padding: '0.5em', borderBottom: '1px solid #eee' }}>{word}</td>
              {originalProbabilities ? (
                <>
                  <td style={{ padding: '0.5em', borderBottom: '1px solid #eee' }}>{(originalProbabilities[i] * 100).toFixed(2)}%</td>
                  <td style={{ padding: '0.5em', borderBottom: '1px solid #eee' }}>{(probabilities[i] * 100).toFixed(2)}%</td>
                  <td style={{ padding: '0.5em', borderBottom: '1px solid #eee' }}>
                    {((probabilities[i] - originalProbabilities[i]) * 100).toFixed(2)}%
                  </td>
                </>
              ) : (
                <td style={{ padding: '0.5em', borderBottom: '1px solid #eee' }}>{(probabilities[i] * 100).toFixed(2)}%</td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ProbabilityScoreboard;
