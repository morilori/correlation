import { useState, useEffect } from 'react';
import './App.css'
import AttentionHeatmap from './AttentionHeatmap';

// Helper to check if a word is punctuation
function isPunctuation(word: string) {
  // Matches any word that is only punctuation (unicode aware)
  return /^[\p{P}\p{S}]+$/u.test(word);
}

function App() {
  const [input, setInput] = useState('')
  const [attentionData, setAttentionData] = useState<{words: string[], attention: number[][]} | null>(null);
  const [attentionError, setAttentionError] = useState<string | null>(null);
  const [includePunctuation, setIncludePunctuation] = useState(true);
  const [includeSelfAttention, setIncludeSelfAttention] = useState(true);
  const [ignoreImmediateNeighbors, setIgnoreImmediateNeighbors] = useState(false);
  const [combineWordPieces, setCombineWordPieces] = useState(false);
  const [selectedTokenIndices, setSelectedTokenIndices] = useState<number[]>([]); // multi-token selection
  const [unknownTokenIndices, setUnknownTokenIndices] = useState<number[]>([]); // indices of words marked as unknown
  const [knownTokenIndices, setKnownTokenIndices] = useState<number[]>([]); // indices of words marked as known
  const [sentencesPerGroup, setSentencesPerGroup] = useState(1);
  const [currentGroupIndex, setCurrentGroupIndex] = useState(0);
  const [allGroupsData, setAllGroupsData] = useState<{words: string[], attention: number[][]}[]>([]);




  // Helper to group sentences for attention visualization
  const getSentenceGroups = (text: string, groupSize: number): string[] => {
    if (!text.trim()) return [];
    const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 0);
    if (sentences.length === 0) return [];
    const groups: string[] = [];
    for (let i = 0; i < sentences.length; i += groupSize) {
      const group = sentences.slice(i, i + groupSize).join(' ');
      groups.push(group);
    }
    return groups;
  };

  // Fetch attention data when input changes
  useEffect(() => {
    async function fetchAttention() {
      if (!input.trim()) {
        setAttentionData(null);
        setAllGroupsData([]);
        setAttentionError(null);
        return;
      }

      const groups = getSentenceGroups(input, sentencesPerGroup);
      if (groups.length === 0) {
        setAttentionData(null);
        setAllGroupsData([]);
        setAttentionError(null);
        return;
      }

      setAttentionError(null);
      const groupResults: {words: string[], attention: number[][]}[] = [];
      
      try {
        for (let i = 0; i < groups.length; i++) {
          const groupText = groups[i];
          const res = await fetch('http://localhost:8000/attention', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: groupText })
          });
          
          if (!res.ok) throw new Error(`Backend error for group ${i + 1}`);
          const data = await res.json();
          groupResults.push({
            words: data.tokens,
            attention: data.attention
          });
        }
        
        setAllGroupsData(groupResults);
        const currentGroup = groupResults[Math.min(currentGroupIndex, groupResults.length - 1)];
        if (currentGroup) {
          setAttentionData({ words: currentGroup.words, attention: currentGroup.attention });
        } else {
          setAttentionData(null);
        }
      } catch (error: unknown) {
        setAttentionError('Could not fetch attention: ' + (error instanceof Error ? error.message : String(error)));
        setAttentionData(null);
        setAllGroupsData([]);
      }
    }

    fetchAttention();
  }, [input, sentencesPerGroup, currentGroupIndex]);

  // Filtered attention data for display
  let filteredWords = attentionData?.words || [];
  let filteredAttention = attentionData?.attention || [];
  if (attentionData && !includePunctuation) {
    // Find indices of non-punctuation words
    const keepIdx = attentionData.words.map((w, i) => !isPunctuation(w) ? i : -1).filter(i => i !== -1);
    filteredWords = keepIdx.map(i => attentionData.words[i]);
    filteredAttention = keepIdx.map(i => keepIdx.map(j => attentionData.attention[i][j]));
  }
  if (!includeSelfAttention && filteredAttention.length) {
    filteredAttention = filteredAttention.map((row, i) => row.map((v, j) => i === j ? 0 : v));
  }
  if (ignoreImmediateNeighbors && filteredAttention.length) {
    filteredAttention = filteredAttention.map((row, i) =>
      row.map((v, j) => (j === i - 1 || j === i + 1) ? 0 : v)
    );
  }
  let displayWords = filteredWords;
  let displayAttention = filteredAttention;
  if (combineWordPieces && filteredWords.length && filteredAttention.length) {
    const groups: number[][] = [];
    let current: number[] = [];
    for (let i = 0; i < filteredWords.length; ++i) {
      const w = filteredWords[i];
      if (w.startsWith('##')) {
        current.push(i);
      } else {
        if (current.length) groups.push(current);
        current = [i];
      }
    }
    if (current.length) groups.push(current);
    displayWords = groups.map(g => g.map(i => filteredWords[i]).join('').replace(/^##/, ''));
    displayAttention = groups.map(g1 =>
      groups.map(g2 => {
        let sum = 0, count = 0;
        for (const i of g1) for (const j of g2) {
          sum += filteredAttention[i][j];
          count++;
        }
        return count ? sum / count : 0;
      })
    );
  }

  return (
    <div>
      <h1>Comprehension Graph</h1>
      <div style={{marginBottom: 12, textAlign: 'left'}}>
        <label>
          <input
            type="checkbox"
            checked={includePunctuation}
            onChange={e => setIncludePunctuation(e.target.checked)}
            style={{marginRight: 6}}
          />
          Include punctuation in calculations and display
        </label>
      </div>
      <div style={{marginBottom: 12, textAlign: 'left'}}>
        <label>
          <input
            type="checkbox"
            checked={includeSelfAttention}
            onChange={e => setIncludeSelfAttention(e.target.checked)}
            style={{marginRight: 6}}
          />
          Include self-attention (diagonal) in score calculation
        </label>
      </div>
      <div style={{marginBottom: 12, textAlign: 'left'}}>
        <label>
          <input
            type="checkbox"
            checked={ignoreImmediateNeighbors}
            onChange={e => setIgnoreImmediateNeighbors(e.target.checked)}
            style={{marginRight: 6}}
          />
          Ignore immediate left/right neighbors in score calculation
        </label>
      </div>
      <div style={{marginBottom: 12, textAlign: 'left'}}>
        <label>
          <input
            type="checkbox"
            checked={combineWordPieces}
            onChange={e => setCombineWordPieces(e.target.checked)}
            style={{marginRight: 6}}
          />
          Combine tokens that are part of the same word (mean attention)
        </label>
      </div>
      <div style={{marginBottom: 12, textAlign: 'left'}}>
        <label>
          Sentences per group:
          <input
            type="number"
            min={1}
            max={10}
            value={sentencesPerGroup}
            onChange={e => setSentencesPerGroup(Math.max(1, Number(e.target.value)))}
            style={{width: 60, marginLeft: 8, padding: '2px 4px'}}
          />
        </label>
        <span style={{marginLeft: 8, fontSize: '0.9em', color: '#666'}}>
          (Splits the text into groups of N sentences. Each group is processed and visualized separately, allowing any text length.)
        </span>
      </div>
      <form style={{marginBottom: 20, width: '100%'}}>
        <textarea 
          value={input} 
          onChange={e => setInput(e.target.value)} 
          rows={6} 
          cols={60} 
          placeholder="Enter your text here..."
          style={{width: '100%', fontFamily: 'monospace', boxSizing: 'border-box', minHeight: 120, resize: 'vertical'}}
        />
        {input.trim() && (
          <div style={{marginTop: 4, fontSize: '0.8em', color: '#666'}}>
            {(() => {
              const groups = getSentenceGroups(input, sentencesPerGroup);
              const totalTokens = groups.reduce((total, group) => total + group.split(/\s+/).length, 0);
              const maxGroupTokens = Math.max(...groups.map(group => group.split(/\s+/).length));
              return (
                <span style={{color: '#198754'}}>
                  {groups.length} group{groups.length > 1 ? 's' : ''} (~{totalTokens} tokens total, max {maxGroupTokens} per group)
                  {groups.length > 1 && ' - Navigate between groups in visualization below'}
                </span>
              );
            })()}
          </div>
        )}
      </form>
      {/* Attention Heatmap Visualization */}
      {/* Attention Heatmap Visualization */}
      {input.trim() && (
        <div style={{margin: '32px 0'}}>
          <h2>Word Attention Heatmap</h2>
          {getSentenceGroups(input, sentencesPerGroup).length > 1 && input.trim() && (
            <div style={{marginBottom: 16, padding: 12, background: '#f0f8ff', borderRadius: 6, textAlign: 'left'}}>
              <b>Processing groups ({sentencesPerGroup} sentence{sentencesPerGroup > 1 ? 's' : ''} per group):</b>
              <div style={{marginTop: 8}}>
                <div style={{marginBottom: 8, padding: 8, background: '#d4edda', borderRadius: 4, color: '#155724', fontSize: '0.9em'}}>
                  ✅ Processing {getSentenceGroups(input, sentencesPerGroup).length} groups individually - no matrix size limits!
                </div>
                
                {allGroupsData.length > 1 && (
                  <div style={{marginBottom: 12, padding: 8, background: '#e7f3ff', borderRadius: 4}}>
                    <div style={{display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6}}>
                      <span style={{fontWeight: 'bold', fontSize: '0.9em'}}>Navigate groups:</span>
                      <button 
                        onClick={() => setCurrentGroupIndex(Math.max(0, currentGroupIndex - 1))}
                        disabled={currentGroupIndex === 0}
                        style={{padding: '2px 8px', fontSize: '0.8em'}}
                      >
                        ← Previous
                      </button>
                      <span style={{fontSize: '0.9em', fontWeight: 'bold'}}>
                        Group {currentGroupIndex + 1} of {allGroupsData.length}
                      </span>
                      <button 
                        onClick={() => setCurrentGroupIndex(Math.min(allGroupsData.length - 1, currentGroupIndex + 1))}
                        disabled={currentGroupIndex === allGroupsData.length - 1}
                        style={{padding: '2px 8px', fontSize: '0.8em'}}
                      >
                        Next →
                      </button>
                    </div>
                    <div style={{fontSize: '0.8em', color: '#666'}}>
                      All groups are shown below. Currently active for interactions: "{getSentenceGroups(input, sentencesPerGroup)[currentGroupIndex]}"
                    </div>
                  </div>
                )}
                
                {getSentenceGroups(input, sentencesPerGroup).map((group, index) => (
                  <div key={index} style={{marginBottom: 8}}>
                    <span style={{fontWeight: 'bold', color: allGroupsData.length > 1 && index === currentGroupIndex ? '#0066cc' : '#666', fontSize: '0.9em'}}>
                      Group {index + 1}{allGroupsData.length > 1 && index === currentGroupIndex ? ' (current)' : ''}: 
                    </span>
                    <span style={{fontFamily: 'monospace', fontSize: '0.9em', color: '#555', marginLeft: 4}}>"{group}"</span>
                  </div>
                ))}
                <div style={{marginTop: 8, fontSize: '0.8em', color: '#666', fontStyle: 'italic'}}>
                  Each group is processed separately. All groups are displayed in sequence below. Navigation controls which group is active for word selection.
                </div>
              </div>
            </div>
          )}
          {/* Unknown word selection UI */}
          {displayWords.length > 0 && (
            <div style={{marginBottom: 10, textAlign: 'left'}}>
              <div style={{display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6}}>
                <b>Mark unknown words:</b>
                <button
                  onClick={() => setUnknownTokenIndices(displayWords.map((_, i) => i))}
                  style={{
                    padding: '4px 8px',
                    fontSize: '12px',
                    backgroundColor: '#ffeeee',
                    border: '1px solid #ffcccc',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Select All
                </button>
                <button
                  onClick={() => setUnknownTokenIndices([])}
                  style={{
                    padding: '4px 8px',
                    fontSize: '12px',
                    backgroundColor: '#f8f8f8',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Clear All
                </button>
              </div>
              <div style={{marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 8}}>
                {displayWords.map((w, i) => (
                  <label key={i} style={{display: 'inline-flex', alignItems: 'center', gap: 2, background: unknownTokenIndices.includes(i) ? '#ffe0e0' : 'transparent', borderRadius: 4, padding: '2px 6px'}}>
                    <input
                      type="checkbox"
                      checked={unknownTokenIndices.includes(i)}
                      onChange={() => {
                        setUnknownTokenIndices(unknownTokenIndices.includes(i)
                          ? unknownTokenIndices.filter(idx => idx !== i)
                          : [...unknownTokenIndices, i]);
                      }}
                      style={{marginRight: 3}}
                    />
                    <span style={{fontWeight: 500}}>{w}</span>
                  </label>
                ))}
              </div>
              <div style={{fontSize: 13, marginTop: 2, color: '#b00'}}>
                Unknown words will <b>only receive</b> attention, not provide it to others.
              </div>
            </div>
          )}
          {/* Known word selection UI */}
          {displayWords.length > 0 && (
            <div style={{marginBottom: 10, textAlign: 'left'}}>
              <div style={{display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6}}>
                <b>Mark known words:</b>
                <button
                  onClick={() => setKnownTokenIndices(displayWords.map((_, i) => i))}
                  style={{
                    padding: '4px 8px',
                    fontSize: '12px',
                    backgroundColor: '#eeffee',
                    border: '1px solid #ccffcc',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Select All
                </button>
                <button
                  onClick={() => setKnownTokenIndices([])}
                  style={{
                    padding: '4px 8px',
                    fontSize: '12px',
                    backgroundColor: '#f8f8f8',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Clear All
                </button>
              </div>
              <div style={{marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 8}}>
                {displayWords.map((w, i) => (
                  <label key={i} style={{display: 'inline-flex', alignItems: 'center', gap: 2, background: knownTokenIndices.includes(i) ? '#e0ffe0' : 'transparent', borderRadius: 4, padding: '2px 6px'}}>
                    <input
                      type="checkbox"
                      checked={knownTokenIndices.includes(i)}
                      onChange={() => {
                        setKnownTokenIndices(knownTokenIndices.includes(i)
                          ? knownTokenIndices.filter(idx => idx !== i)
                          : [...knownTokenIndices, i]);
                      }}
                      style={{marginRight: 3}}
                    />
                    <span style={{fontWeight: 500}}>{w}</span>
                  </label>
                ))}
              </div>
              <div style={{fontSize: 13, marginTop: 2, color: '#080'}}>
                Known words will have <b>norm received = 1</b> (max score for received attention).<br/>
                Words that provide attention to a known word will have their <b>norm provided</b> increased, proportional to the attention they give to the known word. This means that words contributing meaning to known words are also considered more known.
              </div>
            </div>
          )}
          {displayWords.length > 0 && displayAttention.length > 0 ? (
            <>
              <AttentionHeatmap 
                words={displayWords} 
                attention={displayAttention} 
                // setRewrittenText prop removed
                selectedTokenIndices={selectedTokenIndices}
                setSelectedTokenIndices={setSelectedTokenIndices}
                unknownTokenIndices={unknownTokenIndices}
                knownTokenIndices={knownTokenIndices}
              />
            </>
          ) : attentionError ? (
            <div style={{color: 'red'}}>{attentionError}</div>
          ) : (
            <div style={{color: '#888'}}>No attention data.</div>
          )}
        </div>
      )}
      {/* End of attention visualization UI */}
      {/* End of attention visualization UI */}
    </div>
  )
}

export default App
