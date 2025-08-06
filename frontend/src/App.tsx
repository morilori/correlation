import { useState, useEffect } from 'react';
import './App.css';
import AttentionHeatmap from './AttentionHeatmap';
import ProbabilityScoreboard from './ProbabilityScoreboard';

// Helper to check if a word is punctuation
function isPunctuation(word: string) {
  // Matches any word that is only punctuation (unicode aware)
  return /^[\p{P}\p{S}]+$/u.test(word);
}

function App() {
  const [input, setInput] = useState('')
  const [attentionData, setAttentionData] = useState<{words: string[], attention: number[][]} | null>(null);
  const [attentionError, setAttentionError] = useState<string | null>(null);
  const [selectedTokenIndices, setSelectedTokenIndices] = useState<number[]>([]); // multi-token selection
  const [unknownTokenIndices, setUnknownTokenIndices] = useState<number[]>([]); // indices of words marked as unknown
  const [knownTokenIndices, setKnownTokenIndices] = useState<number[]>([]); // indices of words marked as known
  const [sentencesPerGroup, setSentencesPerGroup] = useState(1);
  const [currentGroupIndex, setCurrentGroupIndex] = useState(0);
  const [allGroupsData, setAllGroupsData] = useState<{words: string[], attention: number[][]}[]>([]);

  // Probability scoreboard state
  const [probabilityWords, setProbabilityWords] = useState<string[]>([]);
  const [probabilities, setProbabilities] = useState<number[]>([]);
  const [originalProbabilities, setOriginalProbabilities] = useState<number[]>([]);
  const [showScoreboard, setShowScoreboard] = useState(false);
  // Fetch prediction probabilities when scoreboard is shown or known/unknown indices change
  useEffect(() => {
    if (!showScoreboard || !input.trim()) return;
    const fetchProbabilities = async () => {
      try {
        // Fetch changed probabilities (with known/unknown)
        const res = await fetch('http://localhost:8000/prediction-probabilities', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: input,
            known_indices: knownTokenIndices,
            unknown_indices: unknownTokenIndices,
          }),
        });
        if (!res.ok) throw new Error('Failed to fetch probabilities');
        const data = await res.json();
        setProbabilityWords(data.tokens);
        setProbabilities(data.probabilities);
        // Debug: print tokens, unknown indices, and probabilities
        console.log('BERT tokens:', data.tokens);
        console.log('Unknown indices sent:', knownTokenIndices, unknownTokenIndices);
        console.log('Probabilities:', data.probabilities);

        // Fetch original probabilities (no unknowns)
        const resOrig = await fetch('http://localhost:8000/prediction-probabilities', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: input,
            known_indices: [],
            unknown_indices: [],
          }),
        });
        if (!resOrig.ok) throw new Error('Failed to fetch original probabilities');
        const dataOrig = await resOrig.json();
        setOriginalProbabilities(dataOrig.probabilities);
        // Debug: print original probabilities
        console.log('Original Probabilities:', dataOrig.probabilities);
      } catch (e) {
        setProbabilityWords([]);
        setProbabilities([]);
        setOriginalProbabilities([]);
      }
    };
    fetchProbabilities();
  }, [showScoreboard, input, knownTokenIndices, unknownTokenIndices]);




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

  // Attention data for display (always includes all words including punctuation for display, but punctuation is excluded from calculations)
  let displayWords = attentionData?.words || [];
  let displayAttention = attentionData?.attention || [];
  
  // Self-attention (diagonal) is always included in score calculation
  // Immediate neighbors are always included in score calculation
  // Word pieces are always combined into complete words
  
  // Process display data (always includes all words)
  if (displayWords.length && displayAttention.length) {
    const groups: number[][] = [];
    let current: number[] = [];
    for (let i = 0; i < displayWords.length; ++i) {
      const w = displayWords[i];
      if (w.startsWith('##')) {
        current.push(i);
      } else {
        if (current.length) groups.push(current);
        current = [i];
      }
    }
    if (current.length) groups.push(current);
    displayWords = groups.map(g => g.map(i => displayWords[i]).join('').replace(/^##/, ''));
    displayAttention = groups.map(g1 =>
      groups.map(g2 => {
        let sum = 0, count = 0;
        for (const i of g1) for (const j of g2) {
          sum += displayAttention[i][j];
          count++;
        }
        return count ? sum / count : 0;
      })
    );
    
    // Always exclude punctuation from calculations - zero out attention to/from punctuation words
    displayAttention = displayAttention.map((row, i) =>
      row.map((val, j) => {
        const isPuncI = isPunctuation(displayWords[i]);
        const isPuncJ = isPunctuation(displayWords[j]);
        // Zero out if either word is punctuation
        return (isPuncI || isPuncJ) ? 0 : val;
      })
    );
  }

  return (
    <div>
      <h1>Understany</h1>
      <div style={{marginBottom: 8}}>
        <label style={{marginLeft: 16}}>
          Sentences per group:
          <input
            type="number"
            min={1}
            max={10}
            value={sentencesPerGroup}
            onChange={e => setSentencesPerGroup(Math.max(1, Number(e.target.value)))}
            style={{width: 50, marginLeft: 6, padding: '1px 3px'}}
          />
        </label>
      </div>
      <form style={{marginBottom: 16, width: '100%'}}>
        <textarea 
          value={input} 
          onChange={e => setInput(e.target.value)} 
          rows={4} 
          cols={60} 
          placeholder="Enter text..."
          style={{width: '100%', fontFamily: 'monospace', boxSizing: 'border-box', minHeight: 80, resize: 'vertical'}}
        />
        {input.trim() && getSentenceGroups(input, sentencesPerGroup).length > 1 && (
          <div style={{marginTop: 2, fontSize: '0.8em', color: '#666'}}>
            {getSentenceGroups(input, sentencesPerGroup).length} groups
          </div>
        )}
      </form>
      <button style={{ margin: '1em' }} onClick={() => setShowScoreboard(s => !s)}>
        {showScoreboard ? 'Hide' : 'Show'} Probability Scoreboard
      </button>
      {showScoreboard && (
        <ProbabilityScoreboard words={probabilityWords} probabilities={probabilities} originalProbabilities={originalProbabilities} />
      )}
      {input.trim() && (
        <div style={{margin: '16px 0'}}>
          {getSentenceGroups(input, sentencesPerGroup).length > 1 && input.trim() && allGroupsData.length > 1 && (
            <div style={{ marginBottom: 8 }}>
              <button 
                onClick={() => setCurrentGroupIndex(Math.max(0, currentGroupIndex - 1))}
                disabled={currentGroupIndex === 0}
              >
                ← Previous
              </button>
              <span>
                Group {currentGroupIndex + 1} of {allGroupsData.length}
              </span>
              <button 
                onClick={() => setCurrentGroupIndex(Math.min(allGroupsData.length - 1, currentGroupIndex + 1))}
                disabled={currentGroupIndex === allGroupsData.length - 1}
              >
                Next →
              </button>
            </div>
          )}
          {displayWords.length > 0 && displayAttention.length > 0 ? (
            <>
              <AttentionHeatmap 
                words={displayWords} 
                attention={displayAttention} 
                selectedTokenIndices={selectedTokenIndices}
                setSelectedTokenIndices={setSelectedTokenIndices}
                unknownTokenIndices={unknownTokenIndices}
                knownTokenIndices={knownTokenIndices}
                setUnknownTokenIndices={setUnknownTokenIndices}
                setKnownTokenIndices={setKnownTokenIndices}
                punctuationIndices={displayWords.map((word, i) => isPunctuation(word) ? i : -1).filter(i => i !== -1)}
                probabilities={probabilities}
              />
            </>
          ) : attentionError ? (
            <div style={{color: '#00AAFF', fontSize: '0.9em'}}>{attentionError}</div>
          ) : (
            <div style={{color: '#888', fontSize: '0.9em'}}>No data</div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
