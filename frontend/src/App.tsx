import React, { useState, useEffect } from 'react';
import './App.css';
import AttentionHeatmap from './AttentionHeatmap';

// Helper to check if a word is punctuation
function isPunctuation(word: string) {
  // Matches any word that is only punctuation (unicode aware)
  return /^[\p{P}\p{S}]+$/u.test(word);
}

function App() {
  const [input, setInput] = useState('')
  const [attentionData, setAttentionData] = useState<{words: string[], attention: number[][], ffnActivations?: number[], probabilities?: number[]} | null>(null);
  const [attentionError, setAttentionError] = useState<string | null>(null);
  const [selectedTokenIndices, setSelectedTokenIndices] = useState<number[]>([]); // multi-token selection
  const [unknownTokenIndices, setUnknownTokenIndices] = useState<number[]>([]); // indices of words marked as unknown
  const [knownTokenIndices, setKnownTokenIndices] = useState<number[]>([]); // indices of words marked as known
  const [sentencesPerGroup, setSentencesPerGroup] = useState(1);
  const [currentGroupIndex, setCurrentGroupIndex] = useState(0);
  const [allGroupsData, setAllGroupsData] = useState<{words: string[], attention: number[][], ffnActivations?: number[], probabilities?: number[]}[]>([]);

  // Dynamic probabilities for Meaning Recall (updated when known/unknown words change)
  const [dynamicProbabilities, setDynamicProbabilities] = useState<number[]>([]);
  const [originalProbabilities, setOriginalProbabilities] = useState<number[]>([]);

  // Fetch dynamic probabilities for Meaning Recall when known/unknown indices change
  useEffect(() => {
    if (!input.trim() || !attentionData) return;
    const fetchDynamicProbabilities = async () => {
      try {
        // Calculate the modified attention matrix based on known/unknown words
        // This mirrors the logic in AttentionHeatmap for consistency
        let modifiedAttention = attentionData.attention.map(row => [...row]); // Deep copy
        
        // Apply unknown word logic: unknown words don't provide attention
        unknownTokenIndices.forEach(unknownIdx => {
          if (unknownIdx < modifiedAttention.length) {
            modifiedAttention[unknownIdx] = modifiedAttention[unknownIdx].map(() => 0);
          }
        });
        
        // Apply known word boosts
        if (knownTokenIndices && knownTokenIndices.length > 0) {
          // Compute original received attention
          const originalReceived = attentionData.words.map((_, i) => 
            attentionData.attention.reduce((sum, row) => sum + row[i], 0)
          );
          
          // Filter out punctuation for normalization
          const punctuationIndices = attentionData.words.map((word, i) => isPunctuation(word) ? i : -1).filter(i => i !== -1);
          const nonPunctuationOriginalReceived = originalReceived.filter((_, i) => !punctuationIndices.includes(i));
          const originalMinReceived = nonPunctuationOriginalReceived.length > 0 ? Math.min(...nonPunctuationOriginalReceived) : 0;
          const originalMaxReceived = nonPunctuationOriginalReceived.length > 0 ? Math.max(...nonPunctuationOriginalReceived) : 1;
          
          // Calculate current received attention from modified matrix
          const currentReceived = attentionData.words.map((_, i) => 
            modifiedAttention.reduce((sum, row) => sum + row[i], 0)
          );
          
          // Calculate normalized received attention
          const normReceived = currentReceived.map((v, i) => 
            punctuationIndices.includes(i)
              ? 0
              : (originalMaxReceived - originalMinReceived ? (v - originalMinReceived) / (originalMaxReceived - originalMinReceived) : 0)
          );
          
          // Apply known word boost: set normReceived to 1 and boost providers
          knownTokenIndices.forEach(iKnown => {
            const prevNormReceived = normReceived[iKnown];
            const boost = 1 - prevNormReceived;
            console.log(`Known word ${iKnown}: prevNormReceived=${prevNormReceived.toFixed(3)}, boost=${boost.toFixed(3)}`);
            
            if (boost > 0) {
              // Find total attention given to this known word from the modified matrix
              const totalAttentionToKnown = modifiedAttention.reduce((sum, row) => sum + row[iKnown], 0);
              
              if (totalAttentionToKnown > 0) {
                // Boost attention from providers proportionally
                modifiedAttention.forEach((row, j) => {
                  if (j !== iKnown && row[iKnown] > 0) {
                    const prop = row[iKnown] / totalAttentionToKnown;
                    const attentionBoost = prop * boost * (originalMaxReceived - originalMinReceived);
                    modifiedAttention[j][iKnown] += attentionBoost;
                  }
                });
              }
            }
          });
        }

        // Only send custom attention mask if there are actual modifications
        const hasModifications = (knownTokenIndices && knownTokenIndices.length > 0) || (unknownTokenIndices && unknownTokenIndices.length > 0);
        const requestBody: any = {
          text: input,
          known_indices: knownTokenIndices,
          unknown_indices: unknownTokenIndices,
        };
        
        // Only include custom_attention_mask if there are modifications
        if (hasModifications) {
          requestBody.custom_attention_mask = modifiedAttention;
          console.log('Sending custom attention mask for known/unknown words:', {
            knownTokenIndices,
            unknownTokenIndices,
            modifiedAttentionSample: modifiedAttention.slice(0, 3).map(row => row.slice(0, 3))
          });
        }

        // Fetch current probabilities using the modified attention matrix (only if modifications exist)
        const res = await fetch('http://localhost:8000/prediction-probabilities', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        });
        if (!res.ok) throw new Error('Failed to fetch dynamic probabilities');
        const data = await res.json();
        setDynamicProbabilities(data.probabilities);
        console.log('Received dynamic probabilities:', data.probabilities.slice(0, 5));

        // Fetch original probabilities (no known/unknown changes)
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
        console.log('Received original probabilities:', dataOrig.probabilities.slice(0, 5));
      } catch (e) {
        console.error('Error fetching dynamic probabilities:', e);
        setDynamicProbabilities([]);
        setOriginalProbabilities([]);
      }
    };
    fetchDynamicProbabilities();
  }, [input, knownTokenIndices, unknownTokenIndices, attentionData]);




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
      const groupResults: {words: string[], attention: number[][], ffnActivations?: number[], probabilities?: number[]}[] = [];
      
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
            attention: data.attention,
            ffnActivations: data.ffn_activations || [],
            probabilities: data.probabilities || []
          });
        }
        
        setAllGroupsData(groupResults);
        const currentGroup = groupResults[Math.min(currentGroupIndex, groupResults.length - 1)];
        if (currentGroup) {
          setAttentionData({ 
            words: currentGroup.words, 
            attention: currentGroup.attention,
            ffnActivations: currentGroup.ffnActivations || [],
            probabilities: currentGroup.probabilities || []
          });
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
  let displayFFNActivations = attentionData?.ffnActivations || [];
  let displayProbabilities = attentionData?.probabilities || [];
  
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
    
    // Process FFN activations by averaging over word pieces  
    displayFFNActivations = groups.map(g => {
      let sum = 0;
      for (const i of g) {
        sum += displayFFNActivations[i] || 0;
      }
      return g.length > 0 ? sum / g.length : 0;
    });
    
    // Process probabilities by averaging over word pieces (geometric mean would be more appropriate for probabilities, but using arithmetic for consistency)
    displayProbabilities = groups.map(g => {
      let sum = 0;
      for (const i of g) {
        sum += displayProbabilities[i] || 0;
      }
      return g.length > 0 ? sum / g.length : 0;
    });
    
    // Always exclude punctuation from calculations - zero out attention to/from punctuation words
    displayAttention = displayAttention.map((row, i) =>
      row.map((val, j) => {
        const isPuncI = isPunctuation(displayWords[i]);
        const isPuncJ = isPunctuation(displayWords[j]);
        // Zero out if either word is punctuation
        return (isPuncI || isPuncJ) ? 0 : val;
      })
    );
    
    // Also zero out FFN activations for punctuation
    displayFFNActivations = displayFFNActivations.map((val, i) => 
      isPunctuation(displayWords[i]) ? 0 : val
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
                probabilities={dynamicProbabilities.length > 0 ? dynamicProbabilities : displayProbabilities}
                originalProbabilities={originalProbabilities}
                ffnActivations={displayFFNActivations}
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
