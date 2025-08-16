import React, { useState, useEffect } from 'react';

declare const realData: any[];

interface AttentionHeatmapProps {
  words: string[];
  attention: number[][]; // attention[i][j]: how much word j attends to word i
  normProvided?: number[]; // normalized total provided
  normReceived?: number[]; // normalized total received
  selectedTokenIndices?: number[]; // indices of selected tokens
  setSelectedTokenIndices?: (indices: number[]) => void; // setter
  unknownTokenIndices?: number[]; // indices of words marked as unknown
  knownTokenIndices?: number[]; // indices of words marked as known
  setUnknownTokenIndices?: (indices: number[]) => void; // setter for unknown words
  setKnownTokenIndices?: (indices: number[]) => void; // setter for known words
  punctuationIndices?: number[]; // indices of punctuation words
  probabilities?: number[]; // BERT prediction probabilities for each word
  originalProbabilities?: number[]; // Original BERT prediction probabilities (before known/unknown changes)
  ffnActivations?: number[]; // FFN activation counts for each word
  sentence?: string; // The original sentence string for matching dataset metrics
}

const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({ 
  words, 
  attention, 
  selectedTokenIndices = [], 
  setSelectedTokenIndices, 
  unknownTokenIndices = [], 
  knownTokenIndices = [],
  setUnknownTokenIndices,
  setKnownTokenIndices,
  punctuationIndices = [],
  probabilities = [], // Probabilities for Meaning Recall
  originalProbabilities = [], // Original probabilities for delta calculation
  ffnActivations = [],
  sentence = ''
}) => {
  // Define datasetMetrics at the very top so it is available everywhere
  const datasetMetrics = [
    "SpringerEffort", "SpringerComprehension", "SpringerProbability", "SpringerRecall", "SpringerRetrieved", "SpringerSum", "SpringerReceived", "SpringerProvided"
  ];
  const displayWords = words;
  // Adapt attention: unknown words only receive, not provide
  // Also, recalculate norm scores based on this modified attention
  const displayAttention = attention.map((row, i) =>
    unknownTokenIndices.includes(i)
      ? row.map(() => 0)
      : [...row]
  );

  // Recompute provided/received and norm scores based on displayAttention
  const provided = displayAttention.map(row => row.reduce((a, b) => a + b, 0));
  const received = words.map((_, i) => displayAttention.reduce((sum, row) => sum + row[i], 0));

  // Compute original min/max for received from the original attention matrix (before zeroing unknowns)
  const originalReceived = words.map((_, i) => attention.reduce((sum, row) => sum + row[i], 0));
  
  // Filter out punctuation from originalReceived for normalization calculation
  const nonPunctuationOriginalReceived = originalReceived.filter((_, i) => !punctuationIndices.includes(i));
  const originalMinReceived = nonPunctuationOriginalReceived.length > 0 ? Math.min(...nonPunctuationOriginalReceived) : 0;
  const originalMaxReceived = nonPunctuationOriginalReceived.length > 0 ? Math.max(...nonPunctuationOriginalReceived) : 1;

  // For normProvided, exclude unknowns AND punctuation from min/max calculation
  const nonUnknownNonPunctuationIndices = words.map((_, i) => i).filter(i => !unknownTokenIndices.includes(i) && !punctuationIndices.includes(i));
  const minProvided = nonUnknownNonPunctuationIndices.length > 0 ? Math.min(...nonUnknownNonPunctuationIndices.map(i => provided[i])) : 0;
  const maxProvided = nonUnknownNonPunctuationIndices.length > 0 ? Math.max(...nonUnknownNonPunctuationIndices.map(i => provided[i])) : 1;
  let normProvided = provided.map((v, i) =>
    unknownTokenIndices.includes(i) || punctuationIndices.includes(i)
      ? 0  // Set both unknowns and punctuation to 0
      : (maxProvided - minProvided ? (v - minProvided) / (maxProvided - minProvided) : 0)
  );
  // For normReceived, exclude punctuation from normalization range
  let normReceived = received.map((v, i) => 
    punctuationIndices.includes(i)
      ? 0  // Set punctuation to 0
      : (originalMaxReceived - originalMinReceived ? (v - originalMinReceived) / (originalMaxReceived - originalMinReceived) : 0)
  );

  // If a word is marked as known, set its normReceived to 1 and distribute its previous normReceived among its providers proportionally
  if (knownTokenIndices && knownTokenIndices.length > 0) {
    let normReceivedCopy = [...normReceived];
    let normProvidedCopy = [...normProvided];
    knownTokenIndices.forEach(iKnown => {
      // --- NormReceived logic (as before) ---
      const prevNormReceived = normReceivedCopy[iKnown];
      normReceivedCopy[iKnown] = 1;

      // --- NormProvided boost for providers ---
      // Find all providers (words that give attention to iKnown)
      const totalAttentionToKnown = attention.reduce((sum, row) => sum + row[iKnown], 0);
      const delta = 1 - prevNormReceived;
      if (totalAttentionToKnown > 0 && delta > 0) {
        attention.forEach((row, j) => {
          if (j !== iKnown && row[iKnown] > 0) {
            const prop = row[iKnown] / totalAttentionToKnown;
            normProvidedCopy[j] += prop * delta;
          }
        });
      }
    });
    normReceived = normReceivedCopy;
    normProvided = normProvidedCopy;
  }
  // Total attention received (sum of column)
  const totalReceived = received;
  // Total attention provided (sum of row)
  const totalProvided = provided;

  const [scoreSortMetric, setScoreSortMetric] = useState<'received' | 'provided' | 'normSum' | 'normRetrieved' | 'normRecalled'>('normSum');
  // Percent-based band controls
  const [upperBandPercent, setUpperBandPercent] = useState(20); // percent of words in upper band
  const [lowerBandPercent, setLowerBandPercent] = useState(20); // percent of words in lower band
  // These are percent values (0-100)
  // The thresholds will be computed from the current metric distribution
  const [upperThreshold, setUpperThreshold] = useState(0.6); // computed, not user-editable
  const [lowerThreshold, setLowerThreshold] = useState(0.4); // computed, not user-editable
  
  // Band expansion state
  const [expandedBands, setExpandedBands] = useState<{[key: string]: boolean}>({
    lower: false,
    middle: false, 
    upper: false
  });

  // boostWordProbability removed (unused)

  // Use unique ids for each word position
  const wordObjs = displayWords.map((word, idx) => ({ word, index: idx }));
  
  // Calculate raw sums of original received and provided scores (excluding punctuation)
  // For unknown words, use only their totalReceived score
  const rawSums = wordObjs.map(({ index }) => 
    unknownTokenIndices.includes(index) 
      ? totalReceived[index]  // Unknown words: use only received score
      : totalReceived[index] + totalProvided[index]  // Normal words: use sum
  );
  
  // Filter out punctuation from raw sums for normalization calculation
  const nonPunctuationRawSums = rawSums.filter((_, i) => !punctuationIndices.includes(i));
  
  // Normalize the raw sums to 0-1 range using only non-punctuation values
  const minRawSum = nonPunctuationRawSums.length > 0 ? Math.min(...nonPunctuationRawSums) : 0;
  const maxRawSum = nonPunctuationRawSums.length > 0 ? Math.max(...nonPunctuationRawSums) : 1;
  let normalizedSums = rawSums.map((sum, i) => 
    punctuationIndices.includes(i) 
      ? 0  // Set punctuation normSum to 0
      : (maxRawSum - minRawSum ? (sum - minRawSum) / (maxRawSum - minRawSum) : 0)
  );
  
  // Recalculate normalizedSums after known word processing to reflect the boosted scores
  if (knownTokenIndices && knownTokenIndices.length > 0) {
    // Create new raw sums using the updated normReceived and normProvided values
    const updatedRawSums = wordObjs.map(({ index }) => 
      unknownTokenIndices.includes(index) 
        ? normReceived[index] * (originalMaxReceived - originalMinReceived) + originalMinReceived  // Convert back to original scale
        : (normReceived[index] * (originalMaxReceived - originalMinReceived) + originalMinReceived) + 
          (normProvided[index] * (maxProvided - minProvided) + minProvided)  // Convert both back and sum
    );
    
    // Filter out punctuation from updated raw sums for normalization calculation
    const nonPunctuationUpdatedRawSums = updatedRawSums.filter((_, i) => !punctuationIndices.includes(i));
    
    // Normalize the updated raw sums to 0-1 range using only non-punctuation values
    const minUpdatedRawSum = nonPunctuationUpdatedRawSums.length > 0 ? Math.min(...nonPunctuationUpdatedRawSums) : 0;
    const maxUpdatedRawSum = nonPunctuationUpdatedRawSums.length > 0 ? Math.max(...nonPunctuationUpdatedRawSums) : 1;
    normalizedSums = updatedRawSums.map((sum, i) => 
      punctuationIndices.includes(i) 
        ? 0  // Set punctuation normSum to 0
        : (maxUpdatedRawSum - minUpdatedRawSum ? (sum - minUpdatedRawSum) / (maxUpdatedRawSum - minUpdatedRawSum) : 0)
    );
  }
  
  // Normalize FFN activations (0-1 range) - filter out punctuation for normalization
  const nonPunctuationFFN = ffnActivations.filter((_, i) => !punctuationIndices.includes(i));
  const minFFN = nonPunctuationFFN.length > 0 ? Math.min(...nonPunctuationFFN) : 0;
  const maxFFN = nonPunctuationFFN.length > 0 ? Math.max(...nonPunctuationFFN) : 1;
  const normalizedFFNActivations = ffnActivations.map((count, i) => 
    punctuationIndices.includes(i) 
      ? 0  // Set punctuation FFN activations to 0
      : (maxFFN - minFFN ? (count - minFFN) / (maxFFN - minFFN) : 0)
  );
  
  // Normalize probabilities (0-1 range) - filter out punctuation for normalization
  const nonPunctuationProbs = probabilities && probabilities.filter((_, i) => !punctuationIndices.includes(i)) || [];
  const minProb = nonPunctuationProbs.length > 0 ? Math.min(...nonPunctuationProbs) : 0;
  const maxProb = nonPunctuationProbs.length > 0 ? Math.max(...nonPunctuationProbs) : 1;
  const normalizedProbabilities = probabilities ? probabilities.map((prob, i) => 
    punctuationIndices.includes(i) 
      ? 0  // Set punctuation probabilities to 0
      : (maxProb - minProb ? (prob - minProb) / (maxProb - minProb) : 0)
  ) : [];
  
  // Compute metrics for each word position ONCE, in original order
  const metrics = wordObjs.map(({ word, index }, i) => ({
    word,
    index,
    received: totalReceived[index],
    provided: totalProvided[index],
    normProvided: normProvided[index],
    normReceived: normReceived[index],
    normSum: normalizedSums[i],
    normRetrieved: normalizedFFNActivations[index] || 0, // Normalized FFN activations (0-1) - Retrieved Meaning
    normRecalled: normalizedProbabilities[index] || 0, // Normalized probabilities (0-1) - Meaning Recall
    recallDelta: probabilities && originalProbabilities && probabilities.length === originalProbabilities.length 
      ? ((probabilities[index] || 0) - (originalProbabilities[index] || 0)) * 100 // Delta as percentage
      : 0, // Probability change delta in percentage points
    isUnknown: unknownTokenIndices.includes(index),
    isKnown: knownTokenIndices.includes(index),
  }));
  // Sorting logic: sort metrics, do not recalculate
  // Filter out punctuation from scoreboard since they're always excluded from calculations
  const filteredMetrics = metrics.filter((_, i) => 
    !punctuationIndices.includes(i)
  );
  const scoreboard = [...filteredMetrics];
  scoreboard.sort((a, b) => a[scoreSortMetric] - b[scoreSortMetric]);

  // Text score: average (mean) of normSum for all words (punctuation always excluded)
  const textScore = filteredMetrics.length > 0 ? filteredMetrics.reduce((sum, m) => sum + m.normSum, 0) / filteredMetrics.length : 0;

  // Suggestion: find the words with the lowest normalized difference
  // const worstDescribed = scoreboard.slice(-3).map(s => s.word); // bottom 3

  // Removed selectedWordIdx and selectedWordIdx2 (pick word functionality removed)

  // Color scale for score highlighting
  function getColorScale(arr: number[]) {
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    return (v: number) => {
      const t = (v - min) / (max - min + 1e-8);
      // White (low) to Blue (#00AAFF - high)
      const r = Math.round(255 - (255 - 0) * t);     // 255 → 0
      const g = Math.round(255 - (255 - 170) * t);   // 255 → 170 (AA in hex = 170 in decimal)
      const b = 255;                                  // Always 255
      return `rgb(${r},${g},${b})`;
    };
  }
  // --- Multi-token selection coloring logic ---
  // If tokens are selected, color by attention to/from selected tokens depending on sorting
  let customColorArr: number[] | null = null;
  let customColorLabel = '';
  if (selectedTokenIndices && selectedTokenIndices.length > 0) {
    if (scoreSortMetric === 'provided') {
      // Color by attention given to selected tokens (row to selected columns)
      customColorArr = metrics.map((_, i) =>
        selectedTokenIndices.reduce((sum, selIdx) => sum + displayAttention[i][selIdx], 0)
      );
      customColorLabel = 'attention given to selected';
    } else if (scoreSortMetric === 'received') {
      // Color by attention received from selected tokens (column from selected rows)
      customColorArr = metrics.map((_, i) =>
        selectedTokenIndices.reduce((sum, selIdx) => sum + displayAttention[selIdx][i], 0)
      );
      customColorLabel = 'attention received from selected';
    } else {
      // Default: color by attention given to selected
      customColorArr = metrics.map((_, i) =>
        selectedTokenIndices.reduce((sum, selIdx) => sum + displayAttention[i][selIdx], 0)
      );
      customColorLabel = 'attention given to selected';
    }
  }

  // Always display original text in original order, colored by the current metric or selection
  const originalTextColorArr = customColorArr || metrics.map(m => {
    if (scoreSortMetric === 'received') return m.normReceived;
    if (scoreSortMetric === 'provided') return m.normProvided;
    if (scoreSortMetric === 'normSum') return m.normSum;
    if (scoreSortMetric === 'normRetrieved') return m.normRetrieved;
    if (scoreSortMetric === 'normRecalled') return m.normRecalled;
    return m[scoreSortMetric];
  });
  const originalTextColorScale = getColorScale(originalTextColorArr);

  // Option to color by band
  const [colorByBand, setColorByBand] = useState(false);
  // Band colors: configurable via color pickers
  const [bandColors, setBandColors] = useState({
    above: '#00AAFF', // upper band (highest attention)
    between: '#99DDFF', // middle band
    below: '#CCEEFF', // lower band (lowest attention)
  });

  // Compute thresholds so that upperBandPercent% of words are above upperThreshold, and lowerBandPercent% below lowerThreshold
  function computePercentile(arr: number[], percent: number) {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = Math.max(0, Math.min(sorted.length - 1, Math.round((1 - percent / 100) * (sorted.length - 1))));
    return sorted[idx];
  }
  function computeLowerPercentile(arr: number[], percent: number) {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = Math.max(0, Math.min(sorted.length - 1, Math.round((percent / 100) * (sorted.length - 1))));
    return sorted[idx];
  }
  useEffect(() => {
    // Create filtered color array that excludes punctuation (always excluded from calculations)
    const filteredColorArr = originalTextColorArr.filter((_, i) => 
      !punctuationIndices.includes(i)
    );
    // Compute thresholds from filtered color metric to match scoreboard
    setUpperThreshold(computePercentile(filteredColorArr, upperBandPercent));
    setLowerThreshold(computeLowerPercentile(filteredColorArr, lowerBandPercent));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [originalTextColorArr, upperBandPercent, lowerBandPercent, punctuationIndices]);

  // State to show/hide the Word Attention Heatmap
  const [showWordHeatmap, setShowWordHeatmap] = useState(false);
  const [showFormulas, setShowFormulas] = useState(false);
  const [showCorrelation, setShowCorrelation] = useState(false);
  const [showContextAnalysis, setShowContextAnalysis] = useState(false);

  // White-to-blue color scale for heatmap (matching #00AAFF theme)
  function getBlueWhiteColor(value: number, min: number, max: number) {
    const t = (value - min) / (max - min + 1e-8);
    // Interpolate from white (low) to #00AAFF (high)
    const r = Math.round(255 - (255 - 0) * t);     // 255 → 0
    const g = Math.round(255 - (255 - 170) * t);   // 255 → 170 (AA in hex = 170 in decimal)
    const b = 255;                                  // Always 255
    return `rgb(${r},${g},${b})`;
  }

  // Compute min/max for attention values for color scaling
  const flatAttention = displayAttention.flat();
  
  // Use safe min/max calculation to avoid stack overflow with large arrays
  let attMin = Infinity;
  let attMax = -Infinity;
  for (const value of flatAttention) {
    if (value < attMin) attMin = value;
    if (value > attMax) attMax = value;
  }
  
  // Fallback values if arrays are empty
  if (!isFinite(attMin)) attMin = 0;
  if (!isFinite(attMax)) attMax = 1;

  return (
    <div style={{ overflowX: 'auto', margin: '1em 0' }}>
      {/* Move original text display above the line graph */}
      <div style={{margin: '16px 0', fontSize: '1.1em', lineHeight: 1.7}}>
        <div style={{display: 'flex', alignItems: 'center', gap: 16, marginBottom: 8}}>
          <b>Original Text{selectedTokenIndices && selectedTokenIndices.length > 0 ? ' (colored by ' + customColorLabel + ')' : ' (colored by ' + scoreSortMetric + ')'}:</b>
          <div style={{fontSize: '0.8em'}}>
            <label style={{marginRight: 8}}>Sort by:</label>
            <select value={scoreSortMetric} onChange={e => setScoreSortMetric(e.target.value as 'received' | 'provided' | 'normSum' | 'normRetrieved' | 'normRecalled')} style={{fontSize: 13, padding: '2px 6px'}}>
              <option value="received">Norm. Received</option>
              <option value="provided">Norm. Provided</option>
              <option value="normSum">Norm. Sum</option>
              <option value="normRetrieved">Retrieved Meaning</option>
              <option value="normRecalled">Meaning Recall</option>
            </select>
          </div>
        </div>
        
        {/* Percent-based band controls and color by band toggle */}
        <div style={{margin: '8px 0', fontSize: '0.9em', display: 'flex', alignItems: 'center', gap: 16}}>
          <div>
            <label style={{marginRight: 8}}>Upper band (% of words):</label>
            <input
              type="number"
              value={upperBandPercent}
              onChange={e => setUpperBandPercent(Math.max(0, Math.min(100, Number(e.target.value))))}
              step="1"
              min="0"
              max="100"
              style={{width: '60px', padding: '2px 4px', marginRight: 16}}
            />
            <label style={{marginRight: 8}}>Lower band (% of words):</label>
            <input
              type="number"
              value={lowerBandPercent}
              onChange={e => setLowerBandPercent(Math.max(0, Math.min(100, Number(e.target.value))))}
              step="1"
              min="0"
              max="100"
              style={{width: '60px', padding: '2px 4px', marginRight: 16}}
            />
            <span style={{marginLeft: 12, color: '#888', fontSize: '0.95em'}}>
              (Current thresholds: upper ≥ {upperThreshold.toFixed(3)}, lower ≤ {lowerThreshold.toFixed(3)})
            </span>
          </div>
          <label style={{marginLeft: 24, display: 'flex', alignItems: 'center', gap: 4}}>
            <input
              type="checkbox"
              checked={colorByBand}
              onChange={e => setColorByBand(e.target.checked)}
              style={{marginRight: 4}}
            />
            Color by band
          </label>
        </div>

        {/* Band color selectors */}
        {colorByBand && (
          <div style={{margin: '8px 0', fontSize: '0.9em', display: 'flex', alignItems: 'center', gap: 16, padding: '8px', backgroundColor: '#f5f5f5', borderRadius: '4px'}}>
            <span style={{fontWeight: 'bold', marginRight: 8}}>Band Colors:</span>
            <div style={{display: 'flex', alignItems: 'center', gap: 4}}>
              <label style={{fontSize: '0.85em'}}>Upper:</label>
              <input
                type="color"
                value={bandColors.above}
                onChange={e => setBandColors(prev => ({...prev, above: e.target.value}))}
                style={{width: '30px', height: '24px', border: 'none', borderRadius: '3px', cursor: 'pointer'}}
                title="Color for highest attention words"
              />
            </div>
            <div style={{display: 'flex', alignItems: 'center', gap: 4}}>
              <label style={{fontSize: '0.85em'}}>Middle:</label>
              <input
                type="color"
                value={bandColors.between}
                onChange={e => setBandColors(prev => ({...prev, between: e.target.value}))}
                style={{width: '30px', height: '24px', border: 'none', borderRadius: '3px', cursor: 'pointer'}}
                title="Color for medium attention words"
              />
            </div>
            <div style={{display: 'flex', alignItems: 'center', gap: 4}}>
              <label style={{fontSize: '0.85em'}}>Lower:</label>
              <input
                type="color"
                value={bandColors.below}
                onChange={e => setBandColors(prev => ({...prev, below: e.target.value}))}
                style={{width: '30px', height: '24px', border: 'none', borderRadius: '3px', cursor: 'pointer'}}
                title="Color for lowest attention words"
              />
            </div>
            <button
              onClick={() => setBandColors({
                above: '#00AAFF',
                between: '#99DDFF', 
                below: '#CCEEFF'
              })}
              style={{
                fontSize: '0.8em',
                padding: '4px 8px',
                backgroundColor: '#fff',
                border: '1px solid #ddd',
                borderRadius: '3px',
                cursor: 'pointer',
                marginLeft: 8
              }}
              title="Reset to default colors"
            >
              Reset
            </button>
          </div>
        )}

        <div style={{marginTop: 8, background: 'white', padding: 20, borderRadius: 6, fontFamily: 'Verdana, sans-serif', fontWeight: 'normal', fontSize: '12px', wordBreak: 'normal', position: 'relative', overflowX: 'auto', overflowY: 'visible'}}>
          <div style={{display: 'flex', flexWrap: 'wrap', columnGap: '0px', rowGap: '10px', lineHeight: '1.8'}}>
            {metrics.map((m, i) => {
              const isPunctuation = punctuationIndices.includes(i);
              const isSelected = selectedTokenIndices.includes(i);
              const value = originalTextColorArr[i];
              
              // Determine band for styling
              let band: 'above' | 'between' | 'below' = 'between';
              if (!isPunctuation) {
                if (value >= upperThreshold) {
                  band = 'above';
                } else if (value <= lowerThreshold) {
                  band = 'below';
                }
              }
              
              // Calculate attention directions for gradients
              let leftAttention = 0;
              let rightAttention = 0;
              
              if (!isPunctuation) {
                // Find sentence boundaries
                let sentenceStart = 0;
                let sentenceEnd = metrics.length - 1;
                
                for (let j = i - 1; j >= 0; j--) {
                  if (/[.!?]$/.test(metrics[j].word.trim())) {
                    sentenceStart = j + 1;
                    break;
                  }
                }
                
                for (let j = i + 1; j < metrics.length; j++) {
                  if (/[.!?]$/.test(metrics[j].word.trim())) {
                    sentenceEnd = j;
                    break;
                  }
                }
                
                // Calculate directional attention
                for (let j = sentenceStart; j <= sentenceEnd; j++) {
                  if (j === i) continue;
                  const attentionValue = displayAttention[j][i];
                  
                  if (j < i) {
                    leftAttention += attentionValue;
                  } else {
                    rightAttention += attentionValue;
                  }
                }
              }
              
              // Determine dominant direction
              let dominantDirection = 'none';
              if (leftAttention > rightAttention && leftAttention > 0) {
                dominantDirection = 'left';
              } else if (rightAttention > leftAttention && rightAttention > 0) {
                dominantDirection = 'right';
              }
              
              // Create background with gradient
              const createBackground = () => {
                if (isPunctuation) return 'white';
                
                let baseColor;
                
                if (colorByBand && dominantDirection !== 'none') {
                  // For combined sequences, calculate averaged color across all words in the sequence
                  const sequenceIndices = [];
                  
                  // Find all words in the same direction sequence
                  let sequenceStart = i;
                  let sequenceEnd = i;
                  
                  // Find start of sequence
                  for (let j = i - 1; j >= 0; j--) {
                    if (punctuationIndices.includes(j)) break;
                    
                    // Calculate direction for word j
                    let jLeftAttention = 0, jRightAttention = 0;
                    let jSentenceStart = 0, jSentenceEnd = metrics.length - 1;
                    
                    for (let k = j - 1; k >= 0; k--) {
                      if (/[.!?]$/.test(metrics[k].word.trim())) {
                        jSentenceStart = k + 1;
                        break;
                      }
                    }
                    for (let k = j + 1; k < metrics.length; k++) {
                      if (/[.!?]$/.test(metrics[k].word.trim())) {
                        jSentenceEnd = k;
                        break;
                      }
                    }
                    
                    for (let k = jSentenceStart; k <= jSentenceEnd; k++) {
                      if (k === j) continue;
                      const attentionValue = displayAttention[k][j];
                      if (k < j) jLeftAttention += attentionValue;
                      else jRightAttention += attentionValue;
                    }
                    
                    let jDirection = 'none';
                    if (jLeftAttention > jRightAttention && jLeftAttention > 0) jDirection = 'left';
                    else if (jRightAttention > jLeftAttention && jRightAttention > 0) jDirection = 'right';
                    
                    if (jDirection === dominantDirection) {
                      sequenceStart = j;
                    } else {
                      break;
                    }
                  }
                  
                  // Find end of sequence
                  for (let j = i + 1; j < metrics.length; j++) {
                    if (punctuationIndices.includes(j)) break;
                    
                    // Calculate direction for word j
                    let jLeftAttention = 0, jRightAttention = 0;
                    let jSentenceStart = 0, jSentenceEnd = metrics.length - 1;
                    
                    for (let k = j - 1; k >= 0; k--) {
                      if (/[.!?]$/.test(metrics[k].word.trim())) {
                        jSentenceStart = k + 1;
                        break;
                      }
                    }
                    for (let k = j + 1; k < metrics.length; k++) {
                      if (/[.!?]$/.test(metrics[k].word.trim())) {
                        jSentenceEnd = k;
                        break;
                      }
                    }
                    
                    for (let k = jSentenceStart; k <= jSentenceEnd; k++) {
                      if (k === j) continue;
                      const attentionValue = displayAttention[k][j];
                      if (k < j) jLeftAttention += attentionValue;
                      else jRightAttention += attentionValue;
                    }
                    
                    let jDirection = 'none';
                    if (jLeftAttention > jRightAttention && jLeftAttention > 0) jDirection = 'left';
                    else if (jRightAttention > jLeftAttention && jRightAttention > 0) jDirection = 'right';
                    
                    if (jDirection === dominantDirection) {
                      sequenceEnd = j;
                    } else {
                      break;
                    }
                  }
                  
                  // Collect all indices in the sequence
                  for (let j = sequenceStart; j <= sequenceEnd; j++) {
                    if (!punctuationIndices.includes(j)) {
                      sequenceIndices.push(j);
                    }
                  }
                  
                  // Calculate average band color for the sequence
                  if (sequenceIndices.length > 1) {
                    let totalR = 0, totalG = 0, totalB = 0;
                    
                    sequenceIndices.forEach(idx => {
                      const seqValue = originalTextColorArr[idx];
                      let seqBand: 'above' | 'between' | 'below' = 'between';
                      if (seqValue >= upperThreshold) seqBand = 'above';
                      else if (seqValue <= lowerThreshold) seqBand = 'below';
                      
                      const seqColor = bandColors[seqBand];
                      const rgb = seqColor.match(/^#([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
                      if (rgb) {
                        totalR += parseInt(rgb[1], 16);
                        totalG += parseInt(rgb[2], 16);
                        totalB += parseInt(rgb[3], 16);
                      }
                    });
                    
                    const avgR = Math.round(totalR / sequenceIndices.length);
                    const avgG = Math.round(totalG / sequenceIndices.length);
                    const avgB = Math.round(totalB / sequenceIndices.length);
                    baseColor = `rgb(${avgR}, ${avgG}, ${avgB})`;
                  } else {
                    baseColor = bandColors[band];
                  }
                } else {
                  baseColor = colorByBand 
                    ? bandColors[band]
                    : originalTextColorScale(originalTextColorArr[i]);
                }
                
                if (dominantDirection === 'none') return baseColor;
                
                // Calculate previous and next dominant directions for gradient logic
                const getPrevDirection = () => {
                  if (i === 0 || punctuationIndices.includes(i - 1)) return 'none';
                  
                  let prevLeft = 0, prevRight = 0;
                  
                  // Find sentence boundaries for previous word
                  let sentenceStart = 0, sentenceEnd = metrics.length - 1;
                  
                  for (let j = i - 2; j >= 0; j--) {
                    if (/[.!?]$/.test(metrics[j].word.trim())) {
                      sentenceStart = j + 1;
                      break;
                    }
                  }
                  
                  for (let j = i; j < metrics.length; j++) {
                    if (/[.!?]$/.test(metrics[j].word.trim())) {
                      sentenceEnd = j;
                      break;
                    }
                  }
                  
                  for (let j = sentenceStart; j <= sentenceEnd; j++) {
                    if (j === i - 1) continue;
                    const attentionValue = displayAttention[j][i - 1];
                    if (j < i - 1) prevLeft += attentionValue;
                    else prevRight += attentionValue;
                  }
                  
                  if (prevLeft > prevRight && prevLeft > 0) return 'left';
                  if (prevRight > prevLeft && prevRight > 0) return 'right';
                  return 'none';
                };
                
                const getNextDirection = () => {
                  if (i === metrics.length - 1 || punctuationIndices.includes(i + 1)) return 'none';
                  
                  let nextLeft = 0, nextRight = 0;
                  
                  // Find sentence boundaries for next word
                  let sentenceStart = 0, sentenceEnd = metrics.length - 1;
                  
                  for (let j = i; j >= 0; j--) {
                    if (/[.!?]$/.test(metrics[j].word.trim())) {
                      sentenceStart = j + 1;
                      break;
                    }
                  }
                  
                  for (let j = i + 2; j < metrics.length; j++) {
                    if (/[.!?]$/.test(metrics[j].word.trim())) {
                      sentenceEnd = j;
                      break;
                    }
                  }
                  
                  for (let j = sentenceStart; j <= sentenceEnd; j++) {
                    if (j === i + 1) continue;
                    const attentionValue = displayAttention[j][i + 1];
                    if (j < i + 1) nextLeft += attentionValue;
                    else nextRight += attentionValue;
                  }
                  
                  if (nextLeft > nextRight && nextLeft > 0) return 'left';
                  if (nextRight > nextLeft && nextRight > 0) return 'right';
                  return 'none';
                };
                
                const prevDirection = getPrevDirection();
                const nextDirection = getNextDirection();
                
                switch (dominantDirection) {
                  case 'left':
                    // Only apply gradient if this is the first word in a left sequence
                    const isFirstInLeftSequence = prevDirection !== 'left';
                    return isFirstInLeftSequence 
                      ? `linear-gradient(to right, white, ${baseColor})` 
                      : baseColor;
                  case 'right':
                    // Only apply gradient if this is the last word in a right sequence
                    const isLastInRightSequence = nextDirection !== 'right';
                    return isLastInRightSequence 
                      ? `linear-gradient(to left, white, ${baseColor})` 
                      : baseColor;
                  default:
                    return baseColor;
                }
              };
              
              // Get border radius - only apply radius at sequence boundaries
              const getBorderRadius = () => {
                if (isPunctuation || dominantDirection === 'none') return '3px';
                
                // Check if this is the start or end of a sequence with the same direction
                const prevWord = i > 0 ? metrics[i - 1] : null;
                const nextWord = i < metrics.length - 1 ? metrics[i + 1] : null;
                
                // Calculate previous and next dominant directions
                const getPrevDirection = () => {
                  if (!prevWord || punctuationIndices.includes(i - 1)) return 'none';
                  
                  let prevLeft = 0, prevRight = 0;
                  
                  // Find sentence boundaries for previous word
                  let sentenceStart = 0, sentenceEnd = metrics.length - 1;
                  
                  for (let j = i - 2; j >= 0; j--) {
                    if (/[.!?]$/.test(metrics[j].word.trim())) {
                      sentenceStart = j + 1;
                      break;
                    }
                  }
                  
                  for (let j = i; j < metrics.length; j++) {
                    if (/[.!?]$/.test(metrics[j].word.trim())) {
                      sentenceEnd = j;
                      break;
                    }
                  }
                  
                  for (let j = sentenceStart; j <= sentenceEnd; j++) {
                    if (j === i - 1) continue;
                    const attentionValue = displayAttention[j][i - 1];
                    if (j < i - 1) prevLeft += attentionValue;
                    else prevRight += attentionValue;
                  }
                  
                  if (prevLeft > prevRight && prevLeft > 0) return 'left';
                  if (prevRight > prevLeft && prevRight > 0) return 'right';
                  return 'none';
                };
                
                const getNextDirection = () => {
                  if (!nextWord || punctuationIndices.includes(i + 1)) return 'none';
                  
                  let nextLeft = 0, nextRight = 0;
                  
                  // Find sentence boundaries for next word
                  let sentenceStart = 0, sentenceEnd = metrics.length - 1;
                  
                  for (let j = i; j >= 0; j--) {
                    if (/[.!?]$/.test(metrics[j].word.trim())) {
                      sentenceStart = j + 1;
                      break;
                    }
                  }
                  
                  for (let j = i + 2; j < metrics.length; j++) {
                    if (/[.!?]$/.test(metrics[j].word.trim())) {
                      sentenceEnd = j;
                      break;
                    }
                  }
                  
                  for (let j = sentenceStart; j <= sentenceEnd; j++) {
                    if (j === i + 1) continue;
                    const attentionValue = displayAttention[j][i + 1];
                    if (j < i + 1) nextLeft += attentionValue;
                    else nextRight += attentionValue;
                  }
                  
                  if (nextLeft > nextRight && nextLeft > 0) return 'left';
                  if (nextRight > nextLeft && nextRight > 0) return 'right';
                  return 'none';
                };
                
                const prevDirection = getPrevDirection();
                const nextDirection = getNextDirection();
                
                if (dominantDirection === 'left') {
                  // Only apply right radius if this is the last word in a left sequence
                  const isLastInSequence = nextDirection !== 'left';
                  return isLastInSequence ? '0px 15px 15px 0px' : '0px';
                } else if (dominantDirection === 'right') {
                  // Only apply left radius if this is the first word in a right sequence  
                  const isFirstInSequence = prevDirection !== 'right';
                  return isFirstInSequence ? '15px 0px 0px 15px' : '0px';
                }
                
                return '3px';
              };
              
              // Get font styling
              const getFontStyle = () => {
                if (isPunctuation) {
                  return { fontWeight: 'normal', fontStyle: 'normal' };
                }
                
                switch (band) {
                  case 'above':
                    return { fontWeight: 'bold', fontStyle: 'normal' };
                  case 'below':
                    return { fontWeight: 'normal', fontStyle: 'italic' };
                  default:
                    return { fontWeight: 'normal', fontStyle: 'normal' };
                }
              };
              
              const fontStyle = getFontStyle();
              const maxAttention = Math.max(leftAttention, rightAttention);
              
              // Capitalize first word of sentences
              const isSentenceStart = (index: number): boolean => {
                if (index === 0) return true;
                for (let j = index - 1; j >= 0; j--) {
                  const prevWord = metrics[j].word;
                  if (/[.!?]$/.test(prevWord.trim())) return true;
                  if (!/^[^\w]*$/.test(prevWord)) return false;
                }
                return false;
              };
              
              const capitalizeWord = (word: string): string => {
                if (!word || word.length === 0) return word;
                return word.charAt(0).toUpperCase() + word.slice(1);
              };
              
              // Clean word by removing ## symbols
              const cleanWord = (word: string): string => {
                return word.replace(/##/g, '');
              };
              
              // Create tooltip
              const bandLabel = band === 'above' ? 'above threshold' : 
                               band === 'below' ? 'below threshold' : 'within thresholds';
              
              const directionalInfo = maxAttention > 0 
                ? `Most attention from: ${dominantDirection} (${maxAttention.toFixed(3)})`
                : 'No significant horizontal attention';
              
              const tooltipParts = [
                `Norm sum: ${m.normSum.toFixed(3)} (${bandLabel})`,
                directionalInfo,
                m.isUnknown && 'Unknown word: only receives attention, does not provide.',
                selectedTokenIndices.length > 0 && customColorArr && 
                  `Attention ${scoreSortMetric === 'received' ? 'received from' : 'given to'} selected: ${customColorArr[i].toFixed(3)}`,
                selectedTokenIndices.length === 0 && 
                  `Norm. ${scoreSortMetric}: ${(m as any)[
                    scoreSortMetric === 'normSum' ? 'normSum' : 
                    scoreSortMetric === 'provided' ? 'normProvided' : 
                    scoreSortMetric === 'received' ? 'normReceived' :
                    scoreSortMetric === 'normRetrieved' ? 'normRetrieved' :
                    scoreSortMetric === 'normRecalled' ? 'normRecalled' : 'normSum'
                  ].toFixed(3)}`
              ].filter(Boolean).join(' | ');
              
              return (
                <span
                  key={m.index}
                  style={{
                    background: createBackground(),
                    color: m.isUnknown ? '#00AAFF' : (isPunctuation ? '#000' : '#000'),
                    borderRadius: getBorderRadius(),
                    textAlign: 'center',
                    padding: isSelected ? '2px 6px' : '4px 8px',
                    display: 'inline-block',
                    fontWeight: fontStyle.fontWeight,
                    fontStyle: fontStyle.fontStyle,
                    cursor: 'pointer',
                    border: isSelected ? '2px solid #0072B2' : 'none',
                    transition: 'border 0.1s',
                    textDecoration: m.isUnknown ? 'underline wavy #00AAFF' : undefined,
                    whiteSpace: 'nowrap',
                  }}
                  onClick={() => {
                    if (!setSelectedTokenIndices) return;
                    setSelectedTokenIndices(
                      isSelected 
                        ? selectedTokenIndices.filter(idx => idx !== i)
                        : [...selectedTokenIndices, i]
                    );
                  }}
                  title={tooltipParts}
                >
                  {isSentenceStart(i) ? capitalizeWord(cleanWord(m.word)) : cleanWord(m.word)}
                </span>
              );
            })}
          </div>
        </div>
        {/* Removed ScoreLineGraph component */}
      </div>
      {/* Scoreboard Table: Split by bands with unknown/known marking */}
      <div style={{ width: '100%' }}>
        <b>Scoreboard: Understanding Mechanisms by Attention Bands | Text Inference Score: {textScore.toFixed(3)}</b>
        
        {/* Meaning Dimensions Explanation */}
        <div style={{ 
          margin: '12px 0', 
          padding: '12px', 
          backgroundColor: '#f9f9f9', 
          borderRadius: '6px', 
          fontSize: '0.85em', 
          color: '#666',
          border: '1px solid #e0e0e0'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '6px', color: '#333' }}>Six Meaning Dimensions:</div>
          <div><strong>Received/Provided Meaning:</strong> Attention flow - how much attention words receive from or provide to other words</div>
          <div><strong>Constructed Meaning:</strong> Contextual reasoning - normalized sum of attention interactions</div>
          <div><strong>Retrieved Meaning:</strong> Knowledge access - FFN neuron activations measuring stored knowledge retrieval</div>
          <div><strong>Meaning Recall:</strong> Predictability - probability of correctly predicting each word from context</div>
        </div>
        
        {/* Helper function to determine band for a word */}
        {(() => {
          // Create the same filtered data that excludes punctuation (always excluded from calculations)
          const filteredColorArr = originalTextColorArr.filter((_, i) => 
            !punctuationIndices.includes(i)
          );
          
          // Get the same metric values that were used to calculate thresholds
          const getMetricValueForBanding = (item: typeof scoreboard[0]) => {
            const index = item.index;
            // Find the position of this word in the filtered array (punctuation always excluded)
            const filteredIndex = originalTextColorArr.slice(0, index + 1)
              .filter((_, i) => !punctuationIndices.includes(i))
              .length - 1;
            
            // Use the value from the same filtered array used for thresholds
            return filteredColorArr[filteredIndex];
          };

          const upperBandWords = scoreboard.filter(item => {
            const value = getMetricValueForBanding(item);
            return value >= upperThreshold;
          });
          const middleBandWords = scoreboard.filter(item => {
            const value = getMetricValueForBanding(item);
            return value > lowerThreshold && value < upperThreshold;
          });
          const lowerBandWords = scoreboard.filter(item => {
            const value = getMetricValueForBanding(item);
            return value <= lowerThreshold;
          });

          const renderBandTable = (bandWords: typeof scoreboard, bandName: string, bandColor: string, bandKey: string) => {
            if (bandWords.length === 0) return null;
            
            const isExpanded = expandedBands[bandKey];
            const defaultLimit = 5; // Show 5 words by default
            const displayWords = isExpanded ? bandWords : bandWords.slice(0, defaultLimit);
            const hasMore = bandWords.length > defaultLimit;
            
            return (
              <div key={bandName} style={{ marginBottom: 20 }}>
                <h4 style={{ 
                  margin: '12px 0 6px 0', 
                  padding: '6px 12px', 
                  backgroundColor: bandColor, 
                  borderRadius: '4px',
                  fontSize: '0.9em',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <span>{bandName} Attention Band ({bandWords.length} words)</span>
                  {hasMore && (
                    <button
                      onClick={() => setExpandedBands(prev => ({
                        ...prev,
                        [bandKey]: !prev[bandKey]
                      }))}
                      style={{
                        fontSize: '12px',
                        padding: '2px 6px',
                        backgroundColor: 'rgba(0,0,0,0.1)',
                        border: '1px solid rgba(0,0,0,0.2)',
                        borderRadius: '3px',
                        cursor: 'pointer'
                      }}
                    >
                      {isExpanded ? `Show less (${defaultLimit})` : `Show all (${bandWords.length})`}
                    </button>
                  )}
                </h4>
                <table style={{borderCollapse: 'collapse', fontSize: '0.95em', width: '100%', border: `2px solid ${bandColor}`, tableLayout: 'fixed'}}>
                  <thead>
                    <tr style={{ backgroundColor: bandColor }}>
                      <th style={{textAlign: 'left', padding: '4px 8px', width: '14%'}}>Word</th>
                      <th style={{textAlign: 'right', padding: '4px 8px', width: '12%'}}>Received Meaning</th>
                      <th style={{textAlign: 'right', padding: '4px 8px', width: '12%'}}>Provided Meaning</th>
                      <th style={{textAlign: 'right', padding: '4px 8px', width: '12%'}}>Constructed Meaning</th>
                      <th style={{textAlign: 'right', padding: '4px 8px', width: '12%'}}>Retrieved Meaning</th>
                      <th style={{textAlign: 'right', padding: '4px 8px', width: '12%'}}>Meaning Recall</th>
                      <th style={{textAlign: 'right', padding: '4px 8px', width: '12%'}}>Recall Δ (%)</th>
                      <th style={{textAlign: 'center', padding: '4px 8px', width: '14%'}}>Mark As</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayWords.map((metric) => (
                      <tr key={metric.index}>
                        <td style={{padding: '4px 8px', fontWeight: 500, textAlign: 'left', width: '14%', wordWrap: 'break-word'}}>{metric.word.replace(/##/g, '')}</td>
                        <td style={{padding: '4px 8px', textAlign: 'right', width: '12%'}}>{metric.normReceived.toFixed(3)}</td>
                        <td style={{padding: '4px 8px', textAlign: 'right', width: '12%'}}>{metric.normProvided.toFixed(3)}</td>
                        <td style={{padding: '4px 8px', textAlign: 'right', width: '12%'}}>{metric.normSum.toFixed(3)}</td>
                        <td style={{padding: '4px 8px', textAlign: 'right', width: '12%'}}>{metric.normRetrieved.toFixed(3)}</td>
                        <td style={{padding: '4px 8px', textAlign: 'right', width: '12%'}}>{metric.normRecalled.toFixed(3)}</td>
                        <td style={{padding: '4px 8px', textAlign: 'right', width: '12%', color: metric.recallDelta > 0 ? '#00AA00' : metric.recallDelta < 0 ? '#AA0000' : '#000'}}>
                          {metric.recallDelta > 0 ? '+' : ''}{metric.recallDelta.toFixed(2)}%
                        </td>
                        <td style={{padding: '4px 8px', textAlign: 'center', width: '14%'}}>
                          <div style={{ display: 'flex', gap: '4px', justifyContent: 'center' }}>
                            <button
                              onClick={() => {
                                if (!setUnknownTokenIndices || !setKnownTokenIndices) return;
                                if (unknownTokenIndices.includes(metric.index)) {
                                  setUnknownTokenIndices(unknownTokenIndices.filter(i => i !== metric.index));
                                } else {
                                  setUnknownTokenIndices([...unknownTokenIndices.filter(i => i !== metric.index), metric.index]);
                                  setKnownTokenIndices(knownTokenIndices.filter(i => i !== metric.index));
                                }
                              }}
                              style={{
                                padding: '2px 6px',
                                fontSize: '11px',
                                backgroundColor: unknownTokenIndices.includes(metric.index) ? '#e6f3ff' : '#f8f8f8',
                                border: unknownTokenIndices.includes(metric.index) ? '1px solid #ff6666' : '1px solid #ddd',
                                borderRadius: '3px',
                                cursor: 'pointer',
                                fontWeight: unknownTokenIndices.includes(metric.index) ? 'bold' : 'normal'
                              }}
                              title="Mark as unknown word"
                            >
                              ?
                            </button>
                            <button
                              onClick={() => {
                                if (!setKnownTokenIndices || !setUnknownTokenIndices) return;
                                if (knownTokenIndices.includes(metric.index)) {
                                  setKnownTokenIndices(knownTokenIndices.filter(i => i !== metric.index));
                                } else {
                                  setKnownTokenIndices([...knownTokenIndices.filter(i => i !== metric.index), metric.index]);
                                  setUnknownTokenIndices(unknownTokenIndices.filter(i => i !== metric.index));
                                }
                              }}
                              style={{
                                padding: '2px 6px',
                                fontSize: '11px',
                                backgroundColor: knownTokenIndices.includes(metric.index) ? '#ccffcc' : '#f8f8f8',
                                border: knownTokenIndices.includes(metric.index) ? '1px solid #66ff66' : '1px solid #ddd',
                                borderRadius: '3px',
                                cursor: 'pointer',
                                fontWeight: knownTokenIndices.includes(metric.index) ? 'bold' : 'normal'
                              }}
                              title="Mark as known word"
                            >
                              ✓
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          };

          return (
            <div style={{ marginTop: 12 }}>
              {renderBandTable(lowerBandWords, 'Lower', bandColors.below, 'lower')}
              {renderBandTable(middleBandWords, 'Middle', bandColors.between, 'middle')}
              {renderBandTable(upperBandWords, 'Upper', bandColors.above, 'upper')}
            </div>
          );
        })()}
      </div>


      {/* Dataset and computed metric selection for coloring */}
      {(() => {

        // Find matching sentence in realData
        let match = null;
        if (typeof realData !== 'undefined' && sentence) {
          match = realData.find((d: any) => d.sentence === sentence);
        }
        // Collect all metric keys from first word
        let datasetMetricKeys: string[] = [];
        let datasetSeries: Record<string, (number|null)[]> = {};
        if (match) {
          datasetMetricKeys = Object.keys(match.words[0]).filter(k => k !== 'word');
          for (const key of datasetMetricKeys) {
            datasetSeries[key] = match.words.map((w: any) => typeof w[key] === 'number' ? w[key] : null);
          }
        }
        // Add computed metrics
        const computedMetrics = {
          normSum: metrics.map(m => m.normSum),
          normReceived: metrics.map(m => m.normReceived),
          normProvided: metrics.map(m => m.normProvided),
          normRetrieved: metrics.map(m => m.normRetrieved),
          normRecalled: metrics.map(m => m.normRecalled),
        };
        const allMetricKeys = [...datasetMetricKeys, ...Object.keys(computedMetrics)];
        const allSeries: Record<string, (number|null)[]> = { ...datasetSeries, ...computedMetrics };

        // Metric selection state
        const [selectedMetric, setSelectedMetric] = React.useState<string>(allMetricKeys[0] || 'normSum');
        // Update selectedMetric if metric keys change
        React.useEffect(() => {
          if (!allMetricKeys.includes(selectedMetric)) {
            setSelectedMetric(allMetricKeys[0] || 'normSum');
          }
        }, [allMetricKeys]);

        // Color array for selected metric
        const selectedMetricArr = allSeries[selectedMetric] || metrics.map(m => m.normSum);
        // Filter out nulls for color scale calculation
        const finiteMetricArr = selectedMetricArr.map(v => v == null ? 0 : v);
        const selectedMetricColorFn = getColorScale(finiteMetricArr);

        // If no dataset metrics found, show a message
        if (datasetMetricKeys.length === 0) {
          return (
            <div style={{margin: '16px 0', color: 'red'}}>
              No dataset metrics found for this sentence. Only computed metrics are available.
            </div>
          );
        }

        return (
          <div style={{margin: '16px 0'}}>
            <label style={{marginRight: 8}}>Color text by metric:</label>
            <select value={selectedMetric} onChange={e => setSelectedMetric(e.target.value)} style={{fontSize: 13, padding: '2px 6px'}}>
              {allMetricKeys.map(key => (
                <option key={key} value={key}>{key}</option>
              ))}
            </select>
            <div style={{marginTop: 12, background: 'white', padding: 20, borderRadius: 6, fontFamily: 'Verdana, sans-serif', fontWeight: 'normal', fontSize: '12px', wordBreak: 'normal', position: 'relative', overflowX: 'auto', overflowY: 'visible'}}>
              <div style={{display: 'flex', flexWrap: 'wrap', columnGap: '0px', rowGap: '10px', lineHeight: '1.8'}}>
                {metrics.map((m, i) => {
                  const isPunctuation = punctuationIndices.includes(i);
                  const isSelected = selectedTokenIndices.includes(i);
                  const value = selectedMetricArr[i];
                  // ...existing code for band, background, fontStyle, etc...
                  return (
                    <span
                      key={m.index}
                      style={{
                        background: isPunctuation ? 'white' : selectedMetricColorFn(value == null ? 0 : value),
                        color: m.isUnknown ? '#00AAFF' : (isPunctuation ? '#000' : '#000'),
                        borderRadius: '3px',
                        textAlign: 'center',
                        padding: isSelected ? '2px 6px' : '4px 8px',
                        display: 'inline-block',
                        fontWeight: 'normal',
                        fontStyle: 'normal',
                        cursor: 'pointer',
                        border: isSelected ? '2px solid #0072B2' : 'none',
                        transition: 'border 0.1s',
                        textDecoration: m.isUnknown ? 'underline wavy #00AAFF' : undefined,
                        whiteSpace: 'nowrap',
                      }}
                      onClick={() => {
                        if (!setSelectedTokenIndices) return;
                        setSelectedTokenIndices(
                          isSelected 
                            ? selectedTokenIndices.filter(idx => idx !== i)
                            : [...selectedTokenIndices, i]
                        );
                      }}
                      title={typeof value === 'number' ? `${selectedMetric}: ${value.toFixed(3)}` : ''}
                    >
                      {m.word.replace(/##/g, '')}
                    </span>
                  );
                })}
              </div>
            </div>
          </div>
        );
      })()}

      {/* Word Attention Heatmap toggle and display moved to bottom */}
      <div style={{marginTop: 32, marginBottom: 16}}>
        <button onClick={() => setShowWordHeatmap(v => !v)} style={{fontSize: 14, padding: '4px 12px'}}>
          {showWordHeatmap ? 'Hide' : 'Show'} Word Attention Heatmap
        </button>
        {showWordHeatmap && (
          <div style={{marginTop: 16}}>
            <b>Word Attention Heatmap (white = low, blue = high):</b>
            <table style={{ borderCollapse: 'collapse', marginTop: 8 }}>
              <thead>
                <tr>
                  <th></th>
                  {displayWords.map((w, i) => (
                    <th key={i} style={{ writingMode: 'vertical-rl', fontSize: '0.9em', padding: '8px 8px', textAlign: 'right' }}>{w.replace(/##/g, '')}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {displayAttention.map((row, i) => (
                  <tr key={i}>
                    <th style={{ textAlign: 'right', fontSize: '0.9em', padding: '2px 8px' }}>{displayWords[i].replace(/##/g, '')}</th>
                    {row.map((val, j) => (
                      <td key={j} style={{ background: getBlueWhiteColor(val, attMin, attMax), width: 60, height: 30, textAlign: 'center', fontSize: '0.8em' }}>
                        {val.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ fontSize: '0.8em', marginTop: 4 }}>
              Each cell shows how much the column word attends to the row word.
            </div>
          </div>
        )}
      </div>

      {/* Correlation Matrix for Normalized Scores */}
      <div style={{ width: '100%', marginTop: '20px' }}>
        <div style={{ textAlign: 'center', margin: '24px 0 8px 0' }}>
          <button onClick={() => setShowCorrelation(v => !v)} style={{marginBottom: 8}}>
            {showCorrelation ? 'Hide' : 'Show'} Meaning Processing Mechanisms Correlation Matrix
          </button>
        </div>
        {showCorrelation && (() => {
          // Calculate correlation matrix for meaning processing mechanisms
          const scoreTypes = ['normReceived', 'normProvided', 'normSum', 'normRetrieved', 'normRecalled'];
          const scoreLabels = ['Received Meaning', 'Provided Meaning', 'Constructed Meaning', 'Retrieved Meaning', 'Meaning Recall'];
          
          // Get data for non-punctuation words only
          const scoreData = scoreTypes.map(scoreType => 
            filteredMetrics.map(metric => metric[scoreType as keyof typeof metric] as number)
          );
          
          // Calculate Pearson correlation coefficient
          const calculateCorrelation = (x: number[], y: number[]): number => {
            if (x.length === 0 || y.length === 0) return 0;
            
            const n = x.length;
            const sumX = x.reduce((a, b) => a + b, 0);
            const sumY = y.reduce((a, b) => a + b, 0);
            const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
            const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
            const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
            
            const numerator = n * sumXY - sumX * sumY;
            const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
            
            return denominator === 0 ? 0 : numerator / denominator;
          };
          
          // Build correlation matrix
          const correlationMatrix = scoreTypes.map((_, i) => 
            scoreTypes.map((_, j) => calculateCorrelation(scoreData[i], scoreData[j]))
          );
          
          // Color function for correlation values (-1 to 1)
          const getCorrelationColor = (corr: number): string => {
            const absCorr = Math.abs(corr);
            if (corr > 0) {
              // Positive correlation: white to blue
              const intensity = Math.round(255 * (1 - absCorr));
              return `rgb(${intensity}, ${intensity}, 255)`;
            } else {
              // Negative correlation: white to red
              const intensity = Math.round(255 * (1 - absCorr));
              return `rgb(255, ${intensity}, ${intensity})`;
            }
          };
          
          return (
            <div style={{ margin: '0 auto 24px auto', width: '100%' }}>
              <table style={{
                margin: '0 auto',
                borderCollapse: 'collapse',
                fontSize: '0.9em',
                border: '2px solid #ddd',
                borderRadius: '8px'
              }}>
                <thead>
                  <tr>
                    <th style={{ padding: '8px 12px', backgroundColor: '#f0f0f0', border: '1px solid #ddd' }}></th>
                    {scoreLabels.map((label, i) => (
                      <th key={i} style={{ 
                        padding: '8px 12px', 
                        backgroundColor: '#f0f0f0', 
                        border: '1px solid #ddd',
                        textAlign: 'center',
                        fontSize: '0.85em'
                      }}>
                        {label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {scoreLabels.map((rowLabel, i) => (
                    <tr key={i}>
                      <td style={{ 
                        padding: '8px 12px', 
                        backgroundColor: '#f0f0f0', 
                        border: '1px solid #ddd',
                        fontWeight: 'bold',
                        fontSize: '0.85em'
                      }}>
                        {rowLabel}
                      </td>
                      {correlationMatrix[i].map((corr, j) => (
                        <td key={j} style={{ 
                          padding: '8px 12px', 
                          border: '1px solid #ddd',
                          textAlign: 'center',
                          backgroundColor: getCorrelationColor(corr),
                          fontWeight: i === j ? 'bold' : 'normal'
                        }}>
                          {corr.toFixed(3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ fontSize: '0.8em', marginTop: '8px', textAlign: 'center', color: '#666' }}>
                Correlation between BERT's understanding mechanisms: 
                <strong>Received/Provided Meaning</strong> (attention flow) vs <strong>Constructed/Retrieved Meaning</strong>.
                <br />
                Values range from -1 (negative correlation, red) to +1 (positive correlation, blue).
              </div>
            </div>
          );
        })()}
      </div>

      {/* Constructed vs Retrieved Meaning Analysis */}
      <div style={{ width: '100%', marginTop: '20px' }}>
        <div style={{ textAlign: 'center', margin: '24px 0 8px 0' }}>
          <button onClick={() => setShowContextAnalysis(v => !v)} style={{marginBottom: 8}}>
            {showContextAnalysis ? 'Hide' : 'Show'} Constructed vs Retrieved Meaning Analysis
          </button>
        </div>
        {showContextAnalysis && (() => {
          // Calculate constructed vs retrieved meaning discrepancies
          const analysisData = filteredMetrics.map(metric => ({
            word: metric.word,
            constructedMeaning: metric.normSum, // How much the model constructs meaning through attention
            retrievedMeaning: metric.normRetrieved, // How much the model retrieves stored meaning (FFN activations)
            discrepancy: Math.abs(metric.normSum - metric.normRetrieved), // Absolute difference between Constructed and Retrieved
            meaningBias: metric.normSum - metric.normRetrieved, // Positive = construction-driven, Negative = retrieval-driven
          }));

          // Sort by discrepancy for analysis
          const sortedByDiscrepancy = [...analysisData].sort((a, b) => b.discrepancy - a.discrepancy);
          
          // Create scatter plot data points
          const maxDiscrepancy = Math.max(...analysisData.map(d => d.discrepancy));
          const scatterSize = 400; // SVG size
          const margin = 40;
          const plotSize = scatterSize - 2 * margin;
          
          return (
            <div style={{ margin: '0 auto 24px auto', width: '100%' }}>
              {/* Explanation */}
              <div style={{ 
                fontSize: '0.9em', 
                marginBottom: '16px', 
                padding: '12px', 
                backgroundColor: '#f8f8f8', 
                borderRadius: '8px',
                color: '#444'
              }}>
                <strong>Conceptual Framework:</strong> This analysis reveals how BERT processes information through two complementary mechanisms:
                <br />
                • <strong style={{color: '#0066cc'}}>Constructed Meaning</strong> (attention): Active construction of understanding through contextual reasoning
                <br />
                • <strong style={{color: '#cc6600'}}>Retrieved Meaning</strong> (FFNs): Accessing stored linguistic knowledge and learned patterns
                <br /><br />
                <strong>Interpretation:</strong>
                <br />
                • <strong>Above diagonal:</strong> Construction-driven (model actively builds meaning from context)
                <br />
                • <strong>Below diagonal:</strong> Retrieval-driven (model accesses stored knowledge patterns)
                <br />
                • <strong>Distance from diagonal:</strong> Strength of the meaning-making preference
              </div>
              
              {/* Scatter Plot */}
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '20px' }}>
                <svg width={scatterSize} height={scatterSize} style={{ border: '1px solid #ddd', borderRadius: '8px' }}>
                  {/* Background */}
                  <rect width={scatterSize} height={scatterSize} fill="#fafafa" />
                  
                  {/* Grid lines */}
                  {[0, 0.25, 0.5, 0.75, 1].map(val => {
                    const pos = margin + val * plotSize;
                    return (
                      <g key={val}>
                        <line x1={margin} y1={pos} x2={scatterSize - margin} y2={pos} stroke="#e0e0e0" strokeWidth="1" />
                        <line x1={pos} y1={margin} x2={pos} y2={scatterSize - margin} stroke="#e0e0e0" strokeWidth="1" />
                      </g>
                    );
                  })}
                  
                  {/* Diagonal line (y = x, balanced context processing) */}
                  <line 
                    x1={margin} 
                    y1={scatterSize - margin} 
                    x2={scatterSize - margin} 
                    y2={margin} 
                    stroke="#999" 
                    strokeWidth="2" 
                    strokeDasharray="5,5"
                  />
                  
                  {/* Data points */}
                  {analysisData.map((point, i) => {
                    const x = margin + point.constructedMeaning * plotSize;
                    const y = scatterSize - margin - point.retrievedMeaning * plotSize; // Flip Y axis
                    const isConstructionDriven = point.meaningBias > 0;
                    const color = isConstructionDriven ? '#0066cc' : '#cc6600';
                    const radius = 3 + (point.discrepancy / maxDiscrepancy) * 4; // Size by discrepancy
                    
                    return (
                      <circle
                        key={i}
                        cx={x}
                        cy={y}
                        r={radius}
                        fill={color}
                        fillOpacity={0.7}
                        stroke={color}
                        strokeWidth="1"
                      >
                        <title>{`${point.word}: Constructed=${point.constructedMeaning.toFixed(3)}, Retrieved=${point.retrievedMeaning.toFixed(3)}, Bias=${point.meaningBias > 0 ? '+' : ''}${point.meaningBias.toFixed(3)}`}</title>
                      </circle>
                    );
                  })}
                  
                  {/* Axis labels */}
                  <text x={scatterSize / 2} y={scatterSize - 5} textAnchor="middle" fontSize="12" fill="#666">
                    Constructed Meaning (Attention-based Reasoning)
                  </text>
                  <text 
                    x="15" 
                    y={scatterSize / 2} 
                    textAnchor="middle" 
                    fontSize="12" 
                    fill="#666" 
                    transform={`rotate(-90, 15, ${scatterSize / 2})`}
                  >
                    Retrieved Meaning (Stored Knowledge)
                  </text>
                  
                  {/* Scale labels */}
                  {[0, 0.25, 0.5, 0.75, 1].map(val => {
                    const pos = margin + val * plotSize;
                    return (
                      <g key={val}>
                        <text x={pos} y={scatterSize - 25} textAnchor="middle" fontSize="10" fill="#888">
                          {val.toFixed(2)}
                        </text>
                        <text x="25" y={scatterSize - margin - val * plotSize + 4} textAnchor="middle" fontSize="10" fill="#888">
                          {val.toFixed(2)}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
              
              {/* Top Discrepancies Table */}
              <div>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '1em' }}>Top Constructed vs Retrieved Meaning Discrepancies</h4>
                <table style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: '0.85em',
                  border: '1px solid #ddd',
                  borderRadius: '6px'
                }}>
                  <thead>
                    <tr style={{ backgroundColor: '#f0f0f0' }}>
                      <th style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'left' }}>Word</th>
                      <th style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'right' }}>Constructed Meaning</th>
                      <th style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'right' }}>Retrieved Meaning</th>
                      <th style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'right' }}>Bias</th>
                      <th style={{ padding: '6px 8px', border: '1px solid #ddd', textAlign: 'center' }}>Meaning Strategy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedByDiscrepancy.slice(0, 8).map((item, i) => (
                      <tr key={i}>
                        <td style={{ padding: '4px 8px', border: '1px solid #ddd', fontWeight: 'bold' }}>
                          {item.word.replace(/##/g, '')}
                        </td>
                        <td style={{ padding: '4px 8px', border: '1px solid #ddd', textAlign: 'right' }}>
                          {item.constructedMeaning.toFixed(3)}
                        </td>
                        <td style={{ padding: '4px 8px', border: '1px solid #ddd', textAlign: 'right' }}>
                          {item.retrievedMeaning.toFixed(3)}
                        </td>
                        <td style={{ 
                          padding: '4px 8px', 
                          border: '1px solid #ddd', 
                          textAlign: 'right',
                          color: item.meaningBias > 0 ? '#0066cc' : '#cc6600',
                          fontWeight: 'bold'
                        }}>
                          {item.meaningBias > 0 ? '+' : ''}{item.meaningBias.toFixed(3)}
                        </td>
                        <td style={{ padding: '4px 8px', border: '1px solid #ddd', textAlign: 'center', fontSize: '0.8em' }}>
                          {item.meaningBias > 0.1 ? '🏗️ Active Construction' : 
                           item.meaningBias < -0.1 ? '📚 Knowledge Retrieval' : 
                           '⚖️ Balanced Processing'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ fontSize: '0.8em', marginTop: '8px', color: '#666' }}>
                  <strong>🏗️ Active Construction:</strong> Model actively builds meaning through attention-based reasoning
                  <br />
                  <strong>📚 Knowledge Retrieval:</strong> Model accesses stored linguistic knowledge and learned patterns
                  <br />
                  <strong>⚖️ Balanced Processing:</strong> Model uses both construction and retrieval equally
                </div>
              </div>
            </div>
          );
        })()}
      </div>

      {/* Attention Score Formulas Toggle moved to bottom */}
      <div style={{ textAlign: 'center', margin: '24px 0 8px 0' }}>
        <button onClick={() => setShowFormulas(v => !v)} style={{marginBottom: 8}}>
          {showFormulas ? 'Hide' : 'Show'} Attention Score Formulas
        </button>
      </div>
      {showFormulas && (
        <div style={{margin: '0 auto 24px auto', width: '100%'}}>
          <table style={{
            margin: '0 auto',
            borderCollapse: 'collapse',
            fontSize: '1em',
            width: '100%',
            background: '#f8f8f8',
            borderRadius: 8,
            boxSizing: 'border-box',
            padding: 0
          }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '6px 12px', width: '18%' }}>Score</th>
                <th style={{ textAlign: 'left', padding: '6px 12px', width: '32%' }}>Formula</th>
                <th style={{ textAlign: 'left', padding: '6px 12px' }}>Explanation</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}><b>Total Provided</b> (for word <i>j</i>)</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>total_provided_j = Σ(k=1 to N) A_jk</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>Sum of row <i>j</i> in the heatmap. How much attention word <b>j</b> gives to all words (including itself).</td>
              </tr>
              <tr>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}><b>Total Received</b> (for word <i>i</i>)</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>total_received_i = Σ(k=1 to N) A_ki</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>Sum of column <i>i</i> in the heatmap. How much attention word <b>i</b> receives from all words.</td>
              </tr>
              <tr>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}><b>Total Left</b> (for word <i>i</i>)</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>total_left_i = Σ(j=1 to i-1) A_ij</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>Attention word <b>i</b> gives to words to its left (lower column indices).</td>
              </tr>
              <tr>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}><b>Total Right</b> (for word <i>i</i>)</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>total_right_i = Σ(j=i+1 to N) A_ij</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>Attention word <b>i</b> gives to words to its right (higher column indices).</td>
              </tr>
              <tr>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}><b>Heatmap</b> <i>A<sub>ij</sub></i></td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>A_ij = Σ(l=1 to L) Σ(h=1 to H) Attention^(l,h)_ij</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>Sum of attention from word <b>j</b> to word <b>i</b> over all layers <i>L</i> and heads <i>H</i> (matrix cell in the heatmap).</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default AttentionHeatmap;
