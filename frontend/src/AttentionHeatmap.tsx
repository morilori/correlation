import React, { useState, useEffect } from 'react';

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
  punctuationIndices = []
}) => {
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

  const [scoreSortMetric, setScoreSortMetric] = useState<'received' | 'provided' | 'normSum'>('normSum');
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
  const normalizedSums = rawSums.map((sum, i) => 
    punctuationIndices.includes(i) 
      ? 0  // Set punctuation normSum to 0
      : (maxRawSum - minRawSum ? (sum - minRawSum) / (maxRawSum - minRawSum) : 0)
  );
  
  // Compute metrics for each word position ONCE, in original order
  const metrics = wordObjs.map(({ word, index }, i) => ({
    word,
    index,
    received: totalReceived[index],
    provided: totalProvided[index],
    normProvided: normProvided[index],
    normReceived: normReceived[index],
    normSum: normalizedSums[i],
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
            <select value={scoreSortMetric} onChange={e => setScoreSortMetric(e.target.value as 'received' | 'provided' | 'normSum')} style={{fontSize: 13, padding: '2px 6px'}}>
              <option value="received">Norm. Received</option>
              <option value="provided">Norm. Provided</option>
              <option value="normSum">Norm. Sum</option>
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
                  `Norm. ${scoreSortMetric}: ${(m as any)[scoreSortMetric === 'normSum' ? 'normSum' : 
                                                            scoreSortMetric === 'provided' ? 'normProvided' : 'normReceived'].toFixed(3)}`
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
        <b>Scoreboard: Normalized Scores by Attention Bands | Text Score (mean norm sum per word): {textScore.toFixed(3)}</b>
        
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
                      <th style={{textAlign: 'left', padding: '4px 8px', width: '25%'}}>Word</th>
                      <th style={{textAlign: 'right', padding: '4px 8px', width: '20%'}}>Norm. Received</th>
                      <th style={{textAlign: 'right', padding: '4px 8px', width: '20%'}}>Norm. Provided</th>
                      <th style={{textAlign: 'right', padding: '4px 8px', width: '20%'}}>Norm. Sum</th>
                      <th style={{textAlign: 'center', padding: '4px 8px', width: '15%'}}>Mark As</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayWords.map(({ word, normProvided, normReceived, normSum, index }) => (
                      <tr key={index}>
                        <td style={{padding: '4px 8px', fontWeight: 500, textAlign: 'left', width: '25%', wordWrap: 'break-word'}}>{word.replace(/##/g, '')}</td>
                        <td style={{padding: '4px 8px', textAlign: 'right', width: '20%'}}>{normReceived.toFixed(3)}</td>
                        <td style={{padding: '4px 8px', textAlign: 'right', width: '20%'}}>{normProvided.toFixed(3)}</td>
                        <td style={{padding: '4px 8px', textAlign: 'right', width: '20%'}}>{normSum.toFixed(3)}</td>
                        <td style={{padding: '4px 8px', textAlign: 'center', width: '15%'}}>
                          <div style={{ display: 'flex', gap: '4px', justifyContent: 'center' }}>
                            <button
                              onClick={() => {
                                if (!setUnknownTokenIndices || !setKnownTokenIndices) return;
                                if (unknownTokenIndices.includes(index)) {
                                  setUnknownTokenIndices(unknownTokenIndices.filter(i => i !== index));
                                } else {
                                  setUnknownTokenIndices([...unknownTokenIndices.filter(i => i !== index), index]);
                                  setKnownTokenIndices(knownTokenIndices.filter(i => i !== index));
                                }
                              }}
                              style={{
                                padding: '2px 6px',
                                fontSize: '11px',
                                backgroundColor: unknownTokenIndices.includes(index) ? '#e6f3ff' : '#f8f8f8',
                                border: unknownTokenIndices.includes(index) ? '1px solid #ff6666' : '1px solid #ddd',
                                borderRadius: '3px',
                                cursor: 'pointer',
                                fontWeight: unknownTokenIndices.includes(index) ? 'bold' : 'normal'
                              }}
                              title="Mark as unknown word"
                            >
                              ?
                            </button>
                            <button
                              onClick={() => {
                                if (!setKnownTokenIndices || !setUnknownTokenIndices) return;
                                if (knownTokenIndices.includes(index)) {
                                  setKnownTokenIndices(knownTokenIndices.filter(i => i !== index));
                                } else {
                                  setKnownTokenIndices([...knownTokenIndices.filter(i => i !== index), index]);
                                  setUnknownTokenIndices(unknownTokenIndices.filter(i => i !== index));
                                }
                              }}
                              style={{
                                padding: '2px 6px',
                                fontSize: '11px',
                                backgroundColor: knownTokenIndices.includes(index) ? '#ccffcc' : '#f8f8f8',
                                border: knownTokenIndices.includes(index) ? '1px solid #66ff66' : '1px solid #ddd',
                                borderRadius: '3px',
                                cursor: 'pointer',
                                fontWeight: knownTokenIndices.includes(index) ? 'bold' : 'normal'
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
