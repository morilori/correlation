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
  const originalMinReceived = Math.min(...originalReceived);
  const originalMaxReceived = Math.max(...originalReceived);

  // For normProvided, exclude unknowns from min/max and set their normProvided=0
  const nonUnknownIndices = words.map((_, i) => i).filter(i => !unknownTokenIndices.includes(i));
  const minProvided = nonUnknownIndices.length > 0 ? Math.min(...nonUnknownIndices.map(i => provided[i])) : 0;
  const maxProvided = nonUnknownIndices.length > 0 ? Math.max(...nonUnknownIndices.map(i => provided[i])) : 1;
  let normProvided = provided.map((v, i) =>
    unknownTokenIndices.includes(i)
      ? 0
      : (maxProvided - minProvided ? (v - minProvided) / (maxProvided - minProvided) : 0)
  );
  // For normReceived, use the original min/max (before zeroing unknowns)
  let normReceived = received.map(v => (originalMaxReceived - originalMinReceived ? (v - originalMinReceived) / (originalMaxReceived - originalMinReceived) : 0));

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
  
  // Calculate raw sums of original received and provided scores
  const rawSums = wordObjs.map(({ index }) => totalReceived[index] + totalProvided[index]);
  
  // Normalize the raw sums to 0-1 range
  const minRawSum = Math.min(...rawSums);
  const maxRawSum = Math.max(...rawSums);
  const normalizedSums = rawSums.map(sum => 
    (maxRawSum - minRawSum ? (sum - minRawSum) / (maxRawSum - minRawSum) : 0)
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
  // Band colors: above=blue, between=light blue, below=white
  const bandColors = {
    above: '#00AAFF', // strong blue (highest attention)
    between: '#cce6ff', // light blue (middle)
    below: '#fff', // white (lowest attention)
  };

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

        <div style={{marginTop: 8, background: '#f8f8f8', padding: 20, borderRadius: 6, fontFamily: 'monospace', wordBreak: 'normal', position: 'relative', overflowX: 'auto', overflowY: 'visible'}}>
          {(() => {
            // Constants
            const BAND_HEIGHT_EM = 2.5;
            const ROWS_PER_LINE = 3;
            const LINE_GAP_EM = 4;
            const WORD_GAP = 8;
            
            // Helper functions
            const isSentenceEnd = (word: string) => /[.!?]$/.test(word.trim());
            const getWordWidth = (word: string, index: number) => {
              const displayWord = isSentenceStart(index) ? capitalizeWord(word) : word;
              return displayWord.length * 9 + 14;
            };
            
            // Helper function to determine if a word should be capitalized (sentence beginning)
            const isSentenceStart = (index: number): boolean => {
              if (index === 0) return true; // First word is always sentence start
              // Check if previous word ends with sentence punctuation
              for (let i = index - 1; i >= 0; i--) {
                const prevWord = metrics[i].word;
                if (isSentenceEnd(prevWord)) return true;
                // If we hit a non-punctuation word, this is not a sentence start
                if (!/^[^\w]*$/.test(prevWord)) return false;
              }
              return false;
            };
            
            // Helper function to capitalize first letter of a word
            const capitalizeWord = (word: string): string => {
              if (!word || word.length === 0) return word;
              return word.charAt(0).toUpperCase() + word.slice(1);
            };
            
            // Pre-calculate all word positions and line data in a single pass
            const wordPositions: Array<{
              lineNumber: number;
              leftOffset: number;
              verticalPosition: number;
              band: 'above' | 'between' | 'below';
            }> = [];
            const lineWidths: number[] = [];
            
            let currentLine = 0;
            let currentLineOffset = 0;
            let maxLineNumber = 0;
            
            // Single pass calculation
            metrics.forEach((m, i) => {
              const wordWidth = getWordWidth(m.word, i);
              const isPunctuation = punctuationIndices.includes(i);
              const value = originalTextColorArr[i];
              
              // Determine position and band
              let verticalPosition = 1;
              let band: 'above' | 'between' | 'below' = 'between';
              
              if (!isPunctuation) {
                if (value >= upperThreshold) {
                  verticalPosition = 0;
                  band = 'above';
                } else if (value <= lowerThreshold) {
                  verticalPosition = 2;
                  band = 'below';
                }
              }
              
              wordPositions.push({
                lineNumber: currentLine,
                leftOffset: currentLineOffset,
                verticalPosition,
                band
              });
              
              currentLineOffset += wordWidth + WORD_GAP;
              lineWidths[currentLine] = Math.max(lineWidths[currentLine] || 0, currentLineOffset);
              
              if (isSentenceEnd(m.word)) {
                maxLineNumber = Math.max(maxLineNumber, currentLine);
                currentLine++;
                currentLineOffset = 0;
              }
            });
            
            const totalHeightEm = (maxLineNumber + 1) * (BAND_HEIGHT_EM * ROWS_PER_LINE) + maxLineNumber * LINE_GAP_EM;
            
            // Generate band elements efficiently
            const bands: React.ReactElement[] = [];
            for (let l = 0; l <= maxLineNumber; l++) {
              const lineOffsetEm = l * (BAND_HEIGHT_EM * ROWS_PER_LINE + LINE_GAP_EM);
              const width = Math.max(lineWidths[l] || 200, 200);
              
              const bandConfigs = [
                { top: lineOffsetEm, style: { border: '1.5px solid #bbb', borderRadius: '6px 6px 0 0' } },
                { top: lineOffsetEm + BAND_HEIGHT_EM, style: { borderLeft: '1.5px solid #bbb', borderRight: '1.5px solid #bbb' } },
                { top: lineOffsetEm + 2 * BAND_HEIGHT_EM, style: { border: '1.5px solid #bbb', borderRadius: '0 0 6px 6px' } }
              ];
              
              bandConfigs.forEach((config, idx) => {
                bands.push(
                  <div key={`${l}-${idx}`} style={{
                    position: 'absolute',
                    left: 0,
                    width: `${width}px`,
                    top: `${config.top}em`,
                    height: `${BAND_HEIGHT_EM}em`,
                    background: 'none',
                    zIndex: 10,
                    boxSizing: 'border-box',
                    pointerEvents: 'none',
                    ...config.style
                  }} />
                );
              });
            }
            
            return (
              <div style={{position: 'relative', minHeight: `${totalHeightEm}em`, width: 'max-content', lineHeight: '1.8'}}>
                {bands}
                {metrics.map((m, i) => {
                  const pos = wordPositions[i];
                  const wordWidth = getWordWidth(m.word, i);
                  const topEm = pos.verticalPosition * BAND_HEIGHT_EM + pos.lineNumber * (BAND_HEIGHT_EM * ROWS_PER_LINE + LINE_GAP_EM);
                  const isPunctuation = punctuationIndices.includes(i);
                  const isSelected = selectedTokenIndices.includes(i);
                  
                  // Simplified background calculation
                  const background = isPunctuation 
                    ? '#cce6ff'  // Same as middle band color
                    : colorByBand 
                      ? bandColors[pos.band]
                      : originalTextColorScale(originalTextColorArr[i]);
                  
                  // Optimized tooltip generation
                  const bandLabel = pos.band === 'above' ? 'above threshold' : 
                                   pos.band === 'below' ? 'below threshold' : 'within thresholds';
                  
                  const tooltipParts = [
                    `Norm sum: ${m.normSum.toFixed(3)} (${bandLabel})`,
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
                        position: 'absolute',
                        left: `${pos.leftOffset}px`,
                        top: `${topEm}em`,
                        background,
                        color: m.isUnknown ? '#00AAFF' : '#000',
                        borderRadius: 4,
                        width: `${wordWidth}px`,
                        textAlign: 'center',
                        padding: '3px 0',
                        display: 'inline-block',
                        fontWeight: 500,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        cursor: 'pointer',
                        border: isSelected ? '2px solid #0072B2' : '2px solid transparent',
                        boxShadow: isSelected ? '0 0 4px #0072B2' : undefined,
                        transition: 'border 0.1s, box-shadow 0.1s, top 0.3s ease',
                        textDecoration: m.isUnknown ? 'underline wavy #00AAFF' : undefined,
                        zIndex: 2,
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
                      {isSentenceStart(i) ? capitalizeWord(m.word) : m.word}
                    </span>
                  );
                })}
              </div>
            );
          })()}
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
                        <td style={{padding: '4px 8px', fontWeight: 500, textAlign: 'left', width: '25%', wordWrap: 'break-word'}}>{word}</td>
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
                    <th key={i} style={{ writingMode: 'vertical-rl', fontSize: '0.9em', padding: '2px 4px' }}>{w}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {displayAttention.map((row, i) => (
                  <tr key={i}>
                    <th style={{ textAlign: 'right', fontSize: '0.9em', padding: '2px 4px' }}>{displayWords[i]}</th>
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
