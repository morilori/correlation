// Data loader for CSV file
class DataLoader {
    constructor() {
        this.sentences = [];
        this.loaded = false;
    }

    async loadData() {
        try {
            const response = await fetch('/all_measures.csv');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const csvText = await response.text();
            this.sentences = this.parseCSV(csvText);
            this.loaded = true;
            console.log(`Successfully loaded ${this.sentences.length} sentences from CSV`);
            return this.sentences;
        } catch (error) {
            console.error('Error loading CSV data:', error);
            throw error; // Don't fall back to sample data, let the application handle the error
        }
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        
        // Group by sentence
        const sentenceMap = new Map();
        
        // Process all lines (not just a subset)
        for (let i = 1; i < lines.length; i++) {
            const values = this.parseCSVLine(lines[i]);
            if (values.length !== headers.length) continue; // Skip malformed lines
            
            const row = {};
            
            // Parse each column
            headers.forEach((header, index) => {
                const value = values[index];
                if (value === '' || value === 'NA' || value === 'NULL') {
                    row[header] = null;
                } else if (!isNaN(value) && value !== '') {
                    row[header] = parseFloat(value);
                } else {
                    row[header] = value;
                }
            });
            
            const sentenceText = row.sentence;
            if (!sentenceText) continue; // Skip rows without sentence
            
            if (!sentenceMap.has(sentenceText)) {
                sentenceMap.set(sentenceText, {
                    sentence: sentenceText,
                    words: [],
                    firstWord: null // Track the missing first word
                });
            }
            
            // Extract first word from the sentence if not already done
            if (!sentenceMap.get(sentenceText).firstWord) {
                const firstWord = sentenceText.split(' ')[0];
                sentenceMap.get(sentenceText).firstWord = firstWord;
            }
            
            // Add word data with position
            const metrics = this.extractMetrics(row);
            // Promote aliasing for specific known dotted keys used in UI
            if (metrics['s_GPTNeo_2_7B'] == null && metrics['s_GPTNeo_2.7B'] != null) {
                metrics['s_GPTNeo_2_7B'] = metrics['s_GPTNeo_2.7B'];
            }
            sentenceMap.get(sentenceText).words.push({
                word: row.word,
                position: row.context_length || 0,
                ...metrics
            });
        }
        
        // Return ALL sentences (no artificial limit) and sort words by position
        // Also reconstruct complete sentences by adding the missing first word
        const allSentences = Array.from(sentenceMap.values()).map(sentenceData => {
            const sortedWords = sentenceData.words.sort((a, b) => a.position - b.position);
            
            // Add the missing first word at position 0 with no data
            const firstWordData = {
                word: sentenceData.firstWord,
                position: 0,
                // All metrics are null for the first word since it was excluded from measurements
                ...Object.fromEntries(
                    Object.keys(this.extractMetrics(sortedWords[0] || {})).map(key => [key, null])
                )
            };
            
            return {
                ...sentenceData,
                words: [firstWordData, ...sortedWords]
            };
        });
        console.log(`Parsed ${allSentences.length} unique sentences from ${lines.length - 1} data rows`);
        return allSentences;
    }

    parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                result.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        
        result.push(current.trim()); // Add the last field
        return result;
    }

    extractMetrics(row) {
        // Extract all metric columns (excluding metadata)
        const excludeColumns = ['item', 'word', 'word2', 'sentence', 'context_length', 'sent_id', 'item_id', 'list', 'is_start_end', 'is_multitoken_GPT2', 'is_multitoken_GPT2_medium', 'is_multitoken_GPT2_large', 'is_multitoken_GPT2_xl', 'is_multitoken_GPTNeo_125M', 'is_multitoken_GPTNeo', 'is_multitoken_GPTNeo_2.7B'];
        
        const metrics = {};
        Object.keys(row).forEach(key => {
            if (!excludeColumns.includes(key)) {
                metrics[key] = row[key];
                // Also add underscore alias for keys that contain dots (e.g., s_GPTNeo_2.7B)
                if (key.includes('.')) {
                    const alias = key.replace(/\./g, '_');
                    metrics[alias] = row[key];
                }
            }
        });
        
        return metrics;
    }
}

// Global instance
const dataLoader = new DataLoader();
