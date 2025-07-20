# BERT Attention Visualization

A clean, interactive web application for visualizing BERT attention patterns. This tool helps researchers and developers understand how BERT models attend to different words in a text sequence.

## 🎯 Features

- **Real-time BERT Attention Analysis**: Process any text and see BERT's attention patterns
- **Interactive Heatmap**: Click on words to explore attention relationships
- **Multiple Metrics**: View provided/received attention, directional attention (left/right)
- **Text Grouping**: Handle long texts by processing them in sentence groups
- **Interactive Features**:
  - Multiple word selection with visual highlighting
  - Attention score filtering and normalization
  - Exclude punctuation from calculations (always displayed)
- **Word Marking**: Mark words as known/unknown to analyze comprehension patterns

## 🏗️ Architecture

- **Backend**: FastAPI (Python) with BERT transformers
- **Frontend**: React (TypeScript) with Vite
- **Model**: BERT-base-uncased for attention analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation & Running

1. **Clone the repository**
   ```bash
   git clone https://github.com/morilori/bert-attention-visualization.git
   cd bert-attention-visualization
   ```

2. **Start the application**
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

3. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Manual Setup

If you prefer to set up manually:

#### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 📊 How It Works

1. **Input Text**: Enter any text in the input field
2. **BERT Processing**: The backend uses BERT-base-uncased to generate attention matrices
3. **Visualization**: The frontend displays an interactive heatmap where:
   - Rows represent words giving attention
   - Columns represent words receiving attention
   - Color intensity shows attention strength
4. **Analysis**: Explore attention patterns through various metrics and interactions

## 🔧 API Endpoints

### POST `/attention`
Analyzes text and returns BERT attention data.

**Request Body:**
```json
{
  "text": "Your text here"
}
```

**Response:**
```json
{
  "tokens": ["word1", "word2", ...],
  "attention": [[attention_matrix]]
}
```

## 📁 Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application
│   │   └── attention.py     # BERT attention processing
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main React component
│   │   ├── AttentionHeatmap.tsx  # Attention visualization
│   │   └── attentionUtils.ts     # Utility functions
│   └── package.json         # Node.js dependencies
├── start.sh                 # Quick start script
└── README.md
```

## 🧠 Understanding the Metrics

- **Total Provided**: How much attention a word gives to all other words
- **Total Received**: How much attention a word receives from all other words
- **Total Left**: Attention a word gives to words appearing before it
- **Total Right**: Attention a word gives to words appearing after it
- **Norm Sum**: Balanced combination of provided and received attention

## 🎨 Customization

The application supports various filtering and display options:

- **Punctuation Filtering**: Exclude punctuation from attention calculations (punctuation is always displayed)

*Note: Self-attention (diagonal values), immediate neighbors, and word piece combination are always included in score calculations.*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test them
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers for BERT implementation
- The BERT team at Google for the original model
- FastAPI and React communities for excellent frameworks

## 📚 Research Applications

This tool is useful for:
- Understanding BERT's attention mechanisms
- Analyzing text comprehension patterns
- Educational purposes in NLP courses
- Research in transformer interpretability
- Debugging attention-based models

## 🐛 Troubleshooting

**Port conflicts**: If ports 8000 or 5173 are in use, the start script will attempt to free them.

**Memory issues**: BERT processing requires significant memory. For large texts, the application automatically groups sentences.

**Model download**: On first run, BERT-base-uncased will be downloaded (~400MB).

## 📧 Support

For questions, issues, or contributions, please open an issue on GitHub.
