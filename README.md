# Pandu — Intelligent Computational Archaeological Workspace

Pandu is a desktop application for computational archaeology, designed to help researchers, epigraphists, and historians catalog, analyze, and translate ancient scripts and artifacts using both local and cloud-based AI models.

## Features

### Artifact Database
- **Data Entry**: Add artifacts with metadata (name, writing system, time period, region, source) and associated images via drag-and-drop or file browser
- **Library**: Browse, edit, search, and manage all cataloged artifacts in a table view
- **Database Relocation**: Move or copy your database to custom directories

### AI-Powered Analysis
- **Dual AI Mode**: Switch between local (Ollama) and cloud (Gemini, OpenAI, Anthropic, Mistral) AI models
- **Image Analysis**: Select artifacts from your library and run AI-powered analysis with custom prompts
- **Script Analysis**: Drag-and-drop script images for automatic transcription, translation, and attribute detection
- **Cloud Chat**: Interactive chat interface with cloud AI providers
- **Local Chat**: Chat with locally-hosted models via Ollama

### Training & Pattern Recognition
- **Script Training**: Train AI models on your artifact data to recognize writing systems, letterforms, and transformation patterns
- **Trained Models**: Save, activate, and manage trained script models
- **Statistical Charts**: Visual confidence match percentages for writing system, glyph, period, region, and material

### PDF Report Generation
Generate comprehensive PDF reports containing:
- Full script translation
- Probable attributes (name, writing system, time period, region, source)
- Statistical match percentages with bar chart visualization
- AI reasoning and decision explanation

### Security
- **API keys are stored in memory only** — never written to disk
- On application restart, you must re-enter your API key
- Temporary file permission management with automatic revocation
- Security warnings before sending data to third-party APIs

### Additional Tools
- Built-in terminal for shell commands
- Permission manager for accessing protected files
- Snap-to-screen-edge window management

## Requirements

- Python 3.8+
- PyQt6
- requests
- fpdf2
- google-genai (for Gemini cloud support)
- Ollama (for local AI models)

## Installation

```bash
# Clone the repository
git clone https://github.com/ahmednirjhor29-stack/Pandu.git
cd Pandu

# Install dependencies
pip install PyQt6 requests fpdf2 google-genai

# (Optional) Install Ollama for local AI models
# Visit https://ollama.ai for installation instructions
```

## Usage

```bash
python "Pandu v1.7.py"
```

### Quick Start
1. Launch the application
2. Use **Data Entry** to add artifacts with images
3. Browse your collection in **Library**
4. Switch to **AI Analysis** for AI-powered analysis
5. Use **Script Analysis** for drag-and-drop script translation
6. Train custom models in **Train**

### Cloud AI Setup
1. Go to **AI Analysis** → **Cloud** mode
2. Select your provider (Gemini, OpenAI, Anthropic, Mistral)
3. Enter your API key (stored in memory only for the session)
4. Select a model and click **Use**

### Local AI Setup
1. Install and run [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull llama3`
3. In **AI Analysis** → **Local** mode, the model will appear automatically
4. Click **Target** to activate it

## License

Copyright (C) 2026 Ahmed Nirjhor

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.