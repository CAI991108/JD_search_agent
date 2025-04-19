# 🛍️ JD Search Agent (Demo)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=)
![Gradio Demo](https://img.shields.io/badge/🤗-Power%20by%20Gradio-blue)

Your smart shopping companion for JD.com, powered by multimodal generative AI! 🚀

🤗 For more details breakdown and implementation,please refer to our [Kaggle Notebook](https://www.kaggle.com/code/zijincai/jd-search-agent)
[<img src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/kaggle.svg" width="20"> ]([https://kaggle.com](https://www.kaggle.com/code/zijincai/jd-search-agent))


## ✨ Key Features

- **🤖 Intelligent Product Search**  
  Natural language queries with dynamic requirement clarification
- **👁️ Multimodal Understanding**  
  Combines text search + image OCR + structured data analysis
- **💰 Smart Price Optimization**  
  Auto-detects government subsidies & promotions
- **📊 Comparative Analysis**  
  Generates product comparison tables with reasoning
- **🔄 Stateful Conversations**  
  Maintains context across multi-turn interactions

![Demo Screenshot](demo-screenshot.gif)

## 🧠 Tech Stack

### Core Components
![LangGraph](https://img.shields.io/badge/LangGraph-0.1.0-FF6F00?logo=langchain)
![Playwright](https://img.shields.io/badge/Playwright-1.42.0-2C974B?logo=playwright)
![Gemini](https://img.shields.io/badge/Gemini--Flash-2.0-4285F4?logo=google)
![Qwen-VL](https://img.shields.io/badge/Qwen--VL-2.5-FF6C37)

### Architecture

```mermaid[]
[User Query]
│
▼    
[Intent Analysis]: 
│
├─[Yes]─▶ [JD Product Search] ─▶ [Multimodal Processing]
│                                    │
└─[No]──▶ [Knowledge Base Query] ────┘
▼
[Recommendation Generation]
│
▼
[Structured Output]
```
## 🚀 Getting Started

### Prerequisites
```bash
pip install -qU \
  openai==1.70.0 \
  langchain-core==0.3.50 \
  langchain-google-genai==2.0.8 \
  langgraph==0.3.25 \
  langgraph-prebuilt==0.1.8 \
  playwright==1.51.0 \
  nest-asyncio==1.6.0 \
  typing-extensions==4.13.1

pip install playwright==1.51.0 -qU
PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 playwright install chromium
```

* Python 3.9+
* Playwright browsers: `playwright install chromium`
* Get API Keys: [Google Gemini](https://aistudio.google.com/app/apikey) and [OpenRouter](https://openrouter.ai/settings/keys)
    - remember to replace with your own `GOOGLE_API_KEY` and `OPENAI_API_KEY` in the `.py` files
* Prepare yourt JD cookies (run locally: `login_and_save_cookie.py`)
    - Follow prompts to login to JD.com
    - Cookies will auto-save to `jd_cookies.json `

### Launch Demo
```bash
python Gradio_UI.py
```

Access the Gradio UI at `http://localhost:7861`


## 🌟 Example Interaction

```bash
User:  Find thin gaming laptops under 15000 with RTX 4060
Bot: 🔍 Searching JD.com... Found 3 options!
     1. ROG Zephyrus G14 (¥14,999)
     2. Lenovo Legion 5 Pro (¥13,888)
     3. HP Victus 16 (¥12,999)
     [Comparison table generated]
```

| Input Type | Original Data | AI-Enhanced Data | Information Gain |
|---|---|---|---|
| Product Image | <img src="https://img13.360buyimg.com/n7/jfs/t1/227056/31/37988/142800/67c55ae5F69b793b7/3516f56adfdb1d35.jpg" width=100> | <div style="max-width:200px">爆款抢购<br>官方正品<br>幻X 2025<br>AI MAX 395<br>8000S显卡</div> | +83% Key Information |
| User Query | "Thin and light laptop with dedicated graphics" | <div style="color:#e74c3c">Requirement Analysis:<br>- Graphics Card Type: RTX 4060+<br>- Weight: <1.8kg<br>- Budget: 8000-12000</div> | +62% Semantic Understanding |


<br>

**📸 Demo Screenshots**

<div align="center" style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
  <img src="/assets/demo1.jpg" alt="Demo 1 - Product Search" width="800">
</div>

<div align="center" style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
  <img src="/assets/demo2.jpg" alt="Demo 2 - Comparison Table" width="800">
</div>



<details>
<summary>📷 Click to see image descriptions</summary>

- **top**: Real-time product search interface  
- **bottow**: Interactive comparison table with purchase links
</details>



## 📜 License
This project is licensed under the MIT License - see the [LICENSE](license) file for details.
