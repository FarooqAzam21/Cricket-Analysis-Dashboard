# Ollama Setup Guide for Cricket Analysis Dashboard

## 🚀 Quick Start

The AI Chat feature now works in two modes:

### **Mode 1: With Ollama (Recommended for Better Responses)**
Full LLM-powered responses with cricket context awareness.

### **Mode 2: Fallback Rule-Based (Automatic)**
If Ollama is unavailable, the system automatically uses intelligent rule-based responses.

---

## Installation & Setup

### **1. Install Ollama**

#### **Windows:**
- Download from: https://ollama.ai/download/windows
- Run the installer
- Ollama will start automatically

#### **Mac:**
- Download from: https://ollama.ai/download/mac
- Drag to Applications folder

#### **Linux:**
```bash
curl https://ollama.ai/install.sh | sh
```

### **2. Pull a Cricket-Friendly Model**

After Ollama is installed, download a model:

#### **Option A: Llama 2 (Recommended - ~4GB)**
```bash
ollama pull llama2
```

#### **Option B: Mistral (Smaller & Faster - ~5GB)**
```bash
ollama pull mistral
```

#### **Option C: Neural Chat (Lightweight - ~2GB)**
```bash
ollama pull neural-chat
```

### **3. Start Ollama Service**

#### **Windows/Mac:**
- Ollama runs in the background after installation
- Check: http://localhost:11434 (should show "Ollama is running")

#### **Linux:**
```bash
ollama serve
```

---

## How It Works

### **With Ollama Running ✅**

```
User: "Who is the best batsman?"
  ↓
Cricket Dashboard sends prompt with player data context
  ↓
Ollama LLM analyzes context and generates response
  ↓
Response: "Based on the data, [Player Name] is the top batsman with..."
```

### **Without Ollama (Fallback) ⚙️**

```
User: "Who is the best batsman?"
  ↓
Ollama connection fails
  ↓
System automatically uses rule-based responses
  ↓
Response: "Based on the data, the batsmen with the highest runs are..."
```

---

## Supported Models

| Model | Size | Speed | Quality | Command |
|-------|------|-------|---------|---------|
| Llama 2 | 7B | Medium | Excellent | `ollama pull llama2` |
| Mistral | 7B | Fast | Good | `ollama pull mistral` |
| Neural Chat | 7B | Fast | Good | `ollama pull neural-chat` |

**Note**: The dashboard auto-detects available models and uses the first one it finds.

---

## Troubleshooting

### **Chat Feature Shows Errors**

**1. "Ollama is not running"**
- Start Ollama service:
  - **Windows**: Ollama icon in system tray → Restart
  - **Mac**: Restart from Ollama app
  - **Linux**: `ollama serve` in terminal

**2. "Model not found"**
- Pull a model: `ollama pull llama2`
- Verify: `ollama list`

**3. Connection refused (localhost:11434)**
- Check if Ollama is running: 
  ```bash
  curl http://localhost:11434
  ```
- Expected response: "Ollama is running"

**4. Slow responses**
- Reduce model size:
  ```bash
  ollama pull neural-chat
  ```
- Or wait for model to load (first run may take 30 seconds)

---

## Testing the Chat Feature

1. Open Cricket Analysis Dashboard
2. Navigate to **"Ask Expert (AI)"** menu
3. Try these test prompts:
   - "Who is the best batsman?"
   - "Compare top bowlers"
   - "What's the average strike rate?"

### **Expected Behavior**

- ✅ **With Ollama**: Detailed, contextual responses
- ✅ **Without Ollama**: Good rule-based responses (still useful!)
- ✅ **Always**: No crashes or errors

---

## Performance Tips

### **Faster Response Times**
1. Use smaller models:
   ```bash
   ollama pull neural-chat
   ```

2. Increase Ollama memory (if lagging):
   - On multi-GPU systems, Ollama auto-detects
   - For CPU-only: Ensure adequate RAM available

3. Disable other applications to free system resources

### **Better Responses**
1. Use larger models:
   ```bash
   ollama pull mistral
   ```

2. Provide clear context in your questions
3. Reference specific players in questions

---

## Automatic Model Detection

The system tries models in this order:
1. **llama2** (if available)
2. **mistral** (if available)
3. **neural-chat** (if available)

If none are available, it uses rule-based responses.

---

## FAQ

**Q: Does the chat feature require Ollama?**
> A: No! The fallback mode automatically activates if Ollama is unavailable. You'll still get useful responses.

**Q: Which model should I use?**
> A: For general use, **llama2** is best. For faster responses, use **mistral** or **neural-chat**.

**Q: Can I run multiple models?**
> A: Yes, but the system uses the first available model. Run: `ollama list`

**Q: Is my data sent to the cloud?**
> A: No! Ollama runs locally on your machine. All processing is private.

**Q: How much disk space do I need?**
> A: 4-5GB per model. Have at least 10GB free for optimal performance.

---

## Advanced: Custom Models

To use a different model:

1. Pull it: `ollama pull <model-name>`
2. Edit `src/ai_features.py`:
   ```python
   model_options = ["your-model", "llama2", "mistral"]
   ```

---

## Resources

- **Ollama Official**: https://ollama.ai/
- **Available Models**: https://ollama.ai/library
- **Troubleshooting**: https://github.com/ollama/ollama/issues

---

## Support

If you encounter issues:
1. Check Ollama is running: `curl http://localhost:11434`
2. Verify model installed: `ollama list`
3. Check logs in Dashboard (rule-based mode will activate)
4. Restart Ollama service
