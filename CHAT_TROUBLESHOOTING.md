# Quick Troubleshooting: AI Chat Not Working

## ⚡ Quick Fixes (Try These First)

### **1. Restart the Dashboard**
- Close the Streamlit app
- Run: `streamlit run main.py`
- Try asking a question again

### **2. Check Ollama Service**

**Windows/Mac:**
- Look for Ollama icon in system tray
- If not running, click to restart

**Linux:**
- Check if running: `curl http://localhost:11434`
- If not, start: `ollama serve`

### **3. Install a Model**
```bash
ollama pull llama2
```

---

## 🔍 Detailed Troubleshooting

### **Scenario 1: Chat shows error message**

**Error**: "Ollama is not running"
- ✅ **Solution**: Start Ollama service (see above)

**Error**: "Connection refused"
- ✅ **Solution**: Check if `http://localhost:11434` is accessible
- Run: `curl http://localhost:11434`
- Expected: "Ollama is running"

**Error**: "Model not found"
- ✅ **Solution**: Pull a model: `ollama pull llama2`

---

### **Scenario 2: Chat works but gives generic responses**

**This is normal!** The system is using rule-based mode.

**To get LLM responses:**
1. Ensure Ollama is running
2. Verify a model is installed: `ollama list`
3. Restart the dashboard
4. Try again

---

### **Scenario 3: Chat is very slow**

**Possible causes:**
- Model is loading for the first time (takes 30 seconds)
- System is low on RAM
- Large model is running

**Solutions:**
- Wait for first response (it's faster after)
- Close other applications
- Use smaller model: `ollama pull neural-chat`

---

## 📋 Verification Checklist

- [ ] Ollama is installed: `ollama --version`
- [ ] Ollama is running: `curl http://localhost:11434`
- [ ] Model is pulled: `ollama list` (should show at least one)
- [ ] Dashboard is restarted after installing Ollama
- [ ] You're using correct port: `http://localhost:11434`

---

## 🎯 Expected Behavior

### ✅ **Correct Behavior**

**With Ollama:**
```
User: "Who is the best batsman?"
Assistant: [Detailed response about top batsman with stats]
```

**Without Ollama (Fallback):**
```
User: "Who is the best batsman?"
Assistant: "Based on the data, the batsmen with highest runs are shown in context..."
```

### ❌ **Incorrect Behavior**

- Chat button doesn't respond → Restart dashboard
- Always shows "Ollama Error" → Start Ollama service
- Takes >1 minute to respond → Check system resources

---

## 🔧 Testing Commands

### **Test Ollama Connection**
```bash
curl http://localhost:11434
```
Expected: `Ollama is running`

### **List Installed Models**
```bash
ollama list
```
Expected: At least one model listed

### **Test Model Directly**
```bash
ollama run llama2 "What is cricket?"
```
Expected: Model responds

### **Python Import Test**
```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama2", base_url="http://localhost:11434")
print(llm.invoke("ping"))
```
Expected: Response (not error)

---

## 🚨 Still Not Working?

### **Check Dependencies**
```bash
pip install langchain-community ollama
```

### **Check Python Version**
```python
python --version
```
Requires: Python 3.8+

### **Check Port Availability**
```bash
netstat -ano | findstr 11434  # Windows
lsof -i :11434               # Mac/Linux
```

### **Reinstall Ollama**
1. Uninstall current Ollama
2. Download fresh from https://ollama.ai
3. Install and restart
4. Pull a model: `ollama pull llama2`

---

## 📞 Debug Steps

1. **Check Ollama status**
   ```bash
   curl -v http://localhost:11434
   ```

2. **View Ollama logs** (if installed locally)
   - Windows: Check system tray Ollama app
   - Mac: Check Console.app
   - Linux: Check terminal where `ollama serve` is running

3. **Test specific model**
   ```bash
   ollama run mistral "Hello"
   ```

4. **Check firewall**
   - Allow Ollama through Windows Firewall
   - Mac: System Preferences → Security & Privacy

5. **Verify langchain installation**
   ```bash
   pip show langchain-community
   ```

---

## 💡 Tips

- First response is slower (model loading) - subsequent are faster
- Rule-based mode is a great fallback - still gives useful responses
- The dashboard will **never crash** due to LLM issues
- All processing is local/private

---

## ✅ Summary

| Issue | Check |
|-------|-------|
| No response | Is Ollama running? |
| Slow responses | First run? Or low RAM? |
| Generic responses | Is a model pulled? |
| Connection error | Is port 11434 open? |

**Most common fix**: Start Ollama service 🚀
