# 🧠 OfflineTextClassifier

**OfflineTextClassifier** is a .NET library for performing **zero-shot text classification** fully **offline** using **ML.NET** and **ONNX**. It enables you to dynamically classify any input text into user-defined labels **without retraining** or needing an internet connection.

Ideal for secure, air-gapped, or enterprise environments where cloud-based NLP is not an option.

---

## ✨ What is Zero-Shot Classification?

Zero-shot classification allows you to assign labels to text that the model has **never explicitly seen during training**. You simply provide a list of possible categories (as natural language labels), and the model will predict the most likely label(s) for a given text input based on **semantic understanding**.

---

## ✅ Pros and ❌ Cons

### ✅ Pros

- **💡 Zero-shot flexibility**  
  Define your own labels at runtime — no fine-tuning or training required.

- **🔒 Offline-capable**  
  All inference happens locally using ONNX and ML.NET. No internet or external API required.

- **🧰 ML.NET Native**  
  Designed for seamless integration in .NET Core / Framework apps, including ASP.NET, console, or background services.

- **⚡ Fast & Lightweight**  
  Uses optimized transformer models (e.g., MiniLM) converted to ONNX for fast inference.

---

### ❌ Cons

- **📐 Accuracy Limitations**  
  May not match the accuracy of large cloud-based models like OpenAI/GPT or Hugging Face APIs.

- **📦 ONNX Model Size**  
  Embedding transformer models increases package or deployment size.

- **🧠 Approximate Reasoning**  
  Zero-shot predictions rely on natural language inference (NLI) or semantic similarity—results can vary depending on label phrasing.

- **🗣️ Limited Multilingual Support**  
  Depends on the base model’s language capabilities (e.g., MiniLM is primarily English).

