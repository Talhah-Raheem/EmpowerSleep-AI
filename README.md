# EmpowerSleep AI  
### Personalized sleep guidance powered by AI and evidence-based sleep science

EmpowerSleep AI is an intelligent sleep assistant designed to help users understand their sleep patterns, improve sleep quality, and build healthier nighttime habits.

The system uses **Retrieval-Augmented Generation (RAG)** to provide responses grounded in trusted sleep resources and EmpowerSleep educational content.

🌐 **Visit:** www.empowersleep.ai

---

# Why EmpowerSleep AI?

Many people struggle with sleep but don’t know where to start. Generic advice often ignores individual patterns, habits, and underlying factors.

EmpowerSleep AI provides contextual, evidence-based guidance so users can better understand their sleep challenges and take actionable steps toward improvement.

Built for people who want to:

- Understand why they feel tired even after sleeping  
- Improve **sleep quality**, not just sleep duration  
- Build healthier nighttime routines  
- Learn evidence-based sleep science in simple language  
- Get quick, personalized guidance about sleep habits  

---

# Core Features

## AI Sleep Assistant
Ask questions about sleep habits, sleep quality, circadian rhythms, or fatigue. The AI provides grounded responses based on trusted sleep resources.

## Evidence-Based Responses
Responses are generated using a **Retrieval-Augmented Generation (RAG)** system that references curated sleep research and EmpowerSleep educational content.

## Smart Follow-Up Suggestions
The system suggests helpful follow-up questions so users can explore related sleep topics more easily.

## Feedback System
Users can rate responses with **thumbs-up or thumbs-down feedback** to continuously improve AI answer quality.

## Conversation History
Sessions can be stored so users can revisit previous sleep questions and responses.

---

# Built With

EmpowerSleep AI is built using modern AI and cloud infrastructure.

### Frontend
- Next.js  
- React  
- TypeScript  
- Tailwind CSS  

### Backend
- FastAPI  
- Python  

### AI Infrastructure
- OpenAI API  
- Retrieval-Augmented Generation (RAG)  
- FAISS Vector Search  

### Database & Storage
- Supabase (PostgreSQL)

### Deployment
- Railway (backend hosting)  
- Vercel / Railway (frontend hosting)

---

# Architecture Overview

The system uses a **Retrieval-Augmented Generation pipeline**:

1. A user asks a sleep-related question  
2. The system searches a vector database of trusted sleep resources  
3. Relevant context is retrieved from the knowledge base  
4. The AI generates a grounded response using the retrieved context  
5. The response is returned to the user

This architecture helps ensure responses remain **accurate, safe, and evidence-based**.

---

# Safety and Responsible AI

EmpowerSleep AI is designed to provide **educational guidance**, not medical diagnosis.

Safety mechanisms include:

- Guardrails to avoid medical diagnosis or treatment claims  
- Evidence-based context retrieval  
- Carefully designed prompts to reduce hallucinations  
- Continuous monitoring through user feedback  

Users experiencing severe sleep issues are encouraged to consult healthcare professionals.

---

# Getting Started

Visit the live application:

👉 **https://empowersleep.ai**

Ask a sleep question and explore how AI can help you better understand your sleep.

---

# Feedback

User feedback helps improve the system.

Responses can be rated directly in the interface to help refine the model and improve future answers.

---

*Built with FastAPI · Next.js · FAISS · OpenAI · Supabase*
