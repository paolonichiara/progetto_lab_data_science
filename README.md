### EDA Assistant

# Project Overview
Questo progetto implementa un assistente per Analisi Esplorativa dei Dati (EDA) che combina:
- elaborazione e analisi statistica deterministica in Python
- generazione di insight e interpretazionetramite LLM locale
- validazione tramite multi-agent architecture (Critic / Reflection pattern)
L’obiettivo è produrre insight automatici controllati e verificabili, riducendo errori tipici degli LLM (hallucinations).

# Project Requirements
Il progetto combina:
Capitolo 2 – LLMs
- utilizzo di modello open-source locale (Ollama - `llama3.2:latest`) con modelli quantizzati ottimizzati per l'hardware
- prompt engineering per task analitici
Capitolo 4 – Agents
- implementazione di workflow multi-agente
- pattern: Explainer → Critic (reflection)
- separazione tra generazione e validazione 

## Struttura
EDA_assistant/
├── data/                  # Dataset di input (.csv, .xlsx)
├── outputs/               # Report generati e grafici salvati
├── core/
│   ├── io.py              # Lettura file e data cleaning
│   └── eda/
│       ├── analyzer.py    # Profilazione dei ruoli delle colonne
│       ├── stats.py       # Motore statistico (Pandas, SciPy)
│       └── visualizer.py  # Generazione grafici
├── llm/
│   ├── client.py          # Wrapper per Ollama locale
│   ├── prompts.py         # Prompt engineering (Explainer + Critic)
│   └── explainer.py       # Orchestrazione delle chiamate al LLM
├── agent/
│   └── agent.py           # Classe EDAAgent (Reflection Pipeline)
├── main.py                # Entrypoint principale
└── requirements.txt       # Dipendenze (pandas, scipy, requests...)

## Come eseguire il progetto
1. **Pre-requisiti:** `Ollama` installato e in esecuzione.
2. **Scarica il modello LLM:** ollama run llama3.2:latest
3. **Installa le dipendenze - setup:** pip install -r requirements.txt
4. **Esegui l'analisi - run:** python3 main.py data/sample.csv
   ** Sostituire `data/sample.csv` con qualsiasi file di dataset a tua disposizione

## Esecuzione 
Quando avviato, il sistema compie 5 passi:
1. **Data load:** Carica il dataset gestendo tipi e codifiche.
2. **Column analysis:** Categorizza automaticamente le colonne (numeriche, testo, id, lookup).
3. **Statistical profiling:** Estrae i metadati (valori mancanti, cardinalità, test di normalità).
4. **Visualization:** Salva un set base di grafici esplorativi.
5. **LLM Agents:** 
   - *Explainer*: Genera insight e scrive le conclusioni in base alle statistiche.
   - *Critic*: Verifica la coerenza tra testo generato e dati reali, identificando errori logici o numerici e garantendo che le conclusioni non violino le statistiche calcolate allo step 3.

# Separazione logica: Python vs LLM
- Statistica → Python (deterministico, veloce, affidabile)
- Linguaggio e interpretazione → LLM
* Gli LLM non sono affidabili per calcoli numerici precisi → si evita delega critica al modello
# Critic Agent (Reflection Pattern)
- per ridurre hallucinations e migliorare affidabilità output
- Trade-off: tra maggiore latenza (due chiamate LLM) e migliore qualità finale
# Modello utilizzato
**llama3.2:latest** (≈3B parametri, quantizzato - Llama 3.2 da 3 miliardi di parametri)
- perchè compatibile con CPU (Mac Intel) e migliore qualità rispetto a modelli <1B
- modello abbastanza leggero (circa 2.0 GB di RAM) da poter girare in modo fluido sulla CPU Intel senza causare colli di bottiglia o blocchi di sistema
# Trade-off hardware vs performance
Vincoli: esecuzione locale su CPU
- modello piccolo (3B)
- delega calcolo a librerie ottimizzate (carico computazionale logico-matematico su algoritmi deterministici Python)
- uso LLM solo per reasoning testuale (sintesi testuale e validazione finale)
