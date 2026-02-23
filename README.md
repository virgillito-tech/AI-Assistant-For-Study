# AI Assistant For Study
### (RAG + Ollama)

Un'applicazione web costruita con **Python** e **Streamlit** che permette di analizzare, interrogare e riassumere documenti PDF di grandi dimensioni (come manuali universitari) in modo **completamente locale e gratuito**.

Il sistema sfrutta **Ollama** per far girare modelli LLM avanzati (come Gemma 3) direttamente sul tuo hardware e utilizza la tecnologia **RAG** (Retrieval-Augmented Generation) per navigare agilmente in documenti di centinaia di pagine.

## Funzionalità Principali

L'applicazione offre due modalità di studio:
1. **Sintesi per Capitolo:** Seleziona il range di pagine di un capitolo. Il sistema dividerà il testo in blocchi logici e genererà un rissunto ben strutturato, mantenendo intatte formule e definizioni matematiche.
2. **Ricerca Rapida RAG:** Poni una domanda specifica. L'IA cercherà la risposta tra le centinaia di pagine del manuale, indicando esattamente i concetti rilevanti.
3. **Esportazione in PDF:** Ogni sintesi o risposta generata può essere scaricata istantaneamente in un file PDF pulito e formattato, pronto per la stampa.

## Requisiti di Sistema

Per far girare questo progetto sul tuo computer, devi avere installato:
* [Python 3.8+](https://www.python.org/downloads/)
* [Ollama](https://ollama.com/)

Prima di avviare lo script, assicurati di aver scaricato i modelli necessari tramite terminale:
```bash
ollama pull nomic-embed-text
ollama pull gemma3:12b
```
