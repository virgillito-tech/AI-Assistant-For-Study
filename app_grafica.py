# ==========================================
# 1. IMPORTAZIONE DELLE LIBRERIE 
# ==========================================
import streamlit as st 
# Importa Streamlit e gli assegna il diminutivo "st". Serve per creare tutta l'interfaccia grafica (pulsanti, testi, menu)

import os 
# Importa una libreria base di Python per interagire con il sistema operativo (es. per controllare se una cartella o un file esiste sul PC)

from fpdf import FPDF 
# Importa lo strumento principale per creare e formattare file PDF da zero.

from langchain_community.document_loaders import PyMuPDFLoader 
# Importa il lettore PDF (PyMuPDF). Prende un PDF fisico e ne estrae il testo ignorando immagini rotte o formattazioni strane.

from langchain_ollama import OllamaLLM 
# Importa il "ponte" di comunicazione tra il nostro script Python e il motore Ollama che fa girare Gemma 3 sul PC

from langchain_text_splitters import RecursiveCharacterTextSplitter 
# Importa lo strumento intelligente che taglia i testi lunghi in blocchi più piccoli cercando i punti e le virgole per non spezzare le frasi

from langchain_community.embeddings import OllamaEmbeddings 
# Importa il traduttore matematico. Prende le parole e le trasforma in coordinate numeriche (vettori) per permettere all'IA di calcolare le somiglianze tra i concetti (usato nel RAG)

from langchain_chroma import Chroma 
# Importa ChromaDB, il database locale dove salveremo i testi tradotti in numeri (vettori) dal comando precedente.

from langchain_classic.chains import create_retrieval_chain 
# Importa la "catena" che unisce il database vettoriale all'intelligenza artificiale per cercare le risposte (fase di recupero o "retrieval").

from langchain_classic.chains.combine_documents import create_stuff_documents_chain 
# Importa la funzione che invia i documenti recuperati dal database dentro il prompt da mandare all'IA.

from langchain_core.prompts import ChatPromptTemplate 
# Importa lo strumento per creare i template delle istruzioni (i prompt) inserendo delle variabili dinamiche (come {testo} o {input}).


# ==========================================
# 2. FUNZIONE PER CREARE IL PDF FINALE
# ==========================================

def crea_pdf_risposta(domanda, risposta, nome_file="Sintesi_IA.pdf"):
    # Definisce una funzione personalizzata che accetta 3 parametri: il titolo, il testo generato dall'IA e il nome del file da salvare.
    
    pdf = FPDF() 
    # Crea un oggetto PDF vuoto.
    
    pdf.add_page() 
    # Aggiunge fisicamente la prima pagina al documento.
    
    pdf.set_font("helvetica", size=12) 
    # Imposta il font base del documento 
    
    # --- Intestazione ---
    pdf.set_font("helvetica", style="B", size=14) 
    # Cambia temporaneamente il font mettendolo in Grassetto (B=Bold) e grandezza 14 per il titolo.
    
    pdf.cell(200, 10, txt="Sintesi Generata dall'IA", ln=True, align='C') 
    # Crea una cella invisibile larga 200 e alta 10, ci scrive dentro, va a capo (ln=True) e centra il testo (align='C').
    
    pdf.ln(10) 
    # Aggiunge uno spazio vuoto (un "A capo" di grandezza 10) per distanziare il titolo dal resto.
    
    # --- Titolo/Domanda ---
    pdf.set_font("helvetica", style="B", size=12) 
    # Rimette la grandezza a 12 ma mantiene il grassetto.
    
    pdf.multi_cell(0, 10, txt=f"Riferimento: {domanda}") 
    # Usa multi_cell (che va a capo da solo se il testo è troppo lungo). Lo zero indica "usa tutta la larghezza della pagina".
    
    pdf.ln(5) 
    # Altro spazio vuoto verso il basso.
    
    # --- Risposta dell'IA ---
    pdf.set_font("helvetica", size=12) 
    # Toglie il grassetto per il testo normale del riassunto.
    
    risposta_pulita = str(risposta).encode('latin-1', 'replace').decode('latin-1') 
    # La libreria FPDF base odia le emoji. Questa riga prende la risposta dell'IA, la converte nel set di caratteri base occidentale (latin-1) e sostituisce i caratteri illeggibili con dei punti interrogativi per non far crashare la creazione del PDF.
    
    pdf.multi_cell(0, 8, txt=risposta_pulita) 
    # Stampa tutto il testo del riassunto, andando a capo automaticamente alla fine della riga.
    
    pdf.output(nome_file) 
    # Salva fisicamente il file sul tuo computer con il nome indicato.
    
    return nome_file 
    # Restituisce al resto del programma il nome del file appena salvato per farglielo trovare.


# ==========================================
# 3. FUNZIONE RAG (Indicizzazione di tutto il libro)
# ==========================================

def elabora_documento(percorso_file, cartella_db):
    # Funzione che prende in input il percorso del manuale e il nome della cartella dove salvare il database vettoriale.
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    # Prepara il modello "nomic-embed-text" per trasformare le parole in coordinate matematiche.
    
    loader = PyMuPDFLoader(percorso_file) 
    # Prepara lo strumento di lettura puntandolo al file PDF.
    
    documenti = loader.load() 
    # Esegue la lettura: estrae tutte le pagine e le salva nella variabile 'documenti'.
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
    # Prepara la "ghigliottina": taglierà il libro in blocchi da 1000 caratteri, sovrapponendoli di 200 caratteri per non perdere il filo.
    
    chunk_testo = text_splitter.split_documents(documenti) 
    # Esegue fisicamente il taglio delle pagine creando migliaia di piccoli pezzetti di testo.
    
    vectordb = Chroma.from_documents(documents=chunk_testo, embedding=embeddings, persist_directory=cartella_db) 
    # Crea il database Chroma: prende i pezzetti, li trasforma in numeri usando gli 'embeddings' e li salva fisicamente sul disco nella 'persist_directory'.
    
    return vectordb 
    # Restituisce il database pronto all'uso.


# ==========================================
# 4. INTERFACCIA GRAFICA STREAMLIT
# ==========================================

st.set_page_config(page_title="Analizzatore PDF Universitario", layout="wide") 
# Configura la pagina del browser: le dà un titolo nella scheda e le dice di usare tutto lo spazio orizzontale (wide).

st.title("📄 Assistente allo Studio PDF") 
# Stampa il titolo grande e in grassetto all'inizio della pagina.

uploaded_file = st.file_uploader("Carica il manuale in PDF", type="pdf") 
# Crea il riquadro "Trascina qui il file". Accetta solo file .pdf. Salva il file caricato nella variabile 'uploaded_file'.

if uploaded_file is not None: 
    # IF = SE. Se l'utente ha caricato un file (cioè 'uploaded_file' non è vuoto), allora esegui tutto il codice qui sotto.
    
    percorso_pdf = f"./{uploaded_file.name}" 
    # Crea un nome di percorso fittizio usando il nome originale del file caricato.
    
    with open(percorso_pdf, "wb") as f: 
        # WITH Apre un file fittizio sul tuo PC in modalità Scrittura Binaria e lo chiama temporaneamente 'f'.
        
        f.write(uploaded_file.getbuffer()) 
        # Prende i byte del file che hai caricato nel browser e li scrive fisicamente sull'hard disk del Pc per farli leggere a Python.
    
    st.divider() 
    # Disegna una linea grigia orizzontale nell'interfaccia per separare le sezioni.
    
    st.subheader("Scegli la modalità di studio") 
    # Sottotitolo per il menu.
    
    modalita = st.radio(
        "Come vuoi analizzare il documento?",
        ("Sintesi di un Capitolo (Seleziona le pagine)", "Ricerca Rapida RAG (Fai una domanda su tutto il libro)")
    ) 
    # Crea i due bottoni per farti scegliere cosa fare. Salva la tua scelta nella variabile 'modalita'.
    
    llm = OllamaLLM(model="gemma3:12b", num_ctx=16384, temperature=0.1) 
    # Inizializza l'intelligenza artificiale per il resto dello script. Le dà 16.384 token di memoria massima e una temperatura (0.1) per farla essere analitica e poco creativa.


    # ==========================================
    # MODALITÀ 1: SINTESI A BLOCCHI
    # ==========================================
    if modalita == "Sintesi di un Capitolo (Seleziona le pagine)": 
        # Se hai cliccato sul primo pallino del menu, entra in questo blocco di codice.
        
        st.info("Usa questa modalità per riassumere blocchi di testo continui.") 
        # Stampa un riquadro azzurro informativo.
        
        st.markdown("### Istruzioni per l'IA (Prompt)") 
        # Usa il linguaggio markdown per creare un piccolo titolo testuale.
        
        prompt_utente = st.text_area(
            "Scrivi qui cosa deve fare esattamente l'IA:",
            value="Sei un ...", # Testo di default
            height=150
        ) 
        # Crea il grande riquadro di testo editabile per il prompt. Alto 150 pixel.
        
        col1, col2 = st.columns(2) 
        # Divide lo schermo orizzontalmente in due colonne uguali.
        
        start_page = col1.number_input("Pagina di inizio:", min_value=1, value=1) 
        # Nella prima colonna, crea un selettore numerico per la pagina di partenza. Minimo 1.
        
        end_page = col2.number_input("Pagina di fine:", min_value=1, value=5) 
        # Nella seconda colonna, il selettore per la pagina finale.
        
        if st.button("Genera Sintesi del Capitolo"): 
            # Se clicchi il pulsante "Genera Sintesi...", esegui il codice sotto.
            
            loader = PyMuPDFLoader(percorso_pdf) 
            documenti = loader.load() 
            # Legge il PDF appena salvato sul disco.
            
            if end_page > len(documenti): 
                # Controlla se hai messo un numero di pagina finale superiore alle pagine reali del libro.
                st.error(f"Il documento ha solo {len(documenti)} pagine!") 
                # Se sì, stampa un errore rosso a schermo e si ferma.
                
            else: 
                # ALTRIMENTI (se i numeri sono giusti), procedi.
                st.write("### ⏳ Preparazione del testo in corso...") 
                
                testo_completo = "\n\n".join([doc.page_content for doc in documenti[start_page-1 : end_page]]) 
                # prende le pagine dal tuo start_page all'end_page, ne estrae solo il testo (page_content) e unisce tutto in un'unica gigantesca frase separando le pagine con due "a capo" (\n\n). Python conta da 0, per questo si fa start_page-1.
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000, separators=["\n\n", "\n", ".", " ", ""]) 
                # Configura il tagliatore intelligente per creare maxi-blocchi da 10.000 caratteri.
                
                blocchi_logici = text_splitter.split_text(testo_completo) 
                # Esegue il taglio del testo unito. Ora 'blocchi_logici' è una lista che contiene i vari pezzi.
                
                sintesi_completa = "" 
                # Crea una scatola vuota dove aggiungeremo di volta in volta i riassunti dei vari blocchi.
                
                progress_bar = st.progress(0) 
                # Crea la barra di caricamento visiva impostandola a zero.
                
                status_text = st.empty() 
                # Crea un piccolo spazio di testo "fantasma" che potremo aggiornare dinamicamente senza creare nuove righe a schermo.
                
                totale_step = len(blocchi_logici) 
                # Conta quanti blocchi totali dobbiamo far leggere all'IA.
                
                # --- INIZIO DEL CICLO (LOOP) ---
                for indice, chunk in enumerate(blocchi_logici): 
                    # FOR  Per ogni blocco di testo ('chunk') presente nella lista, ripeti le seguenti istruzioni. 'indice' serve per contare a che giro siamo.
                    
                    status_text.text(f"Analisi e sintesi del blocco logico {indice + 1} di {totale_step}...") 
                    # Aggiorna il testo fantasma dicendo all'utente a che punto siamo.
                    
                    prompt_completo = prompt_utente + "\n\n--- INIZIO TESTO DA SINTETIZZARE ---\n{testo}\n--- FINE TESTO ---" 
                    # Unisce il prompt preso dall'interfaccia grafica con i paletti di sicurezza e il segnaposto {testo}.
                    
                    prompt = ChatPromptTemplate.from_template(prompt_completo) 
                    # Converte la stringa di testo in un formato comprensibile da LangChain.
                    
                    llm_avanzato = OllamaLLM(model="gemma3:12b", num_ctx=16384, temperature=0.1, num_predict=1024) 
                    # Ricarica il modello stringendo il limite di output (num_predict=1024) per obbligarlo a riassumere massicciamente ogni blocco.
                    
                    chain = prompt | llm_avanzato 
                    # Crea la pipeline: le istruzioni del prompt fluiscono (|) dentro il modello IA.
                    
                    risposta_parziale = chain.invoke({"testo": chunk}) 
                    # Chiede all'IA di lavorare. Sostituisce la variabile 'testo' con il 'chunk' attuale e salva il risultato. Questa operazione richiede tempo.
                    
                    sintesi_completa += f"\n\n### Sezione {indice + 1}\n\n" 
                    # Aggiunge alla nostra "scatola" il titolo della sezione. Il simbolo += significa "aggiungi a quello che c'è già".
                    
                    sintesi_completa += risposta_parziale + "\n\n---\n" 
                    # Aggiunge il testo appena generato dall'IA e una linea di separazione.
                    
                    progress_bar.progress((indice + 1) / totale_step) 
                    # Fa avanzare la barra blu di caricamento (calcolando la percentuale matematica).
                # --- FINE DEL CICLO ---
                
                status_text.text("✅ Analisi completata con successo!") 
                # Finiti i blocchi, aggiorna il testo fantasma.
                
                st.write("### 📝 Sintesi Completa Generata:") 
                st.write(sintesi_completa) 
                # Stampa a schermo tutto l'enorme riassunto completo.
                
                titolo_pdf = f"Sintesi Approfondita da pag {start_page} a {end_page}" 
                
                file_pdf_generato = crea_pdf_risposta(titolo_pdf, sintesi_completa) 
                # Chiama la nostra funzione creata all'inizio passandogli i dati per creare il file fisico PDF.
                
                with open(file_pdf_generato, "rb") as pdf_file: 
                    # Apre il file PDF appena creato in modalità "lettura" per permetterne il download.
                    
                    st.download_button(
                        label="⬇️ Scarica Appunti in PDF", 
                        data=pdf_file, 
                        file_name=f"Appunti_Ottimizzati_{start_page}_{end_page}.pdf", 
                        mime="application/pdf"
                    ) 
                    # Crea il pulsante finale di Streamlit. Quando cliccato, scarica i dati ('data=pdf_file') sul PC dell'utente.


    # ==========================================
    # MODALITÀ 2: RICERCA RAG
    # ==========================================
    else: 
        # ALTRIMENTI. Se nel menu a tendina iniziale non era selezionato il primo pulsante, entra qui.
        
        cartella_db = f"./db_{uploaded_file.name.replace('.pdf', '')}" 
        # Genera il nome della cartella dove salvare il database (togliendo l'estensione .pdf dal nome per pulizia).
        
        if not os.path.exists(cartella_db): 
            # Controlla sul tuo hard disk: se questa cartella NON esiste...
            
            with st.spinner("Creazione dell'indice del documento... (Richiesto solo la prima volta)"): 
                # Mostra la rotellina di caricamento di Streamlit all'utente...
                
                vectordb = elabora_documento(percorso_pdf, cartella_db) 
                # ...e chiama la funzione che taglia tutto il libro e crea il database pesante.
                
                st.success("Documento indicizzato!") 
                # Riquadro verde di successo.
        else: 
            # ALTRIMENTI (se la cartella esiste già sul PC)...
            st.success("Database trovato! Caricamento istantaneo.") 
            
            embeddings = OllamaEmbeddings(model="nomic-embed-text") 
            vectordb = Chroma(persist_directory=cartella_db, embedding_function=embeddings) 
            # Semplicemente riapre la cartella e ricarica i vettori in memoria in mezzo secondo.
        
        richiesta = st.text_area("Fai una domanda specifica al libro:") 
        # Riquadro di testo dove inserisci la tua domanda
        
        if st.button("Cerca Risposta"): 
            # Se premi il bottone di ricerca...
            
            if richiesta: 
                # E se il riquadro non è vuoto...
                
                with st.spinner("Ricerca tra le pagine in corso..."): 
                    
                    retriever = vectordb.as_retriever(search_kwargs={"k": 5}) 
                    # Imposta il database come "Cercatore" (retriever). "k=5" gli dice di andare a pescare i 5 pezzi di libro matematicamente più simili alla tua domanda.
                    
                    system_prompt = (
                        "Sei un analista finanziario senior. Rispondi alla domanda dell'utente usando SOLO il contesto fornito.\n"
                        "Sii conciso, chiaro ed estrai le formule se presenti.\n\nContesto:\n{context}"
                    ) 
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}")
                    ]) 
                    # Crea il prompt strutturato in stile Chatbot. Le regole di base (system) contengono il pezzo di libro pescato ({context}), e la voce dell'utente (human) contiene la tua domanda ({input}).
                    
                    question_answer_chain = create_stuff_documents_chain(llm, prompt) 
                    # Spiega a LangChain come infilare i 5 pezzi di testo dentro la variabile {context}.
                    
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain) 
                    # Assembla la macchina finale: La richiesta parte -> va al Retriever che pesca i 5 pezzi -> li passa alla question_answer_chain -> va all'LLM.
                    
                    risposta = rag_chain.invoke({"input": richiesta}) 
                    # Schiaccia il pulsante di avvio. Fa partire tutta la catena e aspetta il risultato.
                    
                    testo_risposta = risposta["answer"] 
                    # Dalla valanga di dati estratti, tira fuori solo la risposta testuale vera e propria.
                    
                    st.write("### 💡 Risposta:") 
                    st.write(testo_risposta) 
                    # Mostra a schermo.
                    
                    file_pdf_generato = crea_pdf_risposta(richiesta, testo_risposta) 
                    # Genera il PDF.
                    
                    with open(file_pdf_generato, "rb") as pdf_file: 
                        st.download_button(label="⬇️ Scarica Risposta in PDF", data=pdf_file, file_name="Risposta_RAG.pdf", mime="application/pdf")
                        # Permette il download della risposta singola.