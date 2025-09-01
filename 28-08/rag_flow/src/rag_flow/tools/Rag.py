from dataclasses import dataclass
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document
from typing import List

load_dotenv()
@dataclass
class Settings:
    # Persistenza FAISS
    persist_dir: str = "faiss_index_example"
    # Text splitting
    chunk_size: int = 700
    chunk_overlap: int = 100
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" o "similarity"
    k: int = 4                      # risultati finali
    fetch_k: int = 20               # candidati iniziali (per MMR)
    mmr_lambda: float = 0.3         # 0 = diversificazione massima, 1 = pertinenza massima
    # Embedding
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    key = os.getenv("API_KEY")
    llm_key = os.getenv("API_LLM_KEY")
    azure_llm_endpoint = os.getenv("AZURE_LLM_ENDPOINT")
    #hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # LM Studio (OpenAI-compatible)
    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    model_name: str = "gpt-4o"
    lmstudio_model_env: str = "LMSTUDIO_MODEL"  # nome del modello in LM Studio, via env var

class Rag:

    def __init__(self,settings: Settings):
        # Chunk -> Embedding -> store  
        self.embedder = self.define_embedder(settings)
        self.llm = self.define_llm(settings)
        #splittare i docs
        chunks = self.split_docs()
        #converte i chunks in una loro rappresentazione vettoriale 
        self.vec_db = self.define_vector_db(settings,chunks)
        print("Vec init")
        self.retriver = self.make_retriever(settings)

        self.chain = self.build_rag_chain()


    def define_llm(self,settings: Settings):
        llm = AzureChatOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=settings.azure_llm_endpoint,
            api_key=settings.llm_key,
            azure_deployment="gpt-4o",  # Replace with actual deployment
            temperature=0.1,
            max_tokens=1000,  # Optional: control response length
        )
        return llm
    
    def define_embedder(self, settings: Settings):
        embedding = AzureOpenAIEmbeddings(
            api_version="2024-12-01-preview",
            azure_endpoint=settings.azure_endpoint,
            api_key=settings.key,
        )
        return embedding
    
    def define_vector_db(self, settings: Settings, chunks):
        index = FAISS.from_documents(
            documents=chunks,
            #passare la funzione di embedding 
            embedding=self.embedder
        )
        return index
    
    def _create_documents(self) -> List[Document]:
        """Crea documenti medici di esempio hardcodati"""
        medical_documents = [
            # Malattie respiratorie
            Document(
                page_content="""L'asma è una malattia respiratoria cronica caratterizzata da infiammazione delle vie aeree.
                Sintomi principali:
                - Dispnea (difficoltà respiratoria)
                - Tosse persistente, specialmente notturna
                - Respiro sibilante
                - Senso di oppressione toracica
                
                Cause:
                - Allergeni (acari, pollini, pelo di animali)
                - Irritanti (fumo, inquinamento)
                - Infezioni respiratorie
                - Esercizio fisico intenso
                - Stress emotivo
                
                Trattamento:
                - Broncodilatatori a breve durata (salbutamolo)
                - Corticosteroidi inalatori per controllo a lungo termine
                - Antileucotrieni
                - Evitare i trigger
                - Piano d'azione personalizzato per gestire le crisi""",
                metadata={"categoria": "respiratorio", "malattia": "asma"}
            ),
            
            Document(
                page_content="""L'influenza è una malattia respiratoria contagiosa causata dai virus influenzali.
                
                Sintomi tipici:
                - Febbre alta improvvisa (38-40°C)
                - Dolori muscolari e articolari
                - Mal di testa intenso
                - Tosse secca
                - Mal di gola
                - Stanchezza estrema
                - Naso che cola o congestionato
                
                Prevenzione:
                - Vaccinazione annuale
                - Lavaggio frequente delle mani
                - Evitare contatti con persone malate
                - Coprire bocca e naso quando si starnutisce
                
                Trattamento:
                - Riposo a letto
                - Idratazione abbondante
                - Antipiretici (paracetamolo, ibuprofene)
                - Antivirali se iniziati entro 48 ore (oseltamivir)""",
                metadata={"categoria": "infettivo", "malattia": "influenza"}
            ),
            
            # Malattie metaboliche
            Document(
                page_content="""Il diabete mellito è una malattia cronica caratterizzata da alti livelli di glucosio nel sangue.
                
                Diabete di Tipo 1:
                - Esordio in età giovane
                - Distruzione autoimmune delle cellule beta pancreatiche
                - Richiede insulina esogena
                
                Diabete di Tipo 2:
                - Più comune negli adulti
                - Resistenza all'insulina
                - Spesso associato a obesità
                
                Sintomi comuni:
                - Poliuria (minzione frequente)
                - Polidipsia (sete eccessiva)
                - Perdita di peso inspiegabile
                - Visione offuscata
                - Stanchezza cronica
                - Guarigione lenta delle ferite
                
                Gestione:
                - Monitoraggio glicemico regolare
                - Dieta equilibrata e controllo carboidrati
                - Esercizio fisico regolare
                - Farmaci orali (metformina) o insulina
                - Controllo del peso
                - Gestione dello stress""",
                metadata={"categoria": "metabolico", "malattia": "diabete"}
            ),
            
            # Malattie cardiovascolari
            Document(
                page_content="""L'ipertensione arteriosa è una condizione caratterizzata da pressione sanguigna costantemente elevata.
                
                Classificazione:
                - Normale: <120/80 mmHg
                - Elevata: 120-129/<80 mmHg
                - Ipertensione stadio 1: 130-139/80-89 mmHg
                - Ipertensione stadio 2: ≥140/90 mmHg
                
                Fattori di rischio:
                - Età avanzata
                - Storia familiare
                - Obesità
                - Sedentarietà
                - Dieta ricca di sodio
                - Stress cronico
                - Fumo e alcol
                
                Complicanze:
                - Infarto miocardico
                - Ictus
                - Insufficienza renale
                - Retinopatia
                
                Trattamento:
                - Modifiche dello stile di vita (dieta DASH)
                - ACE-inibitori
                - Beta-bloccanti
                - Diuretici
                - Calcio-antagonisti
                - Monitoraggio regolare""",
                metadata={"categoria": "cardiovascolare", "malattia": "ipertensione"}
            ),
            
            # Malattie gastrointestinali
            Document(
                page_content="""La gastrite è l'infiammazione della mucosa gastrica che può essere acuta o cronica.
                
                Cause principali:
                - Infezione da Helicobacter pylori
                - Uso prolungato di FANS
                - Consumo eccessivo di alcol
                - Stress grave
                - Reflusso biliare
                
                Sintomi:
                - Dolore epigastrico
                - Nausea e vomito
                - Sensazione di pienezza dopo i pasti
                - Perdita di appetito
                - Bruciore di stomaco
                - Eruttazione frequente
                
                Diagnosi:
                - Endoscopia con biopsia
                - Test per H. pylori
                - Esami del sangue
                
                Trattamento:
                - Inibitori di pompa protonica
                - Antibiotici per H. pylori
                - Antiacidi
                - Modifiche dietetiche
                - Evitare irritanti gastrici""",
                metadata={"categoria": "gastrointestinale", "malattia": "gastrite"}
            ),
            
            # Malattie neurologiche
            Document(
                page_content="""L'emicrania è un disturbo neurologico caratterizzato da mal di testa ricorrenti e intensi.
                
                Caratteristiche:
                - Dolore pulsante unilaterale
                - Durata 4-72 ore
                - Intensità moderata-severa
                
                Sintomi associati:
                - Nausea e vomito
                - Fotofobia (sensibilità alla luce)
                - Fonofobia (sensibilità ai suoni)
                - Aura visiva nel 25% dei casi
                
                Trigger comuni:
                - Stress
                - Cambiamenti ormonali
                - Alcuni alimenti (cioccolato, formaggi stagionati)
                - Alterazioni del sonno
                - Cambiamenti meteorologici
                
                Trattamento:
                - Triptani per attacchi acuti
                - FANS
                - Beta-bloccanti per profilassi
                - Antiepilettici preventivi
                - Tecniche di rilassamento
                - Diario delle emicranie""",
                metadata={"categoria": "neurologico", "malattia": "emicrania"}
            ),
            
            # Allergie
            Document(
                page_content="""La rinite allergica è un'infiammazione della mucosa nasale causata da reazione allergica.
                
                Tipi:
                - Stagionale (febbre da fieno)
                - Perenne (tutto l'anno)
                
                Sintomi tipici:
                - Starnuti ripetuti
                - Rinorrea acquosa
                - Congestione nasale
                - Prurito nasale, oculare e palatale
                - Lacrimazione
                - Occhiaie allergiche
                
                Allergeni comuni:
                - Pollini (graminacee, alberi)
                - Acari della polvere
                - Pelo di animali domestici
                - Muffe
                
                Trattamento:
                - Antistaminici orali
                - Corticosteroidi nasali
                - Decongestionanti
                - Immunoterapia specifica
                - Evitare allergeni
                - Lavaggi nasali con soluzione salina""",
                metadata={"categoria": "allergico", "malattia": "rinite_allergica"}
            ),
            
            # Malattie infettive comuni
            Document(
                page_content="""La polmonite è un'infezione che infiamma gli alveoli polmonari.
                
                Agenti causali:
                - Batterici (Streptococcus pneumoniae più comune)
                - Virali
                - Fungini (in immunocompromessi)
                
                Sintomi:
                - Febbre alta con brividi
                - Tosse produttiva con espettorato
                - Dolore toracico pleuritico
                - Dispnea
                - Tachicardia
                - Confusione (negli anziani)
                
                Diagnosi:
                - Radiografia toracica
                - Esami del sangue (PCR, emocromo)
                - Coltura dell'espettorato
                
                Trattamento:
                - Antibiotici (amoxicillina, macrolidi)
                - Supporto respiratorio se necessario
                - Idratazione
                - Antipiretici
                - Fisioterapia respiratoria
                - Vaccinazione preventiva""",
                metadata={"categoria": "infettivo", "malattia": "polmonite"}
            )
        ]
        
        return medical_documents
    
    def split_docs(self):
        splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ":", ";", " "]
        )
        docs = self._create_documents()
        chunks = splitter.split_documents(docs)
        return chunks
    
    def make_retriever(self,settings: Settings):
        """
        Configura il retriever. Con 'mmr' otteniamo risultati meno ridondanti e più coprenti.
        """
        if settings.search_type == "mmr":
            return self.vec_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
            )
        else:
            return self.vec_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.k},
            )
        
    def build_rag_chain(self):
        """
        Costruisce la catena RAG (retrieval -> prompt -> LLM) con citazioni e regole anti-hallucination.
        """
        template = (
            "Sei un medico esperto. Rispondi nella lingua della domanda. "
            "Usa esclusivamente il contenuto fornito all'interno del database   . "
            "Se l'informazione non è presente, dichiara che non è disponibile. "
            "Includi citazioni tra parentesi quadre nel formato [source:...]. "
            "Sii conciso, accurato e tecnicamente corretto       "
            "Contesto dal database: {context}"
            "Domanda: {question}"
            "Fornisci una risposta dettagliata, precisa e in italiano:"
        )

        prompt = ChatPromptTemplate.from_template(template)

        # LCEL: dict -> prompt -> llm -> parser
        chain = (
            {
                "context": self.retriver, 
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain
    
    def rag_answer(self,question: str) -> str:
        """
        Esegue la catena RAG per una singola domanda.
        """
        return self.chain.invoke(question)
    
    
    

    
