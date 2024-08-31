from docx import Document
import os

def get_docs(path):
    docs = []
    for root, dirs, files in os.walk(path):
        for f in files:
            docs.append(root + f)
            print(docs)
    return docs

def split_document_by_header(doc_path, output_dir, header):
    # Apri il documento Word
    doc = Document(doc_path)
    
    # Variabili per tracciare i file e i contenuti dei paragrafi
    file_index = 0
    current_paragraphs = []

    def write_paragraphs_to_file(paragraphs, index):
        # Salva i paragrafi accumulati in un file di testo
        if paragraphs:
            name = doc_path.split("/")[2]
            name = name.split('.')[0]
            file_name = f"{name}_{index}.txt"
            file_path = f"{output_dir}/{file_name}"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(paragraphs))
            print(f"Sezione {index} salvata in {file_path}")

    # Itera su tutti i paragrafi nel documento
    for para in doc.paragraphs[1:]:
        if para.style.name == header:
            # Se incontriamo un header, salviamo il contenuto accumulato finora in un nuovo file
            write_paragraphs_to_file(current_paragraphs, file_index)
            # Resetta i paragrafi accumulati e incrementa l'indice del file
            current_paragraphs = [para.text]
            file_index += 1
        else:
            # Aggiungi il paragrafo corrente alla lista dei paragrafi accumulati
            current_paragraphs.append(para.text)
    
    # Scrivi l'ultima sezione accumulata
    write_paragraphs_to_file(current_paragraphs, file_index)

# Percorso al documento Word
doc_path = './docs/'
# Directory di output per i file di testo
output_dir = './out/'
# Stile del paragrafo che identifica un nuovo header
header_style = 'Heading 2'

docs = get_docs(doc_path)
# Esegui la funzione
for d in docs:
    split_document_by_header(d, output_dir, header_style)
