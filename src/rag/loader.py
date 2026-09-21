from pathlib import Path
from pypdf import PdfReader
from docx import Document


def load_documents(path):

    documents = Path(path).glob('*')
    files = []

    for doc_path in documents:
        content = {}

        if doc_path.suffix == ".txt":
            content["source"] = doc_path.name
            content["text"] = doc_path.read_text(encoding="utf-8")
            files.append(content)

        elif doc_path.suffix == ".pdf":
            with open(doc_path, 'rb') as file:
                reader = PdfReader(file)
                pdf_text = ''
                for page in reader.pages:
                    pdf_text += page.extract_text() + "\n"

                content["source"] = doc_path.name
                content["text"] = pdf_text
                files.append(content)

        elif doc_path.suffix == ".docx":
            with open(doc_path, 'rb') as file:
                document = Document(file)
                doc_text = ''

                for paragraph in document.paragraphs:
                    doc_text += paragraph.text + "\n"

            content["source"] = doc_path.name
            content["text"] = doc_text
            files.append(content)

        else:
            continue

    return files

if __name__ == "__main__":
    folder = "raw_data/"

    files = load_documents(folder)

    print(len(files))

    for f in files:
        print(f"source: {f['source']} , text: {f['text'][:400]}")

