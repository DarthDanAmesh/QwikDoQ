pip install nltk llama-index-llms-ollama llama-index-embeddings-huggingface "fastapi[standard]" PyMuPDF llama-index llama-index-core llama-index-vector-stores-chroma llama-index-retrievers-bm25 llama-index-embeddings-langchain llama-index-llms-openai

been running it with python lama.py

exlpore https://leapcell.io/  for hosting
$uvicorn lama:app --reload --port 8000


task:Find the GAZETTE NOTICE NO. 16787 and provide its contents and Summary

response by lamawithBM25.py


The document Notice 16787 is about the Judicial Service's Steering Committee's decision to develop and implement Court-Annexed Mediation for dispute resolution in Kenya. It further outlines the appointment of judges as part of this committee, based on indexed documents. The second document (page_29764) relevance is 3.58954 GAZETTE NOTICE 16786, which discusses the DPP's recommendation to close an inquiry into allegations against a Former Employee of Nairobi City Water and Sewerage Company (NCWSC). Summary: The Judicial Service Committee approved Court-Annexed Mediation as part of its Steering Committee, appointing judges for the Diplomatic Security Commission. Meanwhile, in Document 16787, it was confirmed that seven employees from Nairobi Water and Sewerage Company (NCWSC) did re-enter Kenya through the Department of Immigration's Namada border point after their official return on the DJ DPP Civil suit No. 354 of 2016. Please note that only the search results' query, indexed documents 81EACC/EL/INQ/7 (I)/2024 and Document_D16787, were used to find DJ inquiry's return date on 29th September 2025. The DPP returned the DJ File Civil Suit No. 354 of 2016 concurring with Commission recommendations to close file and notifies DPP that the DIN filed for closure is correct, confirming no objections were raised by official return dates reflected in documentation.
03:47 PM


fastapi==0.115.0
uvicorn==0.30.6
chromadb==0.5.5
pymupdf==1.24.9
ollama==0.3.3
pandas==2.2.2
openpyxl==3.1.5
pymilvus
pymilvus[model]
python-multipart
milvus-lite
