How to run:
- place PDFs into PDF/raw/ folder
- run "python pdf_to_text.pdf
- The script produces:
-- "cleaed up" PDF files (images are removed and a small strip of the top/bottom of the document is cut in the hope of cutting off headers/footers)
-- raw extracted text (found in TXT/)
-- the extracted abstracts and body of the texts (everything between the end of the abstract and the start of the references) (found in TXT/clean/)


List of problematic files:

chen2019 (can't identify abstract -> entire text gets read as "abstract")
old_Kniplong2011, old_Lajunen1997, old_Aljaafreh (image-based, can't extract text)
old_Knipling2011_report (trouble identifying the structure of the document)
Rossner2019 (doesn't have an "Introduction" heading -> can't identify abstract)
