# Preprocesses PDF files

import os

directory = os.fsencode("PDF/raw")
pdftotext = "\"C:/Program Files/XPDF/spdf-tools-win-4.02/bin64/pdftotext\""

 #clean up PDFs and convert into txt
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     print("looking at ", filename)
     if filename.endswith(".pdf"):
        os.system("start cpdf.exe PDF/raw/" + filename + " -draft AND -crop \"0mm 15mm 220mm 240mm\" -o PDF/cleaned/" + filename)#
        txt_filename = " TXT/" + filename[:-4]+".txt"
        print("writing txt: ", txt_filename)
        os.system("start pdftotext.exe -raw -clip -nodiag -nopgbrk PDF/cleaned/" + filename + txt_filename)
     else:
        continue

# Clean up txt (remove anything before the abstract and from the references on)
# Everything between the headlines "Abstract" and "Introduction" (in various spellings)
# is assumed to be the abstract
for file in os.listdir("TXT"):
    filename = "TXT/" + os.fsdecode(file)
    print("working on ", filename)
    if not filename.endswith(".txt"):
        continue
    txt_only_filename = "TXT/clean/txt_body_only/" + os.fsdecode(file)
    txt_with_abstr_filename = "TXT/clean/txt_with_abstracts/" + os.fsdecode(file)
    abstr_filename = "TXT/clean/abstracts/" + os.fsdecode(file)

     # Read in the file
    file = open(filename, 'r')
    text = file.read()
    file.close()

    # Remove everything before Abstract
    if ("a b s t r a c t" in text):
        split_txt = text.split("a b s t r a c t")
        i = 1
        text = ""
        while i < len(split_txt): # the word "abstract might appear in the text as well -> want to preserve those parts of the text
            text += split_txt[i]
            i += 1
    elif ("ABSTRACT" in text):
        split_txt = text.split("ABSTRACT")
        i = 1
        text = ""
        while i < len(split_txt): # the word "abstract might appear in the text as well -> want to preserve those parts of the text
            text += split_txt[i]
            i += 1
    elif ("A B S T R A C T" in text):
        split_txt = text.split("A B S T R A C T")
        i = 1
        text = ""
        while i < len(split_txt): # the word "abstract might appear in the text as well -> want to preserve those parts of the text
            text += split_txt[i]
            i += 1
    elif ("Abstract" in text):
        split_txt = text.split("Abstract")
        # If the "abstract" is too long, it's probably not an actual abstract
        # but instead just a random occurence of the word -> don't cut
        if not (len(split_txt[0]) > (0.1 * len(text))):
            i = 1
            text = ""
            while i < len(split_txt): # the word "abstract might appear in the text as well -> want to preserve those parts of the text
                text += split_txt[i]
                i += 1
    elif ("S U M M A R Y" in text):
        split_txt = text.split("S U M M A R Y")
        i = 1
        text = ""
        while i < len(split_txt): # the word "abstract might appear in the text as well -> want to preserve those parts of the text
            text += split_txt[i]
            i += 1
    elif ("SUMMARY" in text):
        split_txt = text.split("SUMMARY")
        i = 1
        text = ""
        while i < len(split_txt): # the word "abstract might appear in the text as well -> want to preserve those parts of the text
            text += split_txt[i]
            i += 1
    elif ("Summary" in text):
        split_txt = text.split("Summary")
        if not (len(split_txt[0]) > (0.1 * len(text))):
            i = 1
            text = ""
            while i < len(split_txt): # the word "abstract might appear in the text as well -> want to preserve those parts of the text
                text += split_txt[i]
                i += 1
    elif ("s u m m a r y" in text):
        split_txt = text.split("s u m m a r y")
        i = 1
        text = ""
        while i < len(split_txt): # the word "abstract might appear in the text as well -> want to preserve those parts of the text
            text += split_txt[i]
            i += 1

    # Isolate abstract from the rest of the text
    if ("INTRODUCTION" in text):
        split_txt = text.split("INTRODUCTION")
        abstract = ""
        text = ""
        abstract = split_txt[0]
        i = 1
        while i < len(split_txt):
            text += split_txt[i]
            i += 1
    elif ("I N T R O D U C T I O N" in text):
        split_txt = text.split("I N T R O D U C T I O N")
        abstract = ""
        text = ""
        abstract = split_txt[0]
        i = 1
        while i < len(split_txt):
            text += split_txt[i]
            i += 1
    elif ("i n t r o d u c t i o n" in text):
        split_txt = text.split("i n t r o d u c t i o n")
        abstract = ""
        text = ""
        abstract = split_txt[0]
        i = 1
        while i < len(split_txt):
            text += split_txt[i]
            i += 1
    elif ("Introduction" in text):
        split_txt = text.split("Introduction")
        abstract = ""
        text = ""
        abstract = split_txt[0]
        i = 1
        while i < len(split_txt):
            text += split_txt[i]
            i += 1

    # Remove everything after (the last occurence of) References
    if "References" in text:
        print("removing references")
        split_txt = text.split("References")
        i = 0
        text = ""
        while i < (len(split_txt) - 1):
            text += split_txt[i]
            i += 1
    elif "REFERENCES" in text:
        print("removing references")
        split_txt = text.split("REFERENCES")
        i = 0
        text = ""
        while i < (len(split_txt) - 1):
            text += split_txt[i]
            i += 1
    elif "R E F E R E N C E S" in text:
        print("removing references")
        split_txt = text.split("R E F E R E N C E S")
        i = 0
        text = ""
        while i < (len(split_txt) - 1):
            text += split_txt[i]
            i += 1
    elif "r e f e r e n c e s" in text:
        print("removing references")
        split_txt = text.split("r e f e r e n c e s")
        i = 0
        text = ""
        while i < (len(split_txt) - 1):
            text += split_txt[i]
            i += 1
    elif "Bibliography" in text:
        print("removing references")
        split_txt = text.split("Bibliography")
        i = 0
        text = ""
        while i < (len(split_txt) - 1):
            text += split_txt[i]
            i += 1
    elif "BIBLIOGRAPHY" in text:
        print("removing references")
        split_txt = text.split("BIBLIOGRAPHY")
        i = 0
        text = ""
        while i < (len(split_txt) - 1):
            text += split_txt[i]
            i += 1
    elif "B I B L I O G R A P H Y" in text:
        print("removing references")
        split_txt = text.split("B I B L I O G R A P H Y")
        i = 0
        text = ""
        while i < (len(split_txt) - 1):
            text += split_txt[i]
            i += 1
    elif "b i b l i o g r a p h y" in text:
        print("removing references")
        split_txt = text.split("b i b l i o g r a p h y")
        i = 0
        text = ""
        while i < (len(split_txt) - 1):
            text += split_txt[i]
            i += 1

    print("saving into files ...", )

    print("... saving ", abstr_filename)
    # Write the file out again
    clean_file = open(abstr_filename, 'w')
    clean_file.write(abstract)
    clean_file.close()

    print("... saving ", txt_with_abstr_filename)
    clean_file = open(txt_with_abstr_filename, 'w')
    clean_file.write(abstract + "\n\n" + text)
    clean_file.close()

    print("... saving ", txt_only_filename)
    clean_file = open(txt_only_filename, 'w')
    clean_file.write(text)
    clean_file.close()
    print("... saving done.")

# !start cpdf.exe -draft PDF/qi2019a.pdf -o cleaned/qi2019a.pdf
