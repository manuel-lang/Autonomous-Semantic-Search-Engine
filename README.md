# NextIterationHackathon2018

## Textract

Supported file types
- .csv via python builtins
- .doc via antiword
- .docx via python-docx2txt
- .eml via python builtins
- .epub via ebooklib
- .gif via tesseract-ocr
- .jpg and .jpeg via tesseract-ocr
- .json via python builtins
- .html and .htm via beautifulsoup4
- .mp3 via sox, SpeechRecognition, and pocketsphinx
- .msg via msg-extractor
- .odt via python builtins
- .ogg via sox, SpeechRecognition, and pocketsphinx
- .pdf via pdftotext (default) or pdfminer.six
- .png via tesseract-ocr
- .pptx via python-pptx
- .ps via ps2text
- .rtf via unrtf
- .tiff and .tif via tesseract-ocr
- .txt via python builtins
- .wav via SpeechRecognition and pocketsphinx
- .xlsx via xlrd
- .xls via xlrd

# brew Dependencies
- Swig etc. for Textract: https://textract.readthedocs.io/en/stable/installation.html
- Ghostscript: https://wiki.scribus.net/canvas/Installation_and_Configuration_of_Ghostscript
- ImageMagick 6: https://github.com/ImageMagick/ImageMagick/issues/953
