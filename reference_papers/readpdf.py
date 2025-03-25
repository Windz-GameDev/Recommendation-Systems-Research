def readPDF():
    import PyPDF2
    import os

    # Get the current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    # PDF file path
    pdf_path = 'paper/LLMBasedGeneration of Item-Description for Recommendation.pdf'
    
    # Create output text file name with the same base name
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file = f"paper/{base_name}.txt"
    
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        print(f"Number of pages in PDF: {len(reader.pages)}")
        
        # Open text file for writing
        with open(output_file, 'w', encoding='utf-8') as txt_file:
            # Process each page
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                txt_file.write(f"--- Page {i+1} ---\n")
                txt_file.write(text)
                txt_file.write("\n\n")
            
            print(f"PDF content written to: {output_file}")

readPDF()