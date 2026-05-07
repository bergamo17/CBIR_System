# CBIR System

Content-Based Image Retrieval using CLIP + FAISS + MobileSAM

## Requirements
- Python 3.9+
- PostgreSQL

## Setup
1. Clone this repository
2. Install all the dependencies: pip install -r requirements.txt
3. Download the mobile_sam.pt and put it in the root folder
4. Prepare the Dataset and the testing image in root folder
5. Copy .env.example and fill the database credentials
6. Run the system: python app.py

## Notes
mobile_sam.pt, Datase_new/, and Official Image/ are not included in this repository.