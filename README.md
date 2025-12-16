# CCTV Surveillance: Text/Image Guided Object Detection & Tracking

This project implements an intelligent **CCTV surveillance system** that detects and tracks objects in video footage based on **user-provided text or image queries**.  
The system follows a CLIP-style vision–language approach combined with object proposals and tracking to identify and follow relevant objects throughout a video.

---

## Features

- Text-based object search (e.g., *"a person wearing a red jacket"*)
- Image-based object search (query by example)
- Object proposal generation from video frames
- Vision–language embedding similarity (cosine similarity)
- Object tracking across entire video
- Modular and extensible pipeline

---

## System Overview

**High-level workflow:**

1. Extract frames from CCTV video  
2. Generate object proposals for each frame  
3. Convert user query (text or image) into embeddings  
4. Convert object crops into embeddings  
5. Match objects using cosine similarity  
6. Track the matched object across frames  

## System Architecture

```text
                 ┌─────────────────────┐
                 │     User Input      │
                 │  (Text / Image)     │
                 └─────────┬───────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │  Query Embedding      │
                │ (Text/Image Encoder)  │
                └─────────┬─────────────┘
                           │
                           │  Cosine Similarity
                           │
┌───────────────┐          ▼          ┌───────────────────────┐
│   CCTV Video  │ ──► Frame Extractor │  Object Proposals     │
│   Footage     │                     │ (Candidate Regions)  │
└───────────────┘                     └─────────┬─────────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │  Object Crop Generator │
                                    └─────────┬──────────────┘
                                              │
                                              ▼
                                ┌──────────────────────────┐
                                │  Object Embeddings       │
                                │  (Image Encoder)         │
                                └─────────┬────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────────┐
                              │  Similarity Ranking         │
                              │ (Query ↔ Object Embeddings) │
                              └─────────┬──────────────────┘
                                        │
                                        ▼
                              ┌────────────────────────────┐
                              │  Best Match Selection      │
                              └─────────┬──────────────────┘
                                        │
                                        ▼
                              ┌────────────────────────────┐
                              │  Object Tracking Module    │
                              │ (Across Video Frames)      │
                              └─────────┬──────────────────┘
                                        │
                                        ▼
                              ┌────────────────────────────┐
                              │  Output                    │
                              │  • Tracked Video           │
                              │  • Annotated Frames        │
                              └────────────────────────────┘
```

## Project Structure

```text
CCTV_surveillance/
│
├── clip_ranker.py            # Ranks object proposals using embedding similarity
├── extract_frames.py         # Extracts frames from input video
├── proposals.py              # Generates object proposals
├── proposals_to_crops.py     # Converts proposals into cropped images
├── track_from_ranked.py      # Tracks selected object across frames
├── practise.py               # Experimental / testing script
│
├── user_inputs/              # User text/image queries
├── outputs/                  # Generated results (ignored in GitHub)
├── sample_videos/            # Input videos (ignored in GitHub)
│
├── .gitignore
└── README.md

```
## Setup
```
python -m venv dlenv
source dlenv/Scripts/activate   # Git Bash / Linux
pip install -r requirements.txt
```
## Usage
```
python extract_frames.py
python proposals.py
python proposals_to_crops.py
python clip_ranker.py
python track_from_ranked.py
```
## Outputs
- Ranked object candidates
- Tracked object video
- Annotated frames
<img width="898" height="506" alt="image" src="https://github.com/user-attachments/assets/80f0c20a-5975-4171-a543-5d2e72509b05" />

## Tech Stack
- Python
- PyTorch
- OpenCV
- Vision–Language Models (CLIP-style)
- Object Proposals using **Selective search algorithm**
- Object Tracking
  
## Future Work
- Replace proposals with more advanced models.
- Train a custom vision–language model
- Real-time CCTV stream processing


