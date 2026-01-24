```mermaid
%%{init: {'theme': 'redux-dark', 'look': 'default', 'layout': 'elk'}}%%
flowchart TB
    %% 1. Define Classes
    classDef input fill:#E3F2FD,stroke:#90CAF9,color:#0D47A1
    classDef config fill:#FFF8E1,stroke:#FFECB3,color:#5D4037
    classDef output fill:#E8F5E9,stroke:#A5D6A7,color:#1B5E20
    classDef decision fill:#FFE0B2,stroke:#FFB74D,color:#E65100
    classDef data fill:#EDE7F6,stroke:#B39DDB,color:#4527A0
    classDef operator fill:#F3E5F5,stroke:#CE93D8,color:#6A1B9A
    classDef process fill:#ECEFF1,stroke:#B0BEC5,color:#263238

    %% 2. Define Nodes
    A@{ shape: terminal, label: "Input Source" }
    A1@{ shape: procs, label: "1. Input Normalization<br/>Type Detection & Validation" }
    
    A2{"Input Type"}
    
    B@{ shape: procs, label: "2. Docling Conversion<br/>OCR or Vision" }
    B2@{ shape: lin-proc, label: "2b. Text Processing<br/>Direct to Markdown" }
    B3@{ shape: lin-proc, label: "2c. Load DoclingDocument<br/>Skip Conversion" }
    
    C{"3. Backend"}
    
    D@{ shape: lin-proc, label: "4a. VLM Extraction<br/>Direct from Document" }
    E@{ shape: lin-proc, label: "4b. Markdown Extraction" }
    
    F{"5. Chunking"}
    
    G@{ shape: tag-proc, label: "6a. Hybrid Chunking<br/>Semantic + Token-Aware" }
    H@{ shape: tag-proc, label: "6b. Full Document" }
    
    I@{ shape: procs, label: "7. Batch Extraction<br/>Process Each Chunk" }
    J@{ shape: tag-proc, label: "8. Pydantic Validation<br/>Type Checking" }
    
    K{"9. Consolidation"}
    
    L@{ shape: lin-proc, label: "10a. Smart Merge<br/>Rule-Based" }
    M@{ shape: lin-proc, label: "10b. LLM Consolidation<br/>Intelligent" }
    
    N@{ shape: procs, label: "11. Graph Conversion<br/>Pydantic → NetworkX" }
    O@{ shape: tag-proc, label: "12. Node ID Generation<br/>Stable Identifiers" }
    
    P@{ shape: tag-proc, label: "13. Export<br/>CSV/Cypher/JSON" }
    Q@{ shape: tag-proc, label: "14. Visualization<br/>HTML + Reports" }

    %% 3. Define Connections
    A --> A1
    A1 --> A2
    
    A2 -- "PDF/Image" --> B
    A2 -- "Text/Markdown" --> B2
    A2 -- "DoclingDocument" --> B3
    
    B --> C
    B2 --> C
    
    B3 --> E
    
    C -- VLM --> D
    C -- LLM --> E
    
    E --> F
    F -- Yes --> G
    F -- No --> H
    
    G --> I
    H --> I
    
    D --> J
    I --> J
    J --> K
    
    K -- Programmatic --> L
    K -- LLM --> M
    
    L --> N
    M --> N
    
    N --> O
    O --> P
    P --> Q

    %% 4. Apply Classes
    class A input
    class A1,B,I,N process
    class B2,B3,D,E,L,M process
    class A2,C,F,K decision
    class G,H,J,O operator
    class P,Q output
```