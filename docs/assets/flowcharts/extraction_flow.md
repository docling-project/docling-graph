%%{init: {'theme': 'redux-dark', 'look': 'default', 'layout': 'elk'}}%%
flowchart TD
    %% 1. Define Classes
    classDef input fill:#E3F2FD,stroke:#90CAF9,color:#0D47A1
    classDef config fill:#FFF8E1,stroke:#FFECB3,color:#5D4037
    classDef output fill:#E8F5E9,stroke:#A5D6A7,color:#1B5E20
    classDef decision fill:#FFE0B2,stroke:#FFB74D,color:#E65100
    classDef data fill:#EDE7F6,stroke:#B39DDB,color:#4527A0
    classDef operator fill:#F3E5F5,stroke:#CE93D8,color:#6A1B9A
    classDef process fill:#ECEFF1,stroke:#B0BEC5,color:#263238

    %% 2. Define Nodes
    Start@{ shape: terminal, label: "Input Source" }
    
    Normalize@{ shape: procs, label: "Input Normalization" }
    CheckInput{"Input Type"}
    
    Convert@{ shape: procs, label: "Document Conversion<br/>PDF/Image" }
    TextProc@{ shape: lin-proc, label: "Text Processing<br/>Text/Markdown" }
    LoadDoc@{ shape: lin-proc, label: "Load DoclingDocument<br/>Skip to Graph" }
    
    CheckMode{"Process. Mode"}
    CheckChunk{"Chunking?"}
    
    PageExtract@{ shape: lin-proc, label: "Page-by-Page Extraction" }
    FullDoc@{ shape: lin-proc, label: "Full Document Extraction" }
    
    Chunk@{ shape: tag-proc, label: "Structure-Aware Chunking" }
    Batch@{ shape: tag-proc, label: "Batch Chunks" }
    
    Extract@{ shape: procs, label: "Extract from Batches" }
    
    CheckMerge{"Multiple Models?"}
    
    Merge@{ shape: lin-proc, label: "Programmatic Merge" }
    Single@{ shape: doc, label: "Single Model" }
    
    CheckConsol{"Consolidation?"}
    Consol@{ shape: procs, label: "LLM Consolidation" }
    
    Final@{ shape: doc, label: "Final Model" }
    Graph@{ shape: db, label: "Knowledge Graph" }

    %% 3. Define Connections
    Start --> Normalize
    Normalize --> CheckInput
    
    CheckInput -- "PDF/Image" --> Convert
    CheckInput -- "Text/Markdown" --> TextProc
    CheckInput -- "DoclingDocument" --> LoadDoc
    
    Convert --> CheckMode
    TextProc --> CheckMode
    
    LoadDoc --> Graph
    
    CheckMode -- Many-to-One --> CheckChunk
    CheckMode -- One-to-One --> PageExtract
    
    CheckChunk -- Yes --> Chunk
    CheckChunk -- No --> FullDoc
    
    Chunk --> Batch
    Batch --> Extract
    
    FullDoc --> Extract
    PageExtract --> Extract
    
    Extract --> CheckMerge
    CheckMerge -- Yes --> Merge
    CheckMerge -- No --> Single
    
    Merge --> CheckConsol
    CheckConsol -- Yes --> Consol
    CheckConsol -- No --> Final
    
    Consol --> Final
    Single --> Final
    Final --> Graph

    %% 4. Apply Classes
    class Start input
    class Normalize,Convert,Extract,Consol process
    class TextProc,LoadDoc,PageExtract,FullDoc,Merge process
    class Chunk,Batch operator
    class CheckInput,CheckMode,CheckChunk,CheckMerge,CheckConsol decision
    class Single data
    class Final,Graph output
```