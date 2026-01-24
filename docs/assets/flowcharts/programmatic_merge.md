```mermaid
%%{init: {'theme': 'redux-dark', 'look': 'default', 'layout': 'elk'}}%%
flowchart LR
    %% 1. Define Classes
    classDef input fill:#E3F2FD,stroke:#90CAF9,color:#0D47A1
    classDef config fill:#FFF8E1,stroke:#FFECB3,color:#5D4037
    classDef output fill:#E8F5E9,stroke:#A5D6A7,color:#1B5E20
    classDef decision fill:#FFE0B2,stroke:#FFB74D,color:#E65100
    classDef data fill:#EDE7F6,stroke:#B39DDB,color:#4527A0
    classDef operator fill:#F3E5F5,stroke:#CE93D8,color:#6A1B9A
    classDef process fill:#ECEFF1,stroke:#B0BEC5,color:#263238

    %% 2. Define Nodes
    A@{ shape: terminal, label: "Model 1" }
    B@{ shape: terminal, label: "Model 2" }
    C@{ shape: terminal, label: "Model 3" }
    
    D@{ shape: lin-proc, label: "Deep Merge" }
    E@{ shape: tag-proc, label: "Deduplicate" }
    F@{ shape: tag-proc, label: "Validate" }
    
    G@{ shape: doc, label: "Final Model" }

    %% 3. Define Connections
    A & B & C --> D
    
    D --> E
    E --> F
    F --> G

    %% 4. Apply Classes
    class A,B,C data
    class D process
    class E,F operator
    class G output
```