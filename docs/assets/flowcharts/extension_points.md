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
    
    B@{ shape: lin-proc, label: "Custom Stage 1" }
    C@{ shape: procs, label: "Docling Conversion" }
    D@{ shape: tag-proc, label: "Custom Backend" }
    E@{ shape: procs, label: "Extraction" }
    F@{ shape: lin-proc, label: "Custom Stage 2" }
    G@{ shape: procs, label: "Graph Conversion" }
    H@{ shape: tag-proc, label: "Custom Exporter" }
    
    I@{ shape: doc, label: "Output" }

    %% 3. Define Connections
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I

    %% 4. Apply Classes
    class A input
    class B,F config
    class C,E,G process
    class D,H operator
    class I output
```