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
    A@{ shape: terminal, label: "Pydantic Models" }
    
    B@{ shape: lin-proc, label: "Node ID<br/>Generation" }
    C@{ shape: lin-proc, label: "Node<br/>Creation" }
    D@{ shape: lin-proc, label: "Edge<br/>Creation" }
    E@{ shape: tag-proc, label: "Graph<br/>Validation" }
    
    F("NetworkX<br/>DiGraph")

    %% 3. Define Connections
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F

    %% 4. Apply Classes
    class A input
    class B,C,D process
    class E operator
    class F output
```