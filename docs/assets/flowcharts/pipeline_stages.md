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
    A("1. Install")
    
    B@{ shape: lin-proc, label: "2. Define Schema" }
    C@{ shape: lin-proc, label: "3. Configure Pipeline" }
    
    D@{ shape: procs, label: "4. Extract Data" }
    E@{ shape: procs, label: "5. Build Graph" }
    
    F@{ shape: tag-proc, label: "6. Export & Visualize" }

    %% 3. Define Connections
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F

    %% 4. Apply Classes
    class A input
    class B,C config
    class D,E process
    class F output
```