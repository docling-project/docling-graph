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
    
    %% Custom Subgraph Style (Transparent with dashed border)
    classDef subgraph_style fill:none,stroke:#969696,stroke-width:2px,stroke-dasharray: 5,color:#969696

    %% 2. Define Nodes
    A@{ shape: procs, label: "BaseLlmClient<br>Template Method Pattern" }
    
    subgraph subGraph0["Client Implementations"]
        B@{ shape: lin-proc, label: "VLLMClient" }
        C@{ shape: lin-proc, label: "OllamaClient" }
        D@{ shape: lin-proc, label: "MistralClient" }
        E@{ shape: lin-proc, label: "OpenAIClient" }
        F@{ shape: lin-proc, label: "GeminiClient" }
        G@{ shape: lin-proc, label: "WatsonXClient" }
    end

    H@{ shape: tag-proc, label: "ResponseHandler<br>JSON Parsing" }
    I("Config<br>models.yaml")

    %% 3. Define Connections
    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    A --> G
    subGraph0 --> H
    subGraph0 --> I

    %% 4. Apply Classes
    class A,B,C,D,E,F,G process
    class H operator
    class I config
    class subGraph0 subgraph_style
```