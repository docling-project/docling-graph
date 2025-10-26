---
config:
  theme: redux
  layout: elk
  look: classic
  elk: {}
---
flowchart TD
    A(["Source Document"]) --> n4["Docling Graph Pipeline"]
    n2(["Config"]) --> n4
    n3(["Pydantic Template"]) --> n4
    n4 --> n5["Extraction Factory"]
    n5 --> n6["Docling Pipeline"]
    n6 --> n7["OCR"] & n8["Vision"] & n25["Extract"]
    n8 --> n9["Markdown Processor"]
    n7 --> n9
    n10["Conversion Strategy"] --> n11["One To One"] & n12["Many To One"]
    n13["Extraction Backend"] --> n14["LLM"] & n15["VLM"]
    n9 --> n13
    n16(["Prompt"]) --> n13
    n15 --> n17(["Extracted Content"])
    n14 --> n17
    n17 --> n10
    n12 --> n18["Smart Template Merger"]
    n18 --> n20(["Populated Pydantic Model(s)"])
    n11 --> n20
    n20 --> n21["Graph Converter"]
    n21 --> n22(["Knowledge Graph"])
    n22 --> n23["Exporter"] & n24["Visualizer"]
    n25 --> n10
    n23 --> n29(["CSV"]) & n30(["Cypher"]) & n31(["JSON"]) & n33["Knowledge Base"]
    n24 --> n28(["Images"]) & n27(["HTML"]) & n26(["Markdown"])
    n30 --> n34["Batch Loader"]
    n31 --> n34
    n29 --> n34
    n34 --> n33
    n5 --> n16
    n4@{ shape: procs}
    n5@{ shape: tag-proc}
    n6@{ shape: procs}
    n7@{ shape: lin-proc}
    n8@{ shape: lin-proc}
    n25@{ shape: lin-proc}
    n9@{ shape: tag-proc}
    n10@{ shape: procs}
    n11@{ shape: lin-proc}
    n12@{ shape: lin-proc}
    n13@{ shape: procs}
    n14@{ shape: lin-proc}
    n15@{ shape: lin-proc}
    n18@{ shape: tag-proc}
    n21@{ shape: tag-proc}
    n23@{ shape: tag-proc}
    n24@{ shape: tag-proc}
    n33@{ shape: db}
    n34@{ shape: tag-proc}
     A:::input
     n4:::process
     n2:::config
     n3:::input
     n5:::operator
     n6:::process
     n7:::process
     n8:::process
     n25:::process
     n9:::operator
     n10:::process
     n11:::process
     n12:::process
     n13:::process
     n14:::process
     n15:::process
     n16:::config
     n17:::output
     n18:::operator
     n20:::output
     n21:::operator
     n22:::output
     n23:::operator
     n24:::operator
     n29:::output
     n30:::output
     n31:::output
     n33:::process
     n28:::output
     n27:::output
     n26:::output
     n34:::operator
    classDef input fill:#E3F2FD,stroke:#90CAF9,color:#0D47A1
    classDef config fill:#FFF8E1,stroke:#FFECB3,color:#5D4037
    classDef output fill:#E8F5E9,stroke:#A5D6A7,color:#1B5E20
    classDef decision fill:#FFE0B2,stroke:#FFB74D,color:#E65100
    classDef data fill:#EDE7F6,stroke:#B39DDB,color:#4527A0
    classDef operator fill:#F3E5F5,stroke:#CE93D8,color:#6A1B9A
    classDef process fill:#ECEFF1,stroke:#B0BEC5,color:#263238