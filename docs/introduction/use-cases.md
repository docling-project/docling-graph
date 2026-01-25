# Use Cases

## Why Domain-Specific Knowledge Graphs?

Different domains have unique requirements for knowledge representation:

- **Chemistry**: Track exact molecular structures and reaction conditions
- **Finance**: Map complex instrument dependencies and risk relationships
- **Legal**: Maintain precise contractual obligations and party relationships
- **Research**: Connect methodologies, results, and citations with full context

Traditional text embeddings lose these precise relationships. Knowledge graphs preserve them.


## Chemistry & Materials Science

Extract materials, measurements, and experimental relationships from research papers.

**Key Entities:** Material, Measurement, Experiment
**Relationships:** USES_MATERIAL, HAS_MEASUREMENT
**Use Case:** Track exact material-property relationships and experimental conditions

## Finance & Legal

Extract parties, obligations, and contractual relationships from legal documents.

**Key Entities:** Party, Obligation, Contract
**Relationships:** OBLIGATED_BY, OBLIGATED_TO, HAS_PARTY
**Use Case:** Track party-obligation relationships and contractual dependencies

## Research & Academia

Extract authors, methodologies, and findings from academic papers.

**Key Entities:** Author, Methodology, Result, ResearchPaper
**Relationships:** HAS_AUTHOR, USES_METHODOLOGY, HAS_RESULT
**Use Case:** Track author collaboration networks and research methodologies

## Healthcare & Medical

Extract patient information, diagnoses, and treatments from medical records.

**Key Entities:** Patient, Diagnosis, Treatment, MedicalRecord
**Relationships:** FOR_PATIENT, HAS_DIAGNOSIS, HAS_TREATMENT
**Use Case:** Track patient-diagnosis-treatment relationships

## Insurance & Risk

Extract coverage details, exclusions, and policy relationships.

**Key Entities:** Coverage, Exclusion, InsurancePolicy
**Relationships:** HELD_BY, PROVIDES_COVERAGE, HAS_EXCLUSION
**Use Case:** Track policy-coverage relationships and risk exposure

## Common Patterns

### 📍 Document → Entities → Properties

```
Document
  ├─ HAS_ENTITY → Entity1
  │   ├─ HAS_PROPERTY → Property1
  │   └─ HAS_PROPERTY → Property2
  └─ HAS_ENTITY → Entity2
      └─ HAS_PROPERTY → Property3
```

**Used in**: Research papers, technical reports, specifications

### 📍 Party → Relationship → Party

```
Party1 ─[RELATIONSHIP]→ Party2
```

**Used in**: Contracts, agreements, organizational charts

### 📍 Process → Steps → Outcomes

```
Process
  ├─ HAS_STEP → Step1
  ├─ HAS_STEP → Step2
  └─ HAS_OUTCOME → Outcome
```

**Used in**: Procedures, experiments, workflows

### Pattern 4: Hierarchical Structures

```
Parent
  ├─ HAS_CHILD → Child1
  │   └─ HAS_CHILD → Grandchild1
  └─ HAS_CHILD → Child2
```

**Used in**: Organizational structures, document sections, taxonomies


## Choosing Your Use Case

**Questions to Ask:**

1. What entities do I need to track?
2. What relationships matter?
3. What queries will I run?
4. What level of detail do I need?


Ready to implement your use case?

1. **[Schema Definition](../fundamentals/schema-definition/index.md)** - Create your Pydantic templates
2. **[Examples](../usage/examples/index.md)** - See complete working examples
3. **[Architecture](architecture.md)** - Understand the system design