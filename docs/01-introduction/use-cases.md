# Use Cases

**Pipeline Stage**: 1 - Introduction & Concepts

**Prerequisites**: 
- [Introduction](index.md)
- [Key Concepts](key-concepts.md)

This page demonstrates how Docling Graph solves real-world problems across different domains.

## Why Domain-Specific Knowledge Graphs?

Different domains have unique requirements for knowledge representation:

- **Chemistry**: Track exact molecular structures and reaction conditions
- **Finance**: Map complex instrument dependencies and risk relationships
- **Legal**: Maintain precise contractual obligations and party relationships
- **Research**: Connect methodologies, results, and citations with full context

Traditional text embeddings lose these precise relationships. Knowledge graphs preserve them.

## Chemistry & Materials Science

### Challenge
Research papers contain complex information about materials, their properties, measurements, and synthesis processes. Understanding "Material A has Property B measured at Condition C" requires exact relationship tracking.

### Solution with Docling Graph

**Template Structure**:
```python
class Material(BaseModel):
    """A chemical material or compound"""
    model_config = {'is_entity': True, 'graph_id_fields': ['name', 'grade']}
    
    name: str = Field(description="Material name or chemical formula")
    grade: Optional[str] = Field(None, description="Purity or grade")
    cas_number: Optional[str] = Field(None, description="CAS registry number")

class Measurement(BaseModel):
    """A measured property"""
    model_config = {'is_entity': False}
    
    property_name: str = Field(description="Property being measured")
    value: float = Field(description="Measured value")
    unit: str = Field(description="Unit of measurement")
    conditions: Optional[str] = Field(None, description="Measurement conditions")

class Experiment(BaseModel):
    """An experimental procedure"""
    model_config = {'is_entity': True, 'graph_id_fields': ['experiment_id']}
    
    experiment_id: str
    materials: List[Material] = edge(label="USES_MATERIAL", default_factory=list)
    measurements: List[Measurement] = edge(label="HAS_MEASUREMENT", default_factory=list)
    temperature: Optional[str] = Field(None, description="Process temperature")
```

**Resulting Graph**:
```
Experiment_EXP001
  ├─ USES_MATERIAL → Material_Lithium_99.9%
  ├─ USES_MATERIAL → Material_Graphite_Battery
  ├─ HAS_MEASUREMENT → Measurement_Viscosity_1.6mPa.s_25C
  └─ HAS_MEASUREMENT → Measurement_Conductivity_10mS/cm_25C
```

**Benefits**:
- Track exact material-property relationships
- Query "What materials were used at 25°C?"
- Find all experiments using Lithium
- Compare measurements across conditions

### Real Example: Battery Research

Extract from: "Rheological Properties of Lithium-Ion Battery Electrolytes"

**Extracted Graph**:
- 15 Material nodes (electrolytes, solvents, salts)
- 45 Measurement nodes (viscosity, conductivity, temperature)
- 8 Experiment nodes
- 120+ relationships

**Queries Enabled**:
- "Which electrolytes have viscosity < 2 mPa·s at 25°C?"
- "What are all measurements for LiPF6-based electrolytes?"
- "Which experiments used both EC and DMC solvents?"

## Finance & Legal

### Challenge
Financial documents contain complex instrument relationships, party obligations, and temporal dependencies. Understanding "Entity A owes Entity B under Contract C with Condition D" requires precise tracking.

### Solution with Docling Graph

**Template Structure**:
```python
class Party(BaseModel):
    """A legal or financial entity"""
    model_config = {'is_entity': True, 'graph_id_fields': ['name', 'tax_id']}
    
    name: str = Field(description="Legal entity name")
    tax_id: Optional[str] = Field(None, description="Tax identification number")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction")

class Obligation(BaseModel):
    """A contractual obligation"""
    model_config = {'is_entity': True, 'graph_id_fields': ['obligation_id']}
    
    obligation_id: str
    description: str = Field(description="Obligation description")
    amount: Optional[MonetaryAmount] = Field(None)
    due_date: Optional[date] = Field(None)
    
    obligor: Party = edge(label="OBLIGATED_BY")
    obligee: Party = edge(label="OBLIGATED_TO")

class Contract(BaseModel):
    """A legal contract"""
    model_config = {'is_entity': True, 'graph_id_fields': ['contract_number']}
    
    contract_number: str
    effective_date: date
    parties: List[Party] = edge(label="HAS_PARTY", default_factory=list)
    obligations: List[Obligation] = edge(label="CONTAINS_OBLIGATION", default_factory=list)
```

**Resulting Graph**:
```
Contract_2024-001
  ├─ HAS_PARTY → Party_AcmeCorp_123456789
  ├─ HAS_PARTY → Party_TechSolutions_987654321
  └─ CONTAINS_OBLIGATION → Obligation_OBL001
      ├─ OBLIGATED_BY → Party_AcmeCorp_123456789
      └─ OBLIGATED_TO → Party_TechSolutions_987654321
```

**Benefits**:
- Track party-obligation relationships
- Query "What are all obligations of Acme Corp?"
- Find contracts expiring in Q1 2024
- Identify circular dependencies

### Real Example: Loan Agreement

Extract from: "Commercial Loan Agreement"

**Extracted Graph**:
- 4 Party nodes (borrower, lender, guarantors)
- 12 Obligation nodes (payments, covenants, conditions)
- 1 Contract node
- 25+ relationships

**Queries Enabled**:
- "Who guarantees this loan?"
- "What are the payment obligations?"
- "Which covenants apply to the borrower?"

## Research & Academia

### Challenge
Research papers contain complex networks of authors, institutions, methodologies, results, and citations. Understanding "Author A from Institution B used Method C to achieve Result D" requires relationship tracking.

### Solution with Docling Graph

**Template Structure**:
```python
class Author(BaseModel):
    """A research author"""
    model_config = {'is_entity': True, 'graph_id_fields': ['name', 'affiliation']}
    
    name: str = Field(description="Author's full name")
    affiliation: str = Field(description="Institution affiliation")
    email: Optional[str] = Field(None)

class Methodology(BaseModel):
    """A research methodology"""
    model_config = {'is_entity': True, 'graph_id_fields': ['name']}
    
    name: str = Field(description="Methodology name")
    description: str = Field(description="Methodology description")
    parameters: Optional[str] = Field(None)

class Result(BaseModel):
    """A research finding"""
    model_config = {'is_entity': False}
    
    finding: str = Field(description="Key finding or result")
    metric: Optional[str] = Field(None)
    value: Optional[str] = Field(None)

class ResearchPaper(BaseModel):
    """A research publication"""
    model_config = {'is_entity': True, 'graph_id_fields': ['title']}
    
    title: str
    abstract: str
    year: int
    
    authors: List[Author] = edge(label="HAS_AUTHOR", default_factory=list)
    methodologies: List[Methodology] = edge(label="USES_METHODOLOGY", default_factory=list)
    results: List[Result] = edge(label="HAS_RESULT", default_factory=list)
```

**Resulting Graph**:
```
ResearchPaper_AdvancedAI
  ├─ HAS_AUTHOR → Author_DrSmith_MIT
  ├─ HAS_AUTHOR → Author_DrJones_Stanford
  ├─ USES_METHODOLOGY → Methodology_DeepLearning
  ├─ USES_METHODOLOGY → Methodology_TransferLearning
  ├─ HAS_RESULT → Result_Accuracy95%
  └─ HAS_RESULT → Result_TrainingTime50%Reduced
```

**Benefits**:
- Track author collaboration networks
- Query "What methodologies did MIT researchers use?"
- Find papers with similar results
- Identify research trends

### Real Example: AI Research Paper

Extract from: "Advances in Neural Architecture Search"

**Extracted Graph**:
- 8 Author nodes
- 5 Methodology nodes
- 12 Result nodes
- 1 Paper node
- 35+ relationships

**Queries Enabled**:
- "Who collaborated with Dr. Smith?"
- "What methodologies achieved >90% accuracy?"
- "Which institutions are researching NAS?"

## Healthcare & Medical

### Challenge
Medical records contain patient information, diagnoses, treatments, medications, and outcomes. Understanding "Patient A received Treatment B for Condition C with Outcome D" requires precise tracking.

### Solution with Docling Graph

**Template Structure**:
```python
class Patient(BaseModel):
    """A patient (anonymized)"""
    model_config = {'is_entity': True, 'graph_id_fields': ['patient_id']}
    
    patient_id: str = Field(description="Anonymized patient identifier")
    age: Optional[int] = Field(None)
    gender: Optional[str] = Field(None)

class Diagnosis(BaseModel):
    """A medical diagnosis"""
    model_config = {'is_entity': True, 'graph_id_fields': ['code']}
    
    code: str = Field(description="ICD-10 or similar code")
    description: str = Field(description="Diagnosis description")
    date: date

class Treatment(BaseModel):
    """A medical treatment"""
    model_config = {'is_entity': False}
    
    treatment_type: str = Field(description="Type of treatment")
    description: str
    start_date: date
    end_date: Optional[date] = Field(None)

class MedicalRecord(BaseModel):
    """A medical record"""
    model_config = {'is_entity': True, 'graph_id_fields': ['record_id']}
    
    record_id: str
    date: date
    
    patient: Patient = edge(label="FOR_PATIENT")
    diagnoses: List[Diagnosis] = edge(label="HAS_DIAGNOSIS", default_factory=list)
    treatments: List[Treatment] = edge(label="HAS_TREATMENT", default_factory=list)
```

**Benefits**:
- Track patient-diagnosis-treatment relationships
- Query "What treatments were used for diagnosis X?"
- Find similar patient cases
- Analyze treatment outcomes

## Insurance & Risk

### Challenge
Insurance policies contain coverage details, exclusions, parties, and conditions. Understanding "Policy A covers Risk B for Party C under Condition D" requires relationship tracking.

### Solution with Docling Graph

**Template Structure**:
```python
class Coverage(BaseModel):
    """An insurance coverage item"""
    model_config = {'is_entity': False}
    
    coverage_type: str = Field(description="Type of coverage")
    description: str
    limit: Optional[MonetaryAmount] = Field(None)
    deductible: Optional[MonetaryAmount] = Field(None)

class Exclusion(BaseModel):
    """A policy exclusion"""
    model_config = {'is_entity': False}
    
    description: str = Field(description="What is excluded")
    conditions: Optional[str] = Field(None)

class InsurancePolicy(BaseModel):
    """An insurance policy"""
    model_config = {'is_entity': True, 'graph_id_fields': ['policy_number']}
    
    policy_number: str
    effective_date: date
    expiration_date: date
    
    policyholder: Person = edge(label="HELD_BY")
    coverages: List[Coverage] = edge(label="PROVIDES_COVERAGE", default_factory=list)
    exclusions: List[Exclusion] = edge(label="HAS_EXCLUSION", default_factory=list)
```

**Benefits**:
- Track policy-coverage relationships
- Query "What policies cover fire damage?"
- Find coverage gaps
- Analyze risk exposure

## Common Patterns Across Domains

### Pattern 1: Document → Entities → Properties

```
Document
  ├─ HAS_ENTITY → Entity1
  │   ├─ HAS_PROPERTY → Property1
  │   └─ HAS_PROPERTY → Property2
  └─ HAS_ENTITY → Entity2
      └─ HAS_PROPERTY → Property3
```

**Used in**: Research papers, technical reports, specifications

### Pattern 2: Party → Relationship → Party

```
Party1 ─[RELATIONSHIP]→ Party2
```

**Used in**: Contracts, agreements, organizational charts

### Pattern 3: Process → Steps → Outcomes

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

### Questions to Ask

1. **What entities do I need to track?**
   - People, organizations, documents, materials, etc.

2. **What relationships matter?**
   - Who issued what? Who owns what? What contains what?

3. **What queries will I run?**
   - "Find all X related to Y"
   - "What are the properties of Z?"

4. **What level of detail do I need?**
   - High-level overview or detailed properties?

### Decision Matrix

| Domain | Entity Types | Key Relationships | Complexity |
|--------|-------------|-------------------|------------|
| Chemistry | Materials, Measurements | USES, HAS_PROPERTY | High |
| Finance | Parties, Obligations | OWES, GUARANTEES | Medium |
| Research | Authors, Papers, Methods | AUTHORED_BY, CITES | Medium |
| Healthcare | Patients, Diagnoses | HAS_DIAGNOSIS, TREATED_WITH | High |
| Insurance | Policies, Coverage | COVERS, EXCLUDES | Medium |
| Legal | Parties, Contracts | PARTY_TO, OBLIGATED_BY | High |

## Next Steps

Ready to implement your use case?

1. **[Schema Definition](../03-schema-definition/index.md)** - Create your Pydantic templates
2. **[Examples](../09-examples/index.md)** - See complete working examples
3. **[Architecture Overview](architecture-overview.md)** - Understand the system design

## Related Documentation

- **[Key Concepts](key-concepts.md)**: Understand entities, edges, and graphs
- **[Pipeline Configuration](../04-pipeline-configuration/index.md)**: Configure for your domain
- **[Graph Management](../06-graph-management/index.md)**: Export and query your graphs

---

**Have a unique use case?** The patterns above can be adapted to any domain requiring precise relationship tracking!