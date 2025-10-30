"""
Pydantic templates for Battery Slurry Ontology.

These models define a comprehensive ontology for battery slurry formulations, processing, and evaluation.
This version adds enum normalization, structured examples, and string coercion for select fields.
"""

from typing import List, Optional, Union, Any
from enum import Enum
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --- Edge Helper Function ---
def Edge(label: str, **kwargs: Any) -> Any:
    return Field(..., json_schema_extra={'edge_label': label}, **kwargs)


# --- Helpers: normalization and parsing ---

def _normalize_enum(enum_cls, v):
    """
    Accept:
      - enum instance
      - value strings (e.g., "Viscosity")
      - member-like strings (e.g., "VISCOSITY", "SOLID_CONTENT")
      - looser strings with spaces/underscores/case (e.g., "solid content", "Solid_Content")
    """
    if isinstance(v, enum_cls):
        return v
    if isinstance(v, str):
        key = re.sub(r'[^A-Za-z0-9]+', '', v).lower()
        mapping = {}
        for m in enum_cls:
            # map by member name and value
            mapping[re.sub(r'[^A-Za-z0-9]+', '', m.name).lower()] = m
            mapping[re.sub(r'[^A-Za-z0-9]+', '', m.value).lower()] = m
        if key in mapping:
            return mapping[key]
    # last-ditch attempt
    try:
        return enum_cls(v)
    except Exception:
        # Prefer safe fallback to OTHER if present
        if 'OTHER' in getattr(enum_cls, '__members__', {}):
            return enum_cls.__members__['OTHER']
        raise


def _parse_measurement_string(s: str, default_name: Optional[str] = None):
    """
    Parse strings like:
      "1.6 mPa.s", "38 wt%", "25 °C", "12 wt %", "80C"
    into a Measurement-like dict: {name, numeric_value, text_value, unit}.
    If numeric parsing fails, preserve as text_value.
    """
    if not isinstance(s, str):
        return s
    m = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)\s*([^\d]+)?$', s)
    if m:
        num = float(m.group(1))
        unit = (m.group(2) or '').strip() or None
        return {
            'name': default_name or 'Value',
            'numeric_value': num,
            'text_value': None,
            'unit': unit,
        }
    # no numeric part; keep raw as text
    return {
        'name': default_name or 'Value',
        'numeric_value': None,
        'text_value': s.strip(),
        'unit': None,
    }


# 1. --- Foundational Building Blocks ---

class Measurement(BaseModel):
    """A flexible model for any named property with a value and optional unit."""
    model_config = ConfigDict(is_entity=False)

    name: str = Field(
        description="The name of the property, e.g., 'Viscosity', 'pH', 'Solid Content'.",
        examples=["Viscosity", "pH", "Solid Content", "Temperature"],
    )

    text_value: Optional[str] = Field(
        default=None,
        description="The textual value of the property, if not numerical.",
        examples=["High", "Low", "Stable"],
    )

    numeric_value: Optional[Union[float, int]] = Field(
        default=None,
        description="The numerical value of the property (float or int).",
        examples=[1.6, 8.2, 35.5, 25],
    )

    unit: Optional[str] = Field(
        default=None,
        description="The unit of measurement, e.g., 'mPa.s', 'wt%', '°C', 'dL/g'.",
        examples=["mPa.s", "wt%", "°C", "dL/g"],
    )

    condition: Optional[str] = Field(
        default=None,
        description="Measurement condition, e.g., 'at 10 s⁻¹ shear rate', 'storage temperature 25°C'.",
        examples=["at 10 s⁻¹ shear rate", "after 5 min rest", "storage temperature 25°C"],
    )


class MaterialProperty(Measurement):
    """Represents a material property, inherits from Measurement."""
    pass


# 2. --- Materials and Composition ---

class MaterialRole(str, Enum):
    ACTIVE_MATERIAL = "Active Material"
    BINDER = "Binder"
    CONDUCTIVE_ADDITIVE = "Conductive Additive"
    SOLVENT = "Solvent"
    DISPERSING_MEDIUM = "Dispersing Medium"
    CO_SOLVENT = "Co-solvent"
    DISPERSANT = "Dispersant"
    SURFACTANT = "Surfactant"
    THICKENER = "Thickener"
    RHEOLOGY_MODIFIER = "Rheology Modifier"
    STABILIZER = "Stabilizer"
    WETTING_AGENT = "Wetting Agent"
    DEFOAMER = "Defoamer"
    LITHIUM_SUPPLEMENT = "Lithium Supplement"
    ELECTROLYTE_ADDITIVE = "Electrolyte Additive"
    OTHER = "Other"


class Material(BaseModel):
    """Any chemical or substance part of the composition."""
    model_config = ConfigDict(graph_id_fields=['name'])

    name: str = Field(
        description="Canonical name of the material.",
        examples=["Polyvinylidene Fluoride", "LiFePO4", "Carbon Black", "N-Methyl-2-pyrrolidone"],
    )

    category: Optional[str] = Field(
        default=None,
        description="Broad classification, e.g., 'Fluoropolymer', 'Additive'.",
        examples=["Fluoropolymer", "Olivine Phosphate", "Additive", "Solvent"],
    )

    chemical_formula: Optional[str] = Field(
        default=None,
        description="Chemical formula.",
        examples=["(C2H2F2)n", "LiFePO4", "C", "C5H9NO"],
    )

    properties: List[MaterialProperty] = Field(
        default_factory=list,
        description="Properties, e.g., particle size, molecular weight.",
        examples=[[{"name": "Particle Size (D50)", "numeric_value": 3.2, "unit": "µm"}]],
    )


class ComponentAmount(Measurement):
    """Amount of a component, inherits Measurement."""
    pass


class Component(BaseModel):
    """Links a material to its role and amount."""
    model_config = ConfigDict(graph_id_fields=['material', 'role'])

    material: Material = Edge(
        label="USES_MATERIAL",
        description="Material used in this component.",
        examples=[{"name": "LiFePO4", "chemical_formula": "LiFePO4", "category": "Olivine Phosphate"}],
    )

    role: MaterialRole = Field(
        description="Function of the material in the slurry.",
        examples=["Active Material", "Binder", "Conductive Additive"],
    )

    amount: ComponentAmount = Field(
        description="Amount specification (weight/volume fraction).",
        examples=[{"name": "Weight Fraction", "numeric_value": 12.0, "unit": "wt%"}],
    )

    @field_validator('role', mode='before')
    @classmethod
    def _role_norm(cls, v):
        return _normalize_enum(MaterialRole, v)

    @field_validator('amount', mode='before')
    @classmethod
    def _amount_coerce(cls, v):
        # Accept strings like "12 wt%" and coerce into ComponentAmount
        if isinstance(v, dict) or v is None:
            return v
        if isinstance(v, str):
            return _parse_measurement_string(v, default_name="Weight Fraction")
        return v


class Property(Measurement):
    """Represents a slurry property."""
    pass


class Slurry(BaseModel):
    """Collection of slurry components."""
    model_config = ConfigDict(graph_id_fields=['components'])

    components: List[Component] = Edge(
        label="HAS_COMPONENT",
        description="Slurry components list.",
        examples=[[
            {
                "material": {"name": "LiFePO4", "chemical_formula": "LiFePO4", "category": "Olivine Phosphate"},
                "role": "Active Material",
                "amount": {"name": "Weight Fraction", "numeric_value": 12.0, "unit": "wt%"},
            },
            {
                "material": {"name": "Polyvinylidene Fluoride", "category": "Fluoropolymer", "chemical_formula": "(C2H2F2)n"},
                "role": "Binder",
                "amount": {"name": "Weight Fraction", "numeric_value": 2.0, "unit": "wt%"},
            }
        ]],
    )

    properties: List[Property] = Field(
        default_factory=list,
        description="Properties of the slurry.",
        examples=[[{"name": "Solid Content", "numeric_value": 38.0, "unit": "wt%"}]],
    )


# 3. --- Process and Evaluation ---

class ProcessStepType(str, Enum):
    MATERIAL_PREPARATION = "Material Preparation"
    PRE_MIXING = "Pre-mixing"
    MIXING = "Mixing"
    HOMOGENIZATION = "Homogenization"
    DEGASSING = "Degassing"
    COATING = "Coating"
    CASTING = "Casting"
    DRYING = "Drying"
    CALENDERING = "Calendering"
    ANNEALING = "Annealing"
    CELL_ASSEMBLY = "Cell Assembly"
    FORMATION_CYCLING = "Formation Cycling"
    AGING = "Aging"
    STORAGE = "Storage"
    TRANSPORTATION = "Transportation"
    OTHER = "Other"


class Parameter(Measurement):
    """Represents a process parameter."""
    pass


class ProcessStep(BaseModel):
    """Describes a process step."""
    model_config = ConfigDict(graph_id_fields=['step_type', 'name'])

    step_type: ProcessStepType = Field(
        description="Type of step.",
        examples=["Mixing", "Coating", "Drying"],
    )

    name: Optional[str] = Field(
        default=None,
        description="Step descriptive name.",
        examples=["Primary Nitrogen Drying", "High-speed Mixing"],
    )

    parameters: List[Parameter] = Field(
        default_factory=list,
        description="Step parameters, e.g., temperature, speed.",
        examples=[[{"name": "Temperature", "numeric_value": 80.0, "unit": "°C"}]],
    )

    @field_validator('step_type', mode='before')
    @classmethod
    def _step_type_norm(cls, v):
        return _normalize_enum(ProcessStepType, v)


class MetricType(str, Enum):
    PEEL_STRENGTH = "Peel Strength"
    ADHESION_STRENGTH = "Adhesion Strength"
    AGGREGATION = "Aggregate Formation"
    GELATION = "Gelation"
    SEDIMENTATION_RATE = "Sedimentation Rate"
    PHASE_SEPARATION = "Phase Separation"
    VISCOSITY = "Viscosity"
    THIXOTROPY = "Thixotropy"
    YIELD_STRESS = "Yield Stress"
    SHEAR_THINNING = "Shear Thinning"
    PH = "pH"
    PH_STABILITY = "pH Stability"
    ZETA_POTENTIAL = "Zeta Potential"
    IONIC_CONDUCTIVITY = "Ionic Conductivity"
    SOLID_CONTENT = "Solid Content"
    PARTICLE_SIZE_DISTRIBUTION = "Particle Size Distribution"
    SURFACE_TENSION = "Surface Tension"
    TEMPERATURE_STABILITY = "Temperature Stability"
    STORAGE_STABILITY = "Storage Stability"
    SHELF_LIFE = "Shelf Life"
    DRYING_TIME = "Drying Time"
    FILM_UNIFORMITY = "Film Uniformity"
    COATING_QUALITY = "Coating Quality"
    WETTABILITY = "Wettability"
    MOISTURE_ABSORPTION = "Moisture Absorption"
    FOAMING_TENDENCY = "Foaming Tendency"
    OTHER = "Other"


class EvaluationMetric(Measurement):
    """Represents an evaluation metric value."""
    pass


class EvaluationResult(BaseModel):
    """Captures experimental outcome or metric."""
    model_config = ConfigDict(graph_id_fields=['metric_type', 'metric'])

    metric_type: MetricType = Field(
        description="Type of performance metric.",
        examples=["Viscosity", "Peel Strength"],
    )

    metric: Optional[EvaluationMetric] = Field(
        default=None,
        description="Name and value of the metric.",
        examples=[[{"name": "Viscosity", "numeric_value": 1.6, "unit": "mPa.s"}]],
    )

    method: Optional[str] = Field(
        default=None,
        description="Measurement method or standard.",
        examples=["JIS K6854-1", "Visual Inspection"],
    )

    comparison_baseline: Optional[str] = Field(
        default=None,
        description="What is compared against.",
        examples=["Previous formulation", "Industry average"],
    )

    trend: Optional[str] = Field(
        default=None,
        description="Tendency shown by the metric.",
        examples=["Increasing", "Stable", "Decreasing"],
    )

    @field_validator('metric_type', mode='before')
    @classmethod
    def _metric_type_norm(cls, v):
        return _normalize_enum(MetricType, v)

    @field_validator('metric', mode='before')
    @classmethod
    def _metric_coerce(cls, v, info):
        # Accept strings like "1.6 mPa.s" and coerce to an object.
        if v is None or isinstance(v, dict):
            return v
        if isinstance(v, str):
            # try to name the metric according to metric_type if available
            mt = info.data.get('metric_type')
            if isinstance(mt, Enum):
                name = mt.value
            else:
                # could be raw string; don't over-normalize here, just pass through
                name = str(mt) if mt is not None else None
            return _parse_measurement_string(v, default_name=name or "Value")
        return v


# 4. --- Main Ontology Entry Point ---

class Extraction(BaseModel):
    """Main experiment instance for a battery slurry."""
    model_config = ConfigDict(graph_id_fields=['slurry_under_test'])

    objective: Optional[str] = Field(
        default=None,
        description="Goal of the experiment.",
        examples=["Improve viscosity for better coating quality", "Reduce binder amount for cost optimization"],
    )

    hypothesis: Optional[str] = Field(
        default=None,
        description="Hypothesis explored or tested.",
        examples=["Adjusting binder ratio will lower viscosity", "Adding dispersant increases stability"],
    )

    slurry_under_test: Slurry = Edge(
        label="HAS_SLURRY",
        description="Slurry formulation tested.",
        examples=[{
            "slurry_id": "SLURRY-001",
            "components": [
                {
                    "material": {"name": "LiFePO4", "chemical_formula": "LiFePO4", "category": "Olivine Phosphate"},
                    "role": "Active Material",
                    "amount": {"name": "Weight Fraction", "numeric_value": 91.0, "unit": "wt%"},
                },
                {
                    "material": {"name": "Carbon Black", "category": "Additive", "chemical_formula": "C"},
                    "role": "Conductive Additive",
                    "amount": {"name": "Weight Fraction", "numeric_value": 6.0, "unit": "wt%"},
                },
                {
                    "material": {"name": "Polyvinylidene Fluoride", "category": "Fluoropolymer", "chemical_formula": "(C2H2F2)n"},
                    "role": "Binder",
                    "amount": {"name": "Weight Fraction", "numeric_value": 3.0, "unit": "wt%"},
                }
            ],
            "properties": [{"name": "Solid Content", "numeric_value": 50.0, "unit": "wt%"}]
        }],
    )

    fabrication_process: List[ProcessStep] = Edge(
        label="HAS_PROCESS_STEP",
        description="List of manufacturing process steps.",
        examples=[[
            {"step_type": "Mixing", "name": "High-shear Mixing", "parameters": [
                {"name": "Speed", "numeric_value": 2000, "unit": "rpm"}
            ]},
            {"step_type": "Coating", "name": "Slot-die Coating"},
            {"step_type": "Drying", "name": "Convection Drying", "parameters": [
                {"name": "Temperature", "numeric_value": 80.0, "unit": "°C"}
            ]}
        ]],
    )

    evaluation_results: List[EvaluationResult] = Edge(
        label="HAS_EVALUATION",
        description="Experiment evaluation results.",
        examples=[[
            {"metric_type": "Viscosity", "metric": {"name": "Viscosity", "numeric_value": 1.6, "unit": "mPa.s"}, "method": "Visual Inspection", "trend": "Increasing"},
            {"metric_type": "pH", "metric": {"name": "pH", "numeric_value": 12.3}, "method": "pH Meter", "trend": "Stable"},
            {"metric_type": "Solid Content", "metric": {"name": "Solid Content", "numeric_value": 50.0, "unit": "wt%"}, "trend": "Stable"}
        ]],
    )

    conclusion: Optional[str] = Field(
        default=None,
        description="Experiment conclusion.",
        examples=["Binder reduction improved viscosity without harming stability"],
    )

    key_findings: List[str] = Field(
        default_factory=list,
        description="Important findings and claims.",
        examples=["Stable dispersion achieved", "Optimized drying time"],
    )

    limitations: Optional[List[str]] = Field(
        default_factory=list,
        description="Stated limitations of the experiment.",
        examples=["Limited range of binder ratios tested"]
    )

    @field_validator('limitations', mode='before')
    def coerce_limitations(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(item) for item in v]
        raise TypeError("limitations should be a list[str] or string")


class Research(BaseModel):
    """Root model for source document of battery slurry experiments."""
    model_config = ConfigDict(graph_id_fields=['title'])

    title: str = Field(
        description="Title of the scientific document.",
        examples=["Preparation and Characterization of Novel Battery Slurries", "Large-Scale Manufacturing of Lithium-Ion Cathodes"],
    )

    experiments: List[Extraction] = Edge(
        label="HAS_EXPERIMENT",
        description="List of experiments included in the document.",
        examples=[[{"experiment_id": "EXP2024-001"}, {"experiment_id": "BATTERY-SLURRY-001"}]],
    )
