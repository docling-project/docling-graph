"""
Pydantic templates for Battery Slurry Ontology.

These models define a comprehensive ontology for battery slurry formulations, processing, and evaluation.

This version incorporates improvements for:
- Multiple slurries per experiment (anode/cathode)
- Range support for measurements
- Enhanced material tracking (supplier, grade, batch)
- Process sequencing and environmental conditions
- Temporal stability tracking
- Research metadata (authors, DOI, institution)
"""

import re
from enum import Enum
from typing import Any, List, Optional, Self, Type, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


# --- Edge Helper Function ---
def edge(label: str, **kwargs: Any) -> Any:
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)


# --- Helpers: normalization and parsing ---
def _normalize_enum(enum_cls: Type[Enum], v: Any) -> Any:
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
        key = re.sub(r"[^A-Za-z0-9]+", "", v).lower()
        mapping = {}
        for m in enum_cls:
            # map by member name and value
            mapping[re.sub(r"[^A-Za-z0-9]+", "", m.name).lower()] = m
            mapping[re.sub(r"[^A-Za-z0-9]+", "", m.value).lower()] = m
        if key in mapping:
            return mapping[key]
        # last-ditch attempt
        try:
            return enum_cls(v)
        except Exception:
            # Prefer safe fallback to OTHER if present
            if "OTHER" in getattr(enum_cls, "__members__", {}):
                return enum_cls.__members__["OTHER"]
            raise
    raise ValueError(f"Cannot normalize {v} to {enum_cls}")


def _parse_measurement_string(
    s: str, default_name: Optional[str] = None, strict: bool = False
) -> Any:
    """
    Parse strings like:
    "1.6 mPa.s", "38 wt%", "25 °C", "12 wt %", "80C", "80-90 °C" (ranges)
    into a Measurement-like dict: {name, numeric_value, numeric_value_min, numeric_value_max, text_value, unit}.
    If numeric parsing fails, preserve as text_value.

    Args:
        s: String to parse
        default_name: Default name if not extractable
        strict: If True, raise error on parse failure; if False, fall back to text_value
    """
    if not isinstance(s, str):
        return s

    # Try to parse range (e.g., "80-90 °C")
    range_match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*([^\d]+)?$", s)
    if range_match:
        min_val = float(range_match.group(1))
        max_val = float(range_match.group(2))
        unit = (range_match.group(3) or "").strip() or None
        return {
            "name": default_name or "Value",
            "numeric_value": None,
            "numeric_value_min": min_val,
            "numeric_value_max": max_val,
            "text_value": None,
            "unit": unit,
        }

    # Try to parse single value
    single_match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([^\d]+)?$", s)
    if single_match:
        num = float(single_match.group(1))
        unit = (single_match.group(2) or "").strip() or None
        return {
            "name": default_name or "Value",
            "numeric_value": num,
            "numeric_value_min": None,
            "numeric_value_max": None,
            "text_value": None,
            "unit": unit,
        }

    # No numeric part found
    if strict:
        raise ValueError(f"Cannot parse '{s}' as measurement")

    # Fallback: keep raw as text
    return {
        "name": default_name or "Value",
        "numeric_value": None,
        "numeric_value_min": None,
        "numeric_value_max": None,
        "text_value": s.strip(),
        "unit": None,
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

    numeric_value_min: Optional[Union[float, int]] = Field(
        default=None,
        description="Minimum value for range measurements.",
        examples=[80, 1.5],
    )

    numeric_value_max: Optional[Union[float, int]] = Field(
        default=None,
        description="Maximum value for range measurements.",
        examples=[90, 2.0],
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

    @model_validator(mode="after")
    def validate_value_consistency(self) -> Self:
        """Ensure value fields are used consistently."""
        has_single = self.numeric_value is not None
        has_min = self.numeric_value_min is not None
        has_max = self.numeric_value_max is not None

        # Allow: single value alone, explicit range (min+max), or implicit range (value+max or value+min)
        # Only reject if all three are set (ambiguous)
        if has_single and has_min and has_max:
            raise ValueError(
                "Cannot specify numeric_value, numeric_value_min, and numeric_value_max simultaneously"
            )

        # If using implicit range pattern (numeric_value with min or max), treat numeric_value as the other bound
        if has_single and (has_min or has_max):
            if has_max and not has_min:
                # Treat numeric_value as min
                self.numeric_value_min = self.numeric_value
                self.numeric_value = None
            elif has_min and not has_max:
                # Treat numeric_value as max
                self.numeric_value_max = self.numeric_value
                self.numeric_value = None

        return self


class MaterialProperty(Measurement):
    """Represents a material property, inherits from Measurement."""

    model_config = ConfigDict(graph_id_fields=["name", "text_value", "numeric_value", "unit"])


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

    model_config = ConfigDict(graph_id_fields=["name"])

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

    supplier: Optional[str] = Field(
        default=None,
        description="Material supplier/manufacturer.",
        examples=["Sigma-Aldrich", "MTI Corporation", "Targray", "BASF"],
    )

    grade: Optional[str] = Field(
        default=None,
        description="Material grade or purity level.",
        examples=["Battery grade", "99.9%", "Technical grade", "Reagent grade"],
    )

    batch_number: Optional[str] = Field(
        default=None,
        description="Batch or lot number for traceability.",
        examples=["BATCH-2024-001", "LOT-XY123"],
    )

    properties: List[MaterialProperty] = Field(
        default_factory=list,
        description="Properties, e.g., particle size, molecular weight.",
        examples=[[{"name": "Particle Size (D50)", "numeric_value": 3.2, "unit": "µm"}]],
    )


class ComponentAmount(Measurement):
    """Amount of a component, inherits Measurement."""


class Component(BaseModel):
    """Links a material to its role and amount."""

    model_config = ConfigDict(graph_id_fields=["material", "role", "amount"])

    material: Material = edge(
        label="USES_MATERIAL",
        description="Material used in this component.",
        examples=[
            {"name": "LiFePO4", "chemical_formula": "LiFePO4", "category": "Olivine Phosphate"}
        ],
    )

    role: MaterialRole = Field(
        description="Function of the material in the slurry.",
        examples=["Active Material", "Binder", "Conductive Additive"],
    )

    amount: Optional[ComponentAmount] = Field(
        description="Amount specification (weight/volume fraction).",
        examples=[{"name": "Weight Fraction", "numeric_value": 12.0, "unit": "wt%"}],
    )

    @field_validator("role", mode="before")
    @classmethod
    def _role_norm(cls, v: Any) -> Any:
        return _normalize_enum(MaterialRole, v)

    @field_validator("amount", mode="before")
    @classmethod
    def _amount_coerce(cls, v: Any) -> Any:
        # Accept strings like "12 wt%" and coerce into ComponentAmount
        if isinstance(v, dict) or v is None:
            return v
        if isinstance(v, str):
            return _parse_measurement_string(v, default_name="Weight Fraction")
        return v


class Property(Measurement):
    """Represents a slurry property."""

    model_config = ConfigDict(graph_id_fields=["name", "text_value", "numeric_value", "unit"])


class SlurryType(str, Enum):
    """Type of battery slurry."""

    CATHODE = "Cathode"
    ANODE = "Anode"
    ELECTROLYTE = "Electrolyte"
    SEPARATOR_COATING = "Separator Coating"
    OTHER = "Other"


class Slurry(BaseModel):
    """Collection of slurry components."""

    model_config = ConfigDict(graph_id_fields=["slurry_id"])

    slurry_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this specific slurry formulation.",
        examples=["CATH-001", "AN-CTRL-01", "SLURRY-2024-001"],
    )

    slurry_type: Optional[SlurryType] = Field(
        default=None,
        description="Whether this is a cathode, anode, or other slurry type.",
        examples=["Cathode", "Anode"],
    )

    description: Optional[str] = Field(
        default=None,
        description="Purpose or variant description.",
        examples=["Control formulation", "High solid content variant", "Low binder optimization"],
    )

    components: List[Component] = edge(
        label="HAS_COMPONENT",
        description="Slurry components list.",
        examples=[
            [
                {
                    "material": {
                        "name": "LiFePO4",
                        "chemical_formula": "LiFePO4",
                        "category": "Olivine Phosphate",
                    },
                    "role": "Active Material",
                    "amount": {"name": "Weight Fraction", "numeric_value": 12.0, "unit": "wt%"},
                },
                {
                    "material": {
                        "name": "Polyvinylidene Fluoride",
                        "category": "Fluoropolymer",
                        "chemical_formula": "(C2H2F2)n",
                    },
                    "role": "Binder",
                    "amount": {"name": "Weight Fraction", "numeric_value": 2.0, "unit": "wt%"},
                },
            ]
        ],
    )

    properties: List[Property] = Field(
        default_factory=list,
        description="Properties of the slurry.",
        examples=[[{"name": "Solid Content", "numeric_value": 38.0, "unit": "wt%"}]],
    )

    @field_validator("slurry_type", mode="before")
    @classmethod
    def _slurry_type_norm(cls, v: Any) -> Any:
        if v is None:
            return v
        return _normalize_enum(SlurryType, v)


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


class ProcessStep(BaseModel):
    """Describes a process step."""

    model_config = ConfigDict(graph_id_fields=["step_type", "name", "sequence_order"])

    step_type: ProcessStepType = Field(
        description="Type of step.",
        examples=["Mixing", "Coating", "Drying"],
    )

    name: Optional[str] = Field(
        default=None,
        description="Step descriptive name.",
        examples=["Primary Nitrogen Drying", "High-speed Mixing", "Slot-die Coating"],
    )

    sequence_order: Optional[int] = Field(
        default=None,
        description="Order in the process sequence (e.g., 1, 2, 3).",
        examples=[1, 2, 3],
    )

    duration: Optional[Measurement] = Field(
        default=None,
        description="Duration of this step.",
        examples=[{"name": "Duration", "numeric_value": 4, "unit": "hours"}],
    )

    equipment: Optional[str] = Field(
        default=None,
        description="Equipment or instrument used.",
        examples=["Planetary mixer", "Slot-die coater", "Twin-screw extruder", "Convection oven"],
    )

    atmosphere: Optional[str] = Field(
        default=None,
        description="Atmospheric conditions during processing.",
        examples=["Nitrogen", "Dry room (<1% RH)", "Air", "Argon"],
    )

    parameters: List[Parameter] = Field(
        default_factory=list,
        description="Step parameters, e.g., temperature, speed.",
        examples=[[{"name": "Temperature", "numeric_value": 80.0, "unit": "°C"}]],
    )

    environmental_conditions: List[Parameter] = Field(
        default_factory=list,
        description="Environmental parameters like humidity, pressure.",
        examples=[
            [
                {"name": "Relative Humidity", "numeric_value": 5, "unit": "%"},
                {"name": "Dew Point", "numeric_value": -40, "unit": "°C"},
            ]
        ],
    )

    @field_validator("step_type", mode="before")
    @classmethod
    def _step_type_norm(cls, v: Any) -> Any:
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

    model_config = ConfigDict(graph_id_fields=["name", "text_value", "numeric_value", "unit"])


class EvaluationResult(BaseModel):
    """Captures experimental outcome or metric."""

    model_config = ConfigDict(graph_id_fields=["metric_type", "metric", "time_point"])

    metric_type: MetricType = Field(
        description="Type of performance metric.",
        examples=["Viscosity", "Peel Strength"],
    )

    metric: Optional[EvaluationMetric] = Field(
        default=None,
        description="Name and value of the metric.",
        examples=[{"name": "Viscosity", "numeric_value": 1.6, "unit": "mPa.s"}],
    )

    time_point: Optional[Measurement] = Field(
        default=None,
        description="Time after slurry preparation when measured (for temporal stability tracking).",
        examples=[
            {"name": "Storage Time", "numeric_value": 24, "unit": "hours"},
            {"name": "Age", "numeric_value": 7, "unit": "days"},
        ],
    )

    method: Optional[str] = Field(
        default=None,
        description="Measurement method or standard.",
        examples=["JIS K6854-1", "Visual Inspection", "Rheometer", "ASTM D1084"],
    )

    comparison_baseline: Optional[str] = Field(
        default=None,
        description="What is compared against.",
        examples=["Previous formulation", "Industry average", "Control sample"],
    )

    trend: Optional[str] = Field(
        default=None,
        description="Tendency shown by the metric.",
        examples=["Increasing", "Stable", "Decreasing"],
    )

    @field_validator("metric_type", mode="before")
    @classmethod
    def _metric_type_norm(cls, v: Any) -> Any:
        return _normalize_enum(MetricType, v)

    @field_validator("metric", mode="before")
    @classmethod
    def _metric_coerce(cls, v: Any, info: ValidationInfo) -> Any:
        # Accept strings like "1.6 mPa.s" and coerce to an object.
        if v is None or isinstance(v, dict):
            return v
        if isinstance(v, str):
            # try to name the metric according to metric_type if available
            mt = info.data.get("metric_type")
            if isinstance(mt, Enum):
                name = mt.value
            else:
                # could be raw string; don't over-normalize here, just pass through
                name = str(mt) if mt is not None else None
            return _parse_measurement_string(v, default_name=name or "Value")
        return v


# 4. --- Main Ontology Entry Points ---
class Experiment(BaseModel):
    """Main experiment instance for battery slurry research."""

    model_config = ConfigDict(graph_id_fields=["experiment_id"])

    experiment_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this experiment.",
        examples=["EXP-2024-001", "BATTERY-SLURRY-001"],
    )

    objective: Optional[str] = Field(
        default=None,
        description="Goal of the experiment.",
        examples=[
            "Improve viscosity for better coating quality",
            "Reduce binder amount for cost optimization",
        ],
    )

    hypothesis: Optional[str] = Field(
        default=None,
        description="Hypothesis explored or tested.",
        examples=[
            "Adjusting binder ratio will lower viscosity",
            "Adding dispersant increases stability",
        ],
    )

    slurries: List[Slurry] = edge(
        label="HAS_SLURRY",
        description="All slurries tested in this experiment (anode, cathode, variants).",
        examples=[
            [
                {
                    "slurry_id": "CATH-001",
                    "slurry_type": "Cathode",
                    "description": "Control cathode formulation",
                    "components": [
                        {
                            "material": {"name": "LiFePO4", "chemical_formula": "LiFePO4"},
                            "role": "Active Material",
                            "amount": {
                                "name": "Weight Fraction",
                                "numeric_value": 91.0,
                                "unit": "wt%",
                            },
                        }
                    ],
                },
                {
                    "slurry_id": "AN-001",
                    "slurry_type": "Anode",
                    "description": "Graphite anode",
                    "components": [
                        {
                            "material": {"name": "Graphite", "chemical_formula": "C"},
                            "role": "Active Material",
                            "amount": {
                                "name": "Weight Fraction",
                                "numeric_value": 94.0,
                                "unit": "wt%",
                            },
                        }
                    ],
                },
            ]
        ],
    )

    control_slurry_id: Optional[str] = Field(
        default=None,
        description="Reference to control/baseline slurry for comparison.",
        examples=["CATH-CTRL-001", "AN-BASE"],
    )

    comparison_notes: Optional[str] = Field(
        default=None,
        description="Qualitative comparison observations.",
        examples=["20% viscosity reduction vs. control", "Improved stability compared to baseline"],
    )

    fabrication_process: List[ProcessStep] = edge(
        label="HAS_PROCESS_STEP",
        description="List of manufacturing process steps.",
        examples=[
            [
                {
                    "step_type": "Mixing",
                    "name": "High-shear Mixing",
                    "sequence_order": 1,
                    "parameters": [{"name": "Speed", "numeric_value": 2000, "unit": "rpm"}],
                },
                {"step_type": "Coating", "name": "Slot-die Coating", "sequence_order": 2},
                {
                    "step_type": "Drying",
                    "name": "Convection Drying",
                    "sequence_order": 3,
                    "parameters": [{"name": "Temperature", "numeric_value": 80.0, "unit": "°C"}],
                },
            ]
        ],
    )

    evaluation_results: List[EvaluationResult] = edge(
        label="HAS_EVALUATION",
        description="Experiment evaluation results.",
        examples=[
            [
                {
                    "metric_type": "Viscosity",
                    "metric": {"name": "Viscosity", "numeric_value": 1.6, "unit": "mPa.s"},
                    "method": "Rheometer",
                    "trend": "Stable",
                },
                {
                    "metric_type": "pH",
                    "metric": {"name": "pH", "numeric_value": 12.3},
                    "method": "pH Meter",
                    "trend": "Stable",
                },
            ]
        ],
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
        examples=[
            "Limited range of binder ratios tested",
            "Single temperature condition evaluated",
        ],
    )

    @field_validator("limitations", mode="before")
    def coerce_limitations(self, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(item) for item in v]
        raise TypeError("limitations should be a list[str] or string")


class Research(BaseModel):
    """Root model for source document of battery slurry experiments."""

    model_config = ConfigDict(graph_id_fields=["title"])

    title: str = Field(
        description="Title of the scientific document.",
        examples=[
            "Preparation and Characterization of Novel Battery Slurries",
            "Large-Scale Manufacturing of Lithium-Ion Cathodes",
        ],
    )

    authors: List[str] = Field(
        default_factory=list,
        description="Document authors.",
        examples=[["Smith, J.", "Doe, A.", "Chen, L."]],
    )

    publication_date: Optional[str] = Field(
        default=None,
        description="Publication or submission date.",
        examples=["2024-03-15", "March 2024"],
    )

    doi: Optional[str] = Field(
        default=None,
        description="Digital Object Identifier.",
        examples=["10.1002/example.12345"],
    )

    institution: Optional[str] = Field(
        default=None,
        description="Primary research institution.",
        examples=["MIT", "Stanford University", "Fraunhofer Institute"],
    )

    experiments: List[Experiment] = edge(
        label="HAS_EXPERIMENT",
        description="List of experiments included in the document.",
        examples=[[{"experiment_id": "EXP2024-001"}, {"experiment_id": "BATTERY-SLURRY-001"}]],
    )
