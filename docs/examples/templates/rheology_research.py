"""
Pydantic templates for Granular Rheology Research Ontology.

These models define a comprehensive ontology for granular materials rheology research,
focusing on vibrated systems, experimental setups, and rheological measurements.

This version incorporates:
- Experimental apparatus and simulation setup details
- Vibration parameters (amplitude, frequency, pressure)
- Rheological measurements (viscosity, friction, granular temperature)
- Material properties (particle size, mass, geometry)
- Numerical simulation parameters (DEM, contact models)
- Research metadata (authors, DOI, institution, references)
"""

import re
from enum import Enum
from typing import Any, List, Optional, Self, Type, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


# --- Edge Helper Function ---
def edge(label: str, **kwargs: Any) -> Any:
    # Extract default/default_factory if present to avoid conflict with Field(...)
    default = kwargs.pop("default", ...)
    default_factory = kwargs.pop("default_factory", None)
    
    if default_factory is not None:
        return Field(default_factory=default_factory, json_schema_extra={"edge_label": label}, **kwargs)
    else:
        return Field(default, json_schema_extra={"edge_label": label}, **kwargs)


# --- Helpers: normalization and parsing ---
def _normalize_enum(enum_cls: Type[Enum], v: Any) -> Any:
    """
    Accept:
    - enum instance
    - value strings (e.g., "Viscosity")
    - member-like strings (e.g., "VISCOSITY", "EFFECTIVE_VISCOSITY")
    - looser strings with spaces/underscores/case (e.g., "effective viscosity", "Effective_Viscosity")
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
    "1.6 mPa.s", "2 mm", "0.27 g", "45 mm", "2600", "80-90 °C" (ranges)
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
        description="The name of the property, e.g., 'Viscosity', 'Frequency', 'Amplitude', 'Pressure'.",
        examples=["Effective Viscosity", "Vibration Frequency", "Particle Radius", "Confining Pressure"],
    )

    text_value: Optional[str] = Field(
        default=None,
        description="The textual value of the property, if not numerical.",
        examples=["High", "Low", "Non-monotonic", "Weakly fluidized"],
    )

    numeric_value: Optional[Union[float, int]] = Field(
        default=None,
        description="The numerical value of the property (float or int).",
        examples=[1.6, 2.0, 2600, 0.27, 45],
    )

    numeric_value_min: Optional[Union[float, int]] = Field(
        default=None,
        description="Minimum value for range measurements.",
        examples=[6, 31, 80],
    )

    numeric_value_max: Optional[Union[float, int]] = Field(
        default=None,
        description="Maximum value for range measurements.",
        examples=[58, 924, 90],
    )

    unit: Optional[str] = Field(
        default=None,
        description="The unit of measurement, e.g., 'mPa.s', 'Hz', 'µm', 'Pa', 'mm', 'g'.",
        examples=["mPa.s", "Hz", "µm", "Pa", "mm", "g", "Nm", "rpm"],
    )

    condition: Optional[str] = Field(
        default=None,
        description="Measurement condition, e.g., 'at frequency f', 'under confining pressure p'.",
        examples=["at intermediate frequency", "under vertical vibration", "in steady state"],
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


# 2. --- Granular Material and System Geometry ---
class GranularMaterial(BaseModel):
    """Describes the granular material used in experiments."""

    model_config = ConfigDict(graph_id_fields=["material_type"])

    material_type: str = Field(
        description="Type of granular material.",
        examples=["Monodispersed spherical particles", "Glass beads", "Sand", "Polymer particles"],
    )

    particle_count: Optional[int] = Field(
        default=None,
        description="Number of particles in the system.",
        examples=[2600, 5000, 10000],
    )

    properties: List[MaterialProperty] = Field(
        default_factory=list,
        description="Material properties such as particle radius, mass, density.",
        examples=[
            [
                {"name": "Particle Radius", "numeric_value": 2.0, "unit": "mm"},
                {"name": "Particle Mass", "numeric_value": 0.27, "unit": "g"},
            ]
        ],
    )


class GeometryType(str, Enum):
    """Type of experimental geometry."""

    VANE_RHEOMETER = "Vane Rheometer"
    DOUBLE_PLATE = "Double Plate"
    SHEAR_CELL = "Shear Cell"
    INCLINED_PLANE = "Inclined Plane"
    CYLINDRICAL_CONTAINER = "Cylindrical Container"
    OTHER = "Other"


class SystemGeometry(BaseModel):
    """Describes the geometric configuration of the experimental setup."""

    model_config = ConfigDict(graph_id_fields=["geometry_type"])

    geometry_type: GeometryType = Field(
        description="Type of experimental geometry.",
        examples=["Vane Rheometer", "Double Plate", "Cylindrical Container"],
    )

    description: Optional[str] = Field(
        default=None,
        description="Detailed description of the geometry.",
        examples=[
            "Cylinder with conical-shaped base",
            "Rectangular-shaped vane rotating around vertical axis",
        ],
    )

    dimensions: List[Measurement] = Field(
        default_factory=list,
        description="Geometric dimensions.",
        examples=[
            [
                {"name": "Lid Radius", "numeric_value": 45, "unit": "mm"},
                {"name": "Container Height", "numeric_value": 100, "unit": "mm"},
            ]
        ],
    )

    @field_validator("geometry_type", mode="before")
    @classmethod
    def _geometry_type_norm(cls, v: Any) -> Any:
        return _normalize_enum(GeometryType, v)


# 3. --- Vibration Parameters ---
class VibrationParameter(Measurement):
    """Represents a vibration parameter."""

    model_config = ConfigDict(graph_id_fields=["name", "numeric_value", "unit"])


class VibrationConditions(BaseModel):
    """Describes the vibration conditions applied to the system."""

    model_config = ConfigDict(graph_id_fields=["vibration_type"])

    vibration_type: str = Field(
        description="Type of vibration applied.",
        examples=["Vertical sinusoidal", "Horizontal", "Random", "Pulsed"],
    )

    amplitude: Optional[VibrationParameter] = Field(
        default=None,
        description="Vibration amplitude.",
        examples=[{"name": "Amplitude", "numeric_value": 26, "unit": "µm"}],
    )

    frequency: Optional[VibrationParameter] = Field(
        default=None,
        description="Vibration frequency.",
        examples=[{"name": "Frequency", "numeric_value": 50, "unit": "Hz"}],
    )

    rescaled_acceleration: Optional[VibrationParameter] = Field(
        default=None,
        description="Rescaled acceleration Γ = (2πf)²A/g.",
        examples=[{"name": "Gamma", "numeric_value": 1.5, "unit": "dimensionless"}],
    )

    confining_pressure: Optional[VibrationParameter] = Field(
        default=None,
        description="Confining pressure applied to the system.",
        examples=[{"name": "Pressure", "numeric_value": 308, "unit": "Pa"}],
    )

    @field_validator("amplitude", "frequency", "rescaled_acceleration", "confining_pressure", mode="before")
    @classmethod
    def _param_coerce(cls, v: Any) -> Any:
        if isinstance(v, dict) or v is None:
            return v
        if isinstance(v, str):
            return _parse_measurement_string(v)
        return v


# 4. --- Simulation and Experimental Methods ---
class SimulationMethod(str, Enum):
    """Type of simulation method."""

    DEM = "Discrete Element Method"
    MOLECULAR_DYNAMICS = "Molecular Dynamics"
    FINITE_ELEMENT = "Finite Element Method"
    LATTICE_BOLTZMANN = "Lattice Boltzmann"
    OTHER = "Other"


class ContactModel(str, Enum):
    """Type of contact model for particle interactions."""

    HERTZ_MINDLIN = "Hertz-Mindlin"
    HOOKEAN = "Hookean"
    LINEAR_SPRING_DASHPOT = "Linear Spring-Dashpot"
    NONLINEAR_ELASTIC = "Nonlinear Elastic"
    OTHER = "Other"


class SimulationParameter(Measurement):
    """Represents a simulation parameter."""


class SimulationSetup(BaseModel):
    """Describes the numerical simulation setup."""

    model_config = ConfigDict(graph_id_fields=["method"])

    method: SimulationMethod = Field(
        description="Simulation method used.",
        examples=["Discrete Element Method", "Molecular Dynamics"],
    )

    software: Optional[str] = Field(
        default=None,
        description="Software package used for simulations.",
        examples=["LAMMPS", "LIGGGHTS", "EDEM", "YADE"],
    )

    contact_model: Optional[ContactModel] = Field(
        default=None,
        description="Contact model for particle interactions.",
        examples=["Hertz-Mindlin", "Hookean"],
    )

    parameters: List[SimulationParameter] = Field(
        default_factory=list,
        description="Simulation parameters such as stiffness, damping, friction coefficient.",
        examples=[
            [
                {"name": "Normal Stiffness", "numeric_value": 6.1e7, "unit": "Pa"},
                {"name": "Tangential Friction", "numeric_value": 0.5, "unit": "dimensionless"},
            ]
        ],
    )

    time_parameters: List[SimulationParameter] = Field(
        default_factory=list,
        description="Time-related parameters such as transient time, observation time.",
        examples=[
            [
                {"name": "Initial Transient", "numeric_value": 13, "unit": "s"},
                {"name": "Observation Time", "numeric_value": 33, "unit": "s"},
            ]
        ],
    )

    @field_validator("method", mode="before")
    @classmethod
    def _method_norm(cls, v: Any) -> Any:
        return _normalize_enum(SimulationMethod, v)

    @field_validator("contact_model", mode="before")
    @classmethod
    def _contact_model_norm(cls, v: Any) -> Any:
        if v is None:
            return v
        return _normalize_enum(ContactModel, v)


# 5. --- Rheological Measurements ---
class RheologicalMetricType(str, Enum):
    """Type of rheological measurement."""

    EFFECTIVE_VISCOSITY = "Effective Viscosity"
    FRICTION_COEFFICIENT = "Friction Coefficient"
    GRANULAR_TEMPERATURE = "Granular Temperature"
    SHEAR_STRESS = "Shear Stress"
    SHEAR_RATE = "Shear Rate"
    YIELD_STRESS = "Yield Stress"
    ANGULAR_VELOCITY = "Angular Velocity"
    KINETIC_ENERGY = "Kinetic Energy"
    DISSIPATION_RATE = "Dissipation Rate"
    FLOW_CURVE = "Flow Curve"
    OTHER = "Other"


class RheologicalMeasurement(BaseModel):
    """Represents a rheological measurement value."""

    model_config = ConfigDict(graph_id_fields=["name", "text_value", "numeric_value", "unit"])

    name: Optional[str] = Field(
        default=None,
        description="The name of the measurement, e.g., 'Effective Viscosity', 'Friction Coefficient'.",
        examples=["Effective Viscosity", "Granular Temperature", "Shear Stress"],
    )

    text_value: Optional[str] = Field(
        default=None,
        description="The textual value of the measurement, if not numerical.",
        examples=["High", "Low", "Non-monotonic", "Weakly fluidized"],
    )

    numeric_value: Optional[Union[float, int]] = Field(
        default=None,
        description="The numerical value of the measurement (float or int).",
        examples=[1.6, 2.0, 0.5],
    )

    numeric_value_min: Optional[Union[float, int]] = Field(
        default=None,
        description="Minimum value for range measurements.",
        examples=[6, 31, 80],
    )

    numeric_value_max: Optional[Union[float, int]] = Field(
        default=None,
        description="Maximum value for range measurements.",
        examples=[58, 924, 90],
    )

    unit: Optional[str] = Field(
        default=None,
        description="The unit of measurement, e.g., 'mPa.s', 'Hz', 'µm', 'Pa'.",
        examples=["mPa.s", "Hz", "µm", "Pa", "dimensionless"],
    )

    condition: Optional[str] = Field(
        default=None,
        description="Measurement condition.",
        examples=["at intermediate frequency", "under vibration"],
    )


class RheologicalResult(BaseModel):
    """Captures rheological measurement results."""

    model_config = ConfigDict(graph_id_fields=["metric_type", "measurement"])

    metric_type: RheologicalMetricType = Field(
        description="Type of rheological metric.",
        examples=["Effective Viscosity", "Friction Coefficient", "Granular Temperature"],
    )

    measurement: Optional[RheologicalMeasurement] = Field(
        default=None,
        description="The measured value.",
        examples=[{"name": "Effective Viscosity", "numeric_value": 1.6, "unit": "mPa.s"}],
    )

    behavior: Optional[str] = Field(
        default=None,
        description="Observed behavior or trend.",
        examples=[
            "Non-monotonic dependence on frequency",
            "Decreases with amplitude",
            "Increases with pressure",
        ],
    )

    scaling_relation: Optional[str] = Field(
        default=None,
        description="Mathematical scaling relation observed.",
        examples=["η ∝ K^(-2)", "µ_min controlled by A²/p", "K = K(A,f)"],
    )

    method: Optional[str] = Field(
        default=None,
        description="Measurement or analysis method.",
        examples=["Vane rheometry", "Microrheology approach", "DEM simulation"],
    )

    @field_validator("metric_type", mode="before")
    @classmethod
    def _metric_type_norm(cls, v: Any) -> Any:
        return _normalize_enum(RheologicalMetricType, v)

    @field_validator("measurement", mode="before")
    @classmethod
    def _measurement_coerce(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None or isinstance(v, dict):
            return v
        if isinstance(v, str):
            mt = info.data.get("metric_type")
            if isinstance(mt, Enum):
                name = mt.value
            else:
                name = str(mt) if mt is not None else None
            return _parse_measurement_string(v, default_name=name or "Value")
        return v


# 6. --- Experiment and Research ---
class Experiment(BaseModel):
    """Main experiment instance for granular rheology research."""

    model_config = ConfigDict(graph_id_fields=["experiment_id"])

    experiment_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this experiment.",
        examples=["EXP-RHEOLOGY-001", "VIBRATED-GRANULAR-2024"],
    )

    objective: Optional[str] = Field(
        default=None,
        description="Goal of the experiment.",
        examples=[
            "Study rheology of dense vibrated granular flows",
            "Investigate non-monotonic viscosity response",
            "Characterize friction weakening under vibration",
        ],
    )

    hypothesis: Optional[str] = Field(
        default=None,
        description="Hypothesis explored or tested.",
        examples=[
            "Rheological response governed by granular temperature",
            "Viscosity controlled by energy ratio K/Kp",
            "Friction weakening requires threshold in A²/p",
        ],
    )

    granular_material: Optional[GranularMaterial] = edge(
        label="USES_MATERIAL",
        description="Granular material used in the experiment.",
        default=None,
        examples=[
            {
                "material_type": "Monodispersed spherical particles",
                "particle_count": 2600,
                "properties": [
                    {"name": "Particle Radius", "numeric_value": 2.0, "unit": "mm"},
                    {"name": "Particle Mass", "numeric_value": 0.27, "unit": "g"},
                ],
            }
        ],
    )

    system_geometry: Optional[SystemGeometry] = edge(
        label="HAS_GEOMETRY",
        description="Geometric configuration of the experimental setup.",
        default=None,
        examples=[
            {
                "geometry_type": "Vane Rheometer",
                "description": "Cylinder with conical base and rotating vane",
                "dimensions": [{"name": "Lid Radius", "numeric_value": 45, "unit": "mm"}],
            }
        ],
    )

    vibration_conditions: Optional[VibrationConditions] = edge(
        label="HAS_VIBRATION",
        description="Vibration conditions applied.",
        default=None,
        examples=[
            {
                "vibration_type": "Vertical sinusoidal",
                "amplitude": {"name": "Amplitude", "numeric_value": 26, "unit": "µm"},
                "frequency": {"name": "Frequency", "numeric_value": 50, "unit": "Hz"},
            }
        ],
    )

    simulation_setup: Optional[SimulationSetup] = edge(
        label="HAS_SIMULATION",
        description="Numerical simulation setup details.",
        default=None,
        examples=[
            {
                "method": "Discrete Element Method",
                "software": "LAMMPS",
                "contact_model": "Hertz-Mindlin",
            }
        ],
    )

    rheological_results: List[RheologicalResult] = edge(
        label="HAS_RESULT",
        description="Rheological measurement results.",
        default_factory=list,
        examples=[
            [
                {
                    "metric_type": "Effective Viscosity",
                    "measurement": {"name": "Effective Viscosity", "numeric_value": 1.6, "unit": "mPa.s"},
                    "behavior": "Non-monotonic dependence on frequency",
                },
                {
                    "metric_type": "Granular Temperature",
                    "behavior": "Maximum at intermediate frequency",
                },
            ]
        ],
    )

    key_findings: List[str] = Field(
        default_factory=list,
        description="Important findings and conclusions.",
        examples=[
            [
                "Viscosity exhibits non-monotonic dependence on frequency",
                "Rheological response controlled by energy ratio K/Kp",
                "Friction weakening lost at high frequencies",
            ]
        ],
    )

    physical_interpretation: Optional[str] = Field(
        default=None,
        description="Physical interpretation of the results.",
        examples=[
            "Competition between grain-scale agitation energy and confining pressure",
            "Energy transfer efficiency governs rheological properties",
        ],
    )


class Research(BaseModel):
    """Root model for granular rheology research document."""

    model_config = ConfigDict(graph_id_fields=["title"])

    title: str = Field(
        description="Title of the scientific document.",
        examples=[
            "Rheology of dense vibrated granular flows",
            "Non-monotonic response controlled by granular temperature",
        ],
    )

    authors: List[str] = Field(
        default_factory=list,
        description="Document authors.",
        examples=[
            [
                "A. Plati",
                "G. Petrillo",
                "L. de Arcangelis",
                "A. Gnoli",
                "A. Puglisi",
                "A. Sarracino",
                "E. Lippiello",
            ]
        ],
    )

    publication_date: Optional[str] = Field(
        default=None,
        description="Publication or submission date.",
        examples=["2025-11-20", "November 20, 2025"],
    )

    doi: Optional[str] = Field(
        default=None,
        description="Digital Object Identifier.",
        examples=["10.1103/PhysRevE.XXX.XXXXXX"],
    )

    institutions: List[str] = Field(
        default_factory=list,
        description="Research institutions involved.",
        examples=[
            [
                "Université Paris-Saclay, CNRS",
                "University of Campania 'Luigi Vanvitelli'",
                "University of Rome 'La Sapienza'",
            ]
        ],
    )

    abstract: Optional[str] = Field(
        default=None,
        description="Abstract or summary of the research.",
        examples=[
            "We study the rheology of dense granular materials subjected to vertical vibration..."
        ],
    )

    experiments: List[Experiment] = edge(
        label="HAS_EXPERIMENT",
        description="List of experiments included in the document.",
        examples=[
            [
                {
                    "experiment_id": "RHEOLOGY-VIBRATED-001",
                    "objective": "Study non-monotonic viscosity response",
                }
            ]
        ],
    )

    references: List[str] = Field(
        default_factory=list,
        description="Cited references.",
        examples=[
            [
                "B. Andreotti, Y. Forterre, and O. Pouliquen, Granular media (2013)",
                "A. Gnoli et al., Phys. Rev. Lett 120, 138001 (2018)",
            ]
        ],
    )

# Made with Bob
