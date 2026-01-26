"""
Pydantic template for Slurry-Battery Rheology Research Papers.

Extracts structured data from battery electrode slurry rheology research papers,
focusing on steady shear and oscillatory rheology measurements. Captures critical
preparation history (mixing order, deagglomeration) that affects gel formation
and yield behavior.

Key entities:
- ScholarlyRheologyPaper: Research paper metadata
- SlurryRheologyStudy: Coherent experimental campaign
- SlurryRheologyExperiment: Unit of scientific comparison
- BatterySlurryBatch: Real produced slurry with preparation history
- RheologyTestRun: Execution of protocol on batch
- RheologyDataset: Machine-readable outputs (curves + metadata)

Key relationships:
- Paper --HAS_STUDY--> Study --HAS_EXPERIMENT--> Experiment
- Experiment --USES_BATCH--> Batch --HAS_FORMULATION--> Formulation
- Experiment --HAS_TEST_RUN--> TestRun --PRODUCES_DATASET--> Dataset
"""

import re
from enum import Enum
from typing import Any, List, Optional, Self, Type, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --- Edge Helper Function (REQUIRED) ---
def edge(label: str, **kwargs: Any) -> Any:
    """
    Helper function to create a Pydantic Field with edge metadata.
    The 'edge_label' defines the type of relationship in the knowledge graph.
    """
    # Extract default if provided, otherwise use ellipsis for required fields
    default = kwargs.pop("default", ...)
    return Field(default, json_schema_extra={"edge_label": label}, **kwargs)


# --- Helper Functions ---
def _normalize_enum(enum_cls: Type[Enum], v: Any) -> Any:
    """
    Normalize enum values to handle various input formats.
    Accepts enum instances, value strings, or member names with flexible formatting.
    Falls back to 'OTHER' member if present.
    """
    if isinstance(v, enum_cls):
        return v

    if isinstance(v, str):
        # Normalize to alphanumeric lowercase
        key = re.sub(r"[^A-Za-z0-9]+", "", v).lower()

        # Build mapping of normalized names/values to enum members
        mapping = {}
        for member in enum_cls:
            normalized_name = re.sub(r"[^A-Za-z0-9]+", "", member.name).lower()
            normalized_value = re.sub(r"[^A-Za-z0-9]+", "", member.value).lower()
            mapping[normalized_name] = member
            mapping[normalized_value] = member

        if key in mapping:
            return mapping[key]

        # Last attempt: direct value match
        try:
            return enum_cls(v)
        except Exception:
            # Safe fallback to OTHER if present
            if "OTHER" in enum_cls.__members__:
                return enum_cls.OTHER
            raise


def _to_quantity(v: Any, default_name: str = "Value") -> Any:
    """
    Convert various input formats to QuantityWithUnit.

    Handles:
    - Already a QuantityWithUnit instance → return as-is
    - Dict → pass through (Pydantic will validate)
    - Number (int/float) → create QuantityWithUnit with numeric_value
    - String with number → parse and create QuantityWithUnit
    - None → return None

    Examples:
        40 → {"name": "Value", "numeric_value": 40}
        "25 °C" → {"name": "Value", "numeric_value": 25, "unit": "°C"}
        {"numeric_value": 100, "unit": "mm"} → pass through
    """
    if v is None:
        return v

    # Already correct type
    if isinstance(v, dict):
        return v

    # Simple numeric value
    if isinstance(v, int | float):
        return {"name": default_name, "numeric_value": float(v), "unit": None}

    # String - try to parse
    if isinstance(v, str):
        # Try to extract number and unit
        match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(.*)$", v)
        if match:
            num_str, unit_str = match.groups()
            return {
                "name": default_name,
                "numeric_value": float(num_str),
                "unit": unit_str.strip() if unit_str.strip() else None,
            }
        # No number found, treat as text value
        return {"name": default_name, "text_value": v.strip()}

    # Unknown type, return as-is and let Pydantic handle validation
    return v


# ============================================================================
# SHARED PRIMITIVE COMPONENTS
# ============================================================================


class QuantityWithUnit(BaseModel):
    """
    Flexible measurement supporting single values, ranges, or text.
    Can represent '25°C', '1.6 mPa.s', '80-90°C', or 'High'.
    Deduplicated by content - identical measurements share the same node.
    """

    model_config = ConfigDict(is_entity=False)

    name: str | None = Field(
        None,
        description=(
            "Name of the measured property (optional - will be auto-generated if missing). "
            "Look for property names like 'Temperature', 'Viscosity', 'Shear rate', 'Time', 'RPM', 'Pressure'. "
            "Extract the full descriptive name as it appears in the document. "
            "If not explicitly stated, leave None and it will be auto-generated."
        ),
        examples=[
            "Temperature",
            "Shear rate",
            "Viscosity",
            "Solid loading",
            "Mixing time",
            "Particle size D50",
        ],
    )

    numeric_value: Union[float, int] | None = Field(
        None,
        description=(
            "Single numerical value for the measurement. "
            "Extract only the number, removing any units or text. "
            "Use this field for single-point measurements. "
            "Leave None if the value is a range or text."
        ),
        examples=[25.0, 1.6, 8.2, 2600, 0.27, 45],
    )

    numeric_value_min: Union[float, int] | None = Field(
        None,
        description=(
            "Minimum value for range measurements. "
            "Look for patterns like '80-90', '1.5 to 2.0', '20-30'. "
            "Extract the lower bound of the range."
        ),
        examples=[80.0, 1.5, 20, 6, 31],
    )

    numeric_value_max: Union[float, int] | None = Field(
        None,
        description=(
            "Maximum value for range measurements. "
            "Look for patterns like '80-90', '1.5 to 2.0', '20-30'. "
            "Extract the upper bound of the range."
        ),
        examples=[90.0, 2.0, 30, 58, 924],
    )

    text_value: str | None = Field(
        None,
        description=(
            "Textual value if not numerical. "
            "Use for qualitative descriptions like 'High', 'Low', 'Stable'. "
            "Extract as-is from the document."
        ),
        examples=["High", "Low", "Stable", "Non-monotonic", "Increasing"],
    )

    unit: str | None = Field(
        None,
        description=(
            "Unit of measurement. "
            "Look for units like '°C', 'mPa.s', 'Hz', 'µm', 'Pa', 'mm', 'g', 'wt%', 'vol%', 'rpm', 's', 'min'. "
            "Extract exactly as written, preserving symbols and formatting. "
            "Leave None for dimensionless quantities."
        ),
        examples=[
            "°C",
            "mPa.s",
            "Hz",
            "µm",
            "Pa",
            "mm",
            "g",
            "wt%",
            "vol%",
            "rpm",
            "s",
            "min",
            "kg/m³",
        ],
    )

    @model_validator(mode="after")
    def validate_and_set_defaults(self) -> Self:
        """
        Ensure value fields are used consistently and auto-generate name if missing.
        This validator is lenient - it coerces instead of rejecting to prevent extraction failures.
        """
        has_single = self.numeric_value is not None
        has_min = self.numeric_value_min is not None
        has_max = self.numeric_value_max is not None

        # Auto-generate name if missing
        if not self.name:
            if has_single or has_min or has_max:
                # Generate name based on value type
                if has_min and has_max:
                    self.name = "Value range"
                elif has_single:
                    self.name = "Value"
                else:
                    self.name = "Measurement"
            elif self.text_value:
                self.name = "Property"
            else:
                self.name = "Value"

        # Lenient handling of ambiguous cases - coerce instead of reject
        if has_single and has_min and has_max:
            # If all three are set, prefer the range and discard single value
            self.numeric_value = None

        # Allow implicit range: if numeric_value + min/max, treat as range
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


# ============================================================================
# LAYER 1: SCHOLARLY DOCUMENT
# ============================================================================


class PersonIdentity(BaseModel):
    """
    Person identity component.
    Deduplicated by content - identical person info shares the same node.
    """

    model_config = ConfigDict(is_entity=False)

    full_name: str = Field(
        description=(
            "Full name of the person. "
            "Look for author names in the header, author list, or affiliations. "
            "Extract as 'FirstName LastName' or as written. "
            "Include middle initials if present."
        ),
        examples=["John Smith", "Maria Garcia", "Jean-Pierre Dupont", "A. Kumar", "李明"],
    )

    orcid: str | None = Field(
        None,
        description=(
            "ORCID identifier if available. "
            "Look for 'ORCID:', 'ORCID iD:', or URLs like 'orcid.org/0000-0002-1234-5678'. "
            "Extract the numeric identifier (0000-0002-1234-5678 format)."
        ),
        examples=["0000-0002-1234-5678", "0000-0001-9876-5432"],
    )


class ScholarlyIdentifier(BaseModel):
    """
    Scholarly identifier component (DOI, arXiv, URL).
    Deduplicated by content - identical identifiers share the same node.
    """

    model_config = ConfigDict(is_entity=False)

    scheme: str = Field(
        description=(
            "Type of identifier. "
            "Look for 'DOI:', 'arXiv:', 'URL:', or similar prefixes. "
            "Common values: 'DOI', 'arXiv', 'URL', 'PMID', 'ISBN'."
        ),
        examples=["DOI", "arXiv", "URL", "PMID", "ISBN"],
    )

    value: str = Field(
        description=(
            "The identifier value. "
            "For DOI: extract the full DOI (e.g., '10.1103/PhysRevE.XXX.XXXXXX'). "
            "For arXiv: extract the arXiv ID (e.g., '2401.12345'). "
            "For URL: extract the complete URL."
        ),
        examples=[
            "10.1103/PhysRevE.XXX.XXXXXX",
            "2401.12345",
            "https://example.com/paper",
            "10.1016/j.jpowsour.2023.233456",
        ],
    )


class ScholarlyRheologyPaper(BaseModel):
    """
    Root model for slurry-battery rheology research paper.
    Uniquely identified by title.
    """

    model_config = ConfigDict(graph_id_fields=["title"])

    title: str = Field(
        description=(
            "Full title of the research paper. "
            "Look for the main title at the top of the document. "
            "Extract exactly as written, including subtitles if present."
        ),
        examples=[
            "Rheological Properties of LiFePO4 Cathode Slurries for Lithium-Ion Batteries",
            "Effect of Binder Molecular Weight on Slurry Rheology and Electrode Performance",
            "Yield Stress and Viscosity of Battery Electrode Slurries: Impact of Mixing Protocol",
        ],
    )

    authors: List[PersonIdentity] = Field(
        default_factory=list,
        description=(
            "List of paper authors. "
            "Look for the author list below the title or in the header. "
            "Extract each author's full name and ORCID if available. "
            "Maintain the order as listed in the paper."
        ),
        examples=[
            [
                {"full_name": "John Smith", "orcid": "0000-0002-1234-5678"},
                {"full_name": "Maria Garcia", "orcid": None},
            ]
        ],
    )

    publication_date: str | None = Field(
        None,
        description=(
            "Publication or submission date. "
            "Look for 'Published:', 'Received:', 'Submitted:', or date stamps. "
            "Extract in any format found (YYYY-MM-DD, Month DD YYYY, etc.)."
        ),
        examples=["2024-01-15", "January 15, 2024", "2024", "15 Jan 2024"],
    )

    identifiers: List[ScholarlyIdentifier] = Field(
        default_factory=list,
        description=(
            "Document identifiers (DOI, arXiv, URL). "
            "Look for 'DOI:', 'arXiv:', or URLs in the header or footer. "
            "Extract all available identifiers."
        ),
        examples=[
            [
                {"scheme": "DOI", "value": "10.1016/j.jpowsour.2023.233456"},
                {"scheme": "arXiv", "value": "2401.12345"},
            ]
        ],
    )

    abstract: str | None = Field(
        None,
        description=(
            "Paper abstract or summary. "
            "Look for 'Abstract' section at the beginning of the paper. "
            "Extract the full text of the abstract."
        ),
        examples=["We investigate the rheological properties of LiFePO4 cathode slurries..."],
    )

    institutions: List[str] = Field(
        default_factory=list,
        description=(
            "Research institutions or affiliations. "
            "Look for affiliations below author names or in footnotes. "
            "Extract full institution names."
        ),
        examples=[
            ["MIT", "Stanford University", "CNRS France"],
            ["University of Tokyo", "Max Planck Institute"],
        ],
    )

    keywords: List[str] = Field(
        default_factory=list,
        description=(
            "Paper keywords or key terms. "
            "Look for 'Keywords:', 'Key words:', or similar sections. "
            "Extract each keyword as a separate string."
        ),
        examples=[
            ["rheology", "battery slurry", "PVDF", "yield stress", "viscosity"],
            ["electrode processing", "mixing", "shear thinning"],
        ],
    )

    studies: List["SlurryRheologyStudy"] = edge(
        label="HAS_STUDY",
        default_factory=list,
        description=(
            "List of experimental studies in the paper. "
            "Each study represents a coherent experimental campaign. "
            "Extract all distinct studies described in the paper."
        ),
    )


# ============================================================================
# LAYER 2: STUDY AND EXPERIMENT
# ============================================================================


class SlurryRheologyStudy(BaseModel):
    """
    Coherent experimental campaign within a paper.
    Example: varying binder MW, carbon black addition, solids loading.
    Uniquely identified by study_id.
    """

    model_config = ConfigDict(graph_id_fields=["study_id"])

    study_id: str = Field(
        description=(
            "Unique identifier for this study. "
            "Create a descriptive ID based on the study objective. "
            "Format: 'STUDY-[brief-description]' or use section number."
        ),
        examples=[
            "STUDY-BINDER-MW-VARIATION",
            "STUDY-SOLID-LOADING-EFFECT",
            "STUDY-MIXING-PROTOCOL",
            "STUDY-SECTION-3.2",
        ],
    )

    objective: str = Field(
        description=(
            "Goal or purpose of this study. "
            "Look for statements like 'The objective is...', 'We investigate...', 'This study examines...'. "
            "Extract the main research question or goal."
        ),
        examples=[
            "Investigate effect of binder molecular weight on slurry rheology",
            "Study impact of solid loading on yield stress and viscosity",
            "Examine role of mixing protocol on gel formation",
        ],
    )

    experiments: List["SlurryRheologyExperiment"] = edge(
        label="HAS_EXPERIMENT",
        default_factory=list,
        description=(
            "List of experiments in this study. "
            "Each experiment is a unit of scientific comparison. "
            "Extract all experiments described in this study."
        ),
    )


class SlurryRheologyExperiment(BaseModel):
    """
    Unit of scientific comparison.
    Example: '20 vol% LFP + PVDF761 vs PVDFHSV'.
    Uniquely identified by experiment_id.
    """

    model_config = ConfigDict(graph_id_fields=["experiment_id"])

    experiment_id: str = Field(
        description=(
            "Unique identifier for this experiment. "
            "Create a descriptive ID based on the experimental conditions. "
            "Format: 'EXP-[brief-description]' or use figure/table reference."
        ),
        examples=[
            "EXP-LFP-PVDF761-20VOL",
            "EXP-NMC-HIGH-LOADING",
            "EXP-FIG-3A",
            "EXP-TABLE-2-SAMPLE-1",
        ],
    )

    description: str | None = Field(
        None,
        description=(
            "Brief description of the experiment. "
            "Summarize the key experimental conditions or variables. "
            "Extract from figure captions, table headers, or text descriptions."
        ),
        examples=[
            "LFP slurry with PVDF binder at 20 vol% solid loading",
            "NMC622 cathode slurry with varying carbon black content",
            "Effect of high-shear mixing on graphite anode slurry",
        ],
    )

    slurry_batch: Optional["BatterySlurryBatch"] = edge(
        label="USES_BATCH",
        default=None,
        description=(
            "The slurry batch used in this experiment. "
            "Links to the actual prepared slurry with its formulation and preparation history."
        ),
    )

    rheometry_runs: List["RheologyTestRun"] = edge(
        label="HAS_TEST_RUN",
        default_factory=list,
        description=(
            "List of rheology test runs performed on this slurry. "
            "Each run represents one execution of a test protocol. "
            "Extract all rheology measurements described for this experiment."
        ),
    )

    reported_findings: List[str] = Field(
        default_factory=list,
        description=(
            "Key findings and observations from this experiment. "
            "Look for statements like 'We observed...', 'Results show...', 'The data indicate...'. "
            "Extract each finding as a separate string."
        ),
        examples=[
            [
                "Viscosity decreases with increasing shear rate (shear thinning)",
                "Yield stress increases with solid loading",
                "Gel formation observed after 24h aging",
            ]
        ],
    )


# ============================================================================
# LAYER 3: FORMULATION
# ============================================================================


class ComponentRole(str, Enum):
    """Role of component in battery slurry formulation."""

    ACTIVE_MATERIAL = "Active Material"
    CONDUCTIVE_ADDITIVE = "Conductive Additive"
    POLYMER_BINDER = "Polymer Binder"
    SOLVENT = "Solvent"
    FUNCTIONAL_ADDITIVE = "Functional Additive"
    OTHER = "Other"


class SlurryComponent(BaseModel):
    """
    Component in battery slurry formulation.
    Combines role, material identity, and amount.
    Uniquely identified by role and material name.
    """

    model_config = ConfigDict(graph_id_fields=["component_role", "material_name"])

    component_role: ComponentRole = Field(
        description=(
            "Role of this component in the slurry. "
            "Look for 'active material', 'binder', 'conductive additive', 'solvent'. "
            "Common: Active Material (LFP, NMC), Conductive Additive (carbon black), Polymer Binder (PVDF, CMC), Solvent (NMP, water)."
        ),
        examples=["Active Material", "Conductive Additive", "Polymer Binder", "Solvent"],
    )

    material_name: str = Field(
        description=(
            "Material name or chemical formula (e.g., 'LiFePO4', 'PVDF', 'Carbon black Super C65'). "
            "Extract the full name or abbreviation as written."
        ),
        examples=[
            "LiFePO4",
            "NMC622",
            "PVDF",
            "NMP",
            "Carbon black Super C65",
            "Graphite SLC1506T",
        ],
    )

    material_supplier: str | None = Field(
        None,
        description="Material supplier (e.g., 'Sigma-Aldrich', 'MTI Corp', 'Timcal')",
        examples=["Sigma-Aldrich", "MTI Corp", "Timcal"],
    )

    material_grade: str | None = Field(
        None,
        description="Catalog number or grade (e.g., 'HSV900', 'C-NERGY C65', 'Grade 761')",
        examples=["HSV900", "C-NERGY C65", "Grade 761"],
    )

    amount_value: float | None = Field(
        None,
        description=(
            "Numeric amount of this component. "
            "Extract the number only, without units. "
            "Look for percentages, masses, or ratios."
        ),
        examples=[85.0, 10.0, 5.0, 2.5, 100.0],
    )

    amount_unit: str | None = Field(
        None,
        description=(
            "Unit or basis for the amount. "
            "Look for 'wt%', 'vol%', 'g', 'parts per 100 parts active material'. "
            "Common in battery literature: 'wt% relative to active material', 'wt% of total solids'. "
            "Extract exactly as written."
        ),
        examples=[
            "wt%",
            "vol%",
            "g",
            "wt% relative to active material",
            "parts per 100 parts active",
            "wt% of total solids",
        ],
    )

    particle_size: QuantityWithUnit | None = Field(
        None,
        description=(
            "Particle size information (for particulate components). "
            "Look for 'D50', 'D10', 'D90', 'particle size', 'diameter'. "
            "Extract value and unit (µm, nm, mm)."
        ),
        examples=[
            {"name": "D50", "numeric_value": 5.0, "unit": "µm"},
            {
                "name": "Particle diameter",
                "numeric_value_min": 2.0,
                "numeric_value_max": 10.0,
                "unit": "µm",
            },
        ],
    )

    molecular_weight: QuantityWithUnit | None = Field(
        None,
        description=(
            "Molecular weight (for polymers like binders). "
            "Look for 'MW', 'Mw', 'molecular weight', 'Mn'. "
            "Extract value and unit (g/mol, kDa, Da)."
        ),
        examples=[
            {"name": "Mw", "numeric_value": 534000, "unit": "g/mol"},
            {"name": "Molecular weight", "numeric_value": 534, "unit": "kDa"},
        ],
    )

    @field_validator("component_role", mode="before")
    @classmethod
    def normalize_role(cls, v: Any) -> Any:
        """Normalize component role enum."""
        return _normalize_enum(ComponentRole, v)


class SlurryFormulation(BaseModel):
    """
    Battery slurry formulation specification (the recipe).
    Uniquely identified by formulation_id.
    """

    model_config = ConfigDict(graph_id_fields=["formulation_id"])

    formulation_id: str = Field(
        description=(
            "Unique identifier for this formulation. "
            "Create a descriptive ID based on the composition. "
            "Format: 'FORM-[brief-description]' or use table reference."
        ),
        examples=[
            "FORM-LFP-PVDF-C65",
            "FORM-NMC-85-10-5",
            "FORM-TABLE-1-SAMPLE-A",
            "FORM-STANDARD-CATHODE",
        ],
    )

    description: str | None = Field(
        None,
        description=(
            "Brief description of the formulation. "
            "Summarize the main components and their ratios. "
            "Extract from text or create from component list."
        ),
        examples=[
            "LFP cathode: 85 wt% LFP, 10 wt% PVDF, 5 wt% C65",
            "Standard NMC622 formulation with CMC/SBR binder",
            "Graphite anode with 96:2:2 active:binder:carbon ratio",
        ],
    )

    components: List[SlurryComponent] = edge(
        label="HAS_COMPONENT",
        default_factory=list,
        description=(
            "List of components in this formulation. "
            "Extract all materials with their roles and amounts. "
            "Look for composition tables or text descriptions like '85 wt% LFP, 10 wt% PVDF, 5 wt% carbon black'."
        ),
    )

    target_solid_loading: QuantityWithUnit | None = Field(
        None,
        description=(
            "Target solid content of the slurry. "
            "Look for 'solid loading', 'solids content', 'solid fraction'. "
            "Extract numeric value and unit (wt%, vol%, g/L). "
            "Common range: 20-70 wt% for battery slurries."
        ),
        examples=[
            {"name": "Solid loading", "numeric_value": 45, "unit": "wt%"},
            {"name": "Solids content", "numeric_value": 60, "unit": "vol%"},
            {
                "name": "Solid loading",
                "numeric_value_min": 40,
                "numeric_value_max": 50,
                "unit": "wt%",
            },
        ],
    )


# ============================================================================
# LAYER 4: BATCH AND PREPARATION
# ============================================================================


class StepType(str, Enum):
    """Type of slurry preparation step."""

    DRY_PREMIX = "Dry Premix"
    BINDER_DISSOLUTION = "Binder Dissolution"
    CONDUCTIVE_ADDITIVE_DISPERSION = "Conductive Additive Dispersion"
    ACTIVE_MATERIAL_INCORPORATION = "Active Material Incorporation"
    HIGH_SHEAR_MIXING = "High Shear Mixing"
    VACUUM_MIXING = "Vacuum Mixing"
    DEAGGLOMERATION = "Deagglomeration"
    REST = "Rest"
    DEGASSING = "Degassing"
    OTHER = "Other"


class PreparationStep(BaseModel):
    """
    Single step in slurry preparation process.
    Order-of-addition and mixing conditions affect gel formation.
    Uniquely identified by step_id.
    """

    model_config = ConfigDict(graph_id_fields=["step_id"])

    step_id: str = Field(
        description=(
            "Unique identifier for this preparation step. "
            "Create an ID indicating the step number and type. "
            "Format: 'STEP-[number]-[brief-description]'."
        ),
        examples=[
            "STEP-1-BINDER-DISSOLUTION",
            "STEP-2-CARBON-DISPERSION",
            "STEP-3-ACTIVE-ADDITION",
            "STEP-4-HIGH-SHEAR-MIX",
        ],
    )

    step_type: StepType = Field(
        description=(
            "Type of preparation step. "
            "Look for descriptions of mixing stages like 'dissolve binder', 'disperse carbon black', 'add active material'. "
            "Common steps: Binder Dissolution, Conductive Additive Dispersion, Active Material Incorporation, "
            "High Shear Mixing, Vacuum Mixing, Deagglomeration, Rest, Degassing."
        ),
        examples=[
            "Binder Dissolution",
            "Conductive Additive Dispersion",
            "Active Material Incorporation",
            "High Shear Mixing",
            "Vacuum Mixing",
            "Degassing",
        ],
    )

    description: str | None = Field(
        None,
        description=(
            "Detailed description of this step. "
            "Extract the full procedure description from the methods section. "
            "Include any specific instructions or conditions."
        ),
        examples=[
            "PVDF binder dissolved in NMP at 60°C with stirring for 2 hours",
            "Carbon black dispersed in NMP using ultrasonic homogenizer for 30 min",
            "Active material added gradually while mixing at 2000 rpm",
        ],
    )

    equipment_name: str | None = Field(
        None,
        description=(
            "Equipment name or type used for this step. "
            "Look for 'planetary mixer', 'high-speed disperser', 'ultrasonic homogenizer'. "
            "Extract the equipment type or model name."
        ),
        examples=[
            "Planetary mixer",
            "Ultrasonic homogenizer",
            "Thinky mixer",
            "High-speed disperser",
        ],
    )

    equipment_model: str | None = Field(
        None,
        description="Equipment model number (e.g., 'ARE-310', 'T25 digital ULTRA-TURRAX')",
        examples=["ARE-310", "T25 digital ULTRA-TURRAX", "PM100"],
    )

    equipment_vendor: str | None = Field(
        None,
        description="Equipment manufacturer (e.g., 'Thinky', 'IKA', 'Retsch')",
        examples=["Thinky", "IKA", "Retsch"],
    )

    parameters: List[QuantityWithUnit] = Field(
        default_factory=list,
        description=(
            "Process parameters for this step. "
            "Look for RPM, time, temperature, vacuum level, power, etc. "
            "Extract each parameter with its value and unit. "
            "Common parameters: mixing speed (rpm), duration (min, h), temperature (°C), vacuum (mbar)."
        ),
        examples=[
            [
                {"name": "Mixing speed", "numeric_value": 2000, "unit": "rpm"},
                {"name": "Duration", "numeric_value": 30, "unit": "min"},
                {"name": "Temperature", "numeric_value": 60, "unit": "°C"},
            ]
        ],
    )

    @field_validator("step_type", mode="before")
    @classmethod
    def normalize_step_type(cls, v: Any) -> Any:
        """Normalize step type enum."""
        return _normalize_enum(StepType, v)


class BatterySlurryBatch(BaseModel):
    """
    Real produced slurry batch used for testing.
    Links formulation (recipe) with actual preparation process.
    Uniquely identified by batch_id.
    """

    model_config = ConfigDict(graph_id_fields=["batch_id"])

    batch_id: str = Field(
        description=(
            "Unique identifier for this slurry batch. "
            "Create a descriptive ID or use the batch code from the paper. "
            "Format: 'BATCH-[description]' or use sample name."
        ),
        examples=["BATCH-LFP-PVDF-001", "BATCH-SAMPLE-A", "BATCH-HIGH-LOADING", "BATCH-FIG-2"],
    )

    description: str | None = Field(
        None,
        description=(
            "Brief description of this batch. "
            "Summarize the key characteristics or experimental conditions."
        ),
        examples=[
            "LFP cathode slurry with PVDF binder, 45 wt% solid loading",
            "Standard formulation prepared with high-shear mixing",
            "Aged batch after 24h rest at room temperature",
        ],
    )

    formulation: SlurryFormulation | None = edge(
        label="HAS_FORMULATION",
        default=None,
        description=(
            "The formulation (recipe) used for this batch. Links to the composition specification."
        ),
    )

    preparation_history: List[PreparationStep] = edge(
        label="HAS_PREPARATION_STEP",
        default_factory=list,
        description=(
            "Ordered list of preparation steps. "
            "Extract all mixing and processing steps in chronological order. "
            "Look for methods section describing 'First...', 'Then...', 'Finally...'. "
            "Order is critical - affects gel formation and yield behavior."
        ),
    )

    post_mix_age_time: QuantityWithUnit | None = Field(
        None,
        description=(
            "Aging or rest time after mixing. "
            "Look for 'aged for', 'rested for', 'stored for' before testing. "
            "Extract duration and unit (h, days, min)."
        ),
        examples=[
            {"name": "Aging time", "numeric_value": 24, "unit": "h"},
            {"name": "Rest period", "numeric_value": 2, "unit": "days"},
        ],
    )

    storage_temperature: QuantityWithUnit | None = Field(
        None,
        description=(
            "Storage temperature during aging. "
            "Look for 'stored at', 'kept at', 'room temperature'. "
            "Extract temperature and unit (°C, K)."
        ),
        examples=[
            {"name": "Storage temperature", "numeric_value": 25, "unit": "°C"},
            {"name": "Temperature", "text_value": "Room temperature"},
        ],
    )


# ============================================================================
# LAYER 5: RHEOMETRY SETUP
# ============================================================================


class GeometryType(str, Enum):
    """Type of rheometer geometry."""

    PLATE_PLATE = "Plate-Plate"
    CONE_PLATE = "Cone-Plate"
    COUETTE = "Couette"
    VANE_IN_CUP = "Vane-in-Cup"
    PARALLEL_PLATE_ROUGHENED = "Parallel Plate Roughened"
    OTHER = "Other"


class TestMode(str, Enum):
    """Rheology test mode."""

    STEADY_SHEAR_FLOW_CURVE = "Steady Shear Flow Curve"
    OSCILLATION_AMPLITUDE_SWEEP = "Oscillation Amplitude Sweep"
    OSCILLATION_FREQUENCY_SWEEP = "Oscillation Frequency Sweep"
    CREEP_RECOVERY = "Creep Recovery"
    STEP_SHEAR_RATE = "Step Shear Rate"
    STEP_SHEAR_STRESS = "Step Shear Stress"
    OTHER = "Other"


class RheometerSetup(BaseModel):
    """
    Rheometer instrument and geometry configuration.
    Uniquely identified by instrument vendor and model.
    """

    model_config = ConfigDict(graph_id_fields=["instrument_vendor", "instrument_model"])

    instrument_vendor: str = Field(
        description=(
            "Rheometer manufacturer. "
            "Look for company names like 'TA Instruments', 'Anton Paar', 'Malvern', 'Thermo Fisher'. "
            "Extract as written in the methods section."
        ),
        examples=["TA Instruments", "Anton Paar", "Malvern", "Thermo Fisher", "Brookfield"],
    )

    instrument_model: str = Field(
        description=(
            "Rheometer model designation. "
            "Look for model names like 'DHR-3', 'MCR 302', 'Kinexus Pro+'. "
            "Extract exactly as written."
        ),
        examples=["DHR-3", "MCR 302", "Kinexus Pro+", "HAAKE MARS", "Discovery HR-2"],
    )

    geometry_type: GeometryType = Field(
        description=(
            "Type of measuring geometry. "
            "Look for descriptions like 'parallel plate', 'cone-plate', 'vane', 'Couette'. "
            "Common for slurries: parallel plate, vane-in-cup (to prevent slip)."
        ),
        examples=["Plate-Plate", "Cone-Plate", "Vane-in-Cup", "Parallel Plate Roughened"],
    )

    tool_diameter: QuantityWithUnit | None = Field(
        None,
        description=(
            "Diameter of the measuring tool (plate, cone, vane). "
            "Look for 'diameter', 'Ø', 'D ='. "
            "Extract value and unit (mm, cm)."
        ),
        examples=[
            {"name": "Plate diameter", "numeric_value": 40, "unit": "mm"},
            {"name": "Vane diameter", "numeric_value": 25, "unit": "mm"},
        ],
    )

    gap: QuantityWithUnit | None = Field(
        None,
        description=(
            "Gap between measuring surfaces. "
            "Look for 'gap', 'gap height', 'measuring gap'. "
            "Extract value and unit (mm, µm)."
        ),
        examples=[
            {"name": "Gap", "numeric_value": 1.0, "unit": "mm"},
            {"name": "Gap height", "numeric_value": 500, "unit": "µm"},
        ],
    )

    surface_treatment: str | None = Field(
        None,
        description=(
            "Surface treatment to prevent slip. "
            "Look for 'roughened', 'sandblasted', 'serrated', 'cross-hatched'. "
            "Extract as written."
        ),
        examples=["Roughened", "Sandblasted", "Serrated", "Cross-hatched", "Smooth"],
    )

    temperature_control: str | None = Field(
        None,
        description=(
            "Temperature control method. "
            "Look for 'Peltier', 'water bath', 'environmental chamber'. "
            "Extract as written."
        ),
        examples=["Peltier plate", "Water bath", "Environmental chamber", "Active cooling"],
    )

    @field_validator("tool_diameter", "gap", mode="before")
    @classmethod
    def convert_to_quantity(cls, v: Any) -> Any:
        """Convert simple numeric values to QuantityWithUnit."""
        return _to_quantity(v, default_name="Measurement")

    @field_validator("geometry_type", mode="before")
    @classmethod
    def normalize_geometry(cls, v: Any) -> Any:
        """Normalize geometry type enum."""
        return _normalize_enum(GeometryType, v)


class SweepParameters(BaseModel):
    """
    Parameters for sweep tests (flow curve, amplitude sweep, frequency sweep).
    Component - deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    x_axis_quantity: str = Field(
        description=(
            "Quantity varied on x-axis. "
            "Look for 'shear rate', 'shear stress', 'strain amplitude', 'frequency'. "
            "Extract the controlled variable name."
        ),
        examples=[
            "Shear rate",
            "Shear stress",
            "Strain amplitude",
            "Angular frequency",
            "Frequency",
        ],
    )

    x_start: QuantityWithUnit | None = Field(
        None,
        description=(
            "Starting value of x-axis sweep. Extract the minimum or starting value with unit."
        ),
        examples=[
            {"name": "Shear rate start", "numeric_value": 0.01, "unit": "1/s"},
            {"name": "Frequency start", "numeric_value": 0.1, "unit": "Hz"},
        ],
    )

    x_end: QuantityWithUnit | None = Field(
        None,
        description=(
            "Ending value of x-axis sweep. Extract the maximum or ending value with unit."
        ),
        examples=[
            {"name": "Shear rate end", "numeric_value": 1000, "unit": "1/s"},
            {"name": "Frequency end", "numeric_value": 100, "unit": "Hz"},
        ],
    )

    num_points: int | None = Field(
        None,
        description=(
            "Number of measurement points in sweep. "
            "Look for 'points per decade', 'number of points', 'data points'. "
            "Extract the integer value."
        ),
        examples=[10, 20, 50, 100],
    )

    sweep_direction: str | None = Field(
        None,
        description=(
            "Direction of sweep: 'LowToHigh', 'HighToLow', or 'Cyclic'. "
            "Indicates whether the sweep goes from low to high values, high to low, or cycles."
        ),
        examples=["LowToHigh", "HighToLow", "Cyclic"],
    )

    @field_validator("x_start", "x_end", mode="before")
    @classmethod
    def convert_to_quantity(cls, v: Any) -> Any:
        """Convert simple numeric values to QuantityWithUnit."""
        return _to_quantity(v, default_name="Value")


class TestProtocol(BaseModel):
    """
    Rheology test protocol specification.
    Uniquely identified by protocol_id.
    """

    model_config = ConfigDict(graph_id_fields=["protocol_id"])

    protocol_id: str = Field(
        description=(
            "Unique identifier for this protocol. "
            "Create a descriptive ID based on the test type. "
            "Format: 'PROTOCOL-[test-type]'."
        ),
        examples=[
            "PROTOCOL-FLOW-CURVE",
            "PROTOCOL-AMPLITUDE-SWEEP",
            "PROTOCOL-FREQUENCY-SWEEP",
            "PROTOCOL-CREEP-RECOVERY",
        ],
    )

    test_mode: TestMode = Field(
        description=(
            "Type of rheology test. "
            "Look for test descriptions like 'flow curve', 'amplitude sweep', 'frequency sweep', 'oscillation'. "
            "Common modes: Steady Shear Flow Curve, Oscillation Amplitude Sweep, Oscillation Frequency Sweep."
        ),
        examples=[
            "Steady Shear Flow Curve",
            "Oscillation Amplitude Sweep",
            "Oscillation Frequency Sweep",
            "Creep Recovery",
        ],
    )

    temperature_setpoint: QuantityWithUnit | None = Field(
        None,
        description=(
            "Test temperature. Look for 'at', 'temperature', 'T ='. Extract value and unit (°C, K)."
        ),
        examples=[
            {"name": "Temperature", "numeric_value": 25, "unit": "°C"},
            {"name": "Test temperature", "numeric_value": 298, "unit": "K"},
        ],
    )

    equilibration_time: QuantityWithUnit | None = Field(
        None,
        description=(
            "Equilibration or waiting time before test. "
            "Look for 'equilibrated for', 'waited', 'pre-conditioning'. "
            "Extract duration and unit (min, s)."
        ),
        examples=[
            {"name": "Equilibration time", "numeric_value": 5, "unit": "min"},
            {"name": "Wait time", "numeric_value": 300, "unit": "s"},
        ],
    )

    pre_shear_rate: QuantityWithUnit | None = Field(
        None,
        description=(
            "Pre-shear rate if applied. "
            "Look for 'pre-sheared at', 'conditioning shear rate'. "
            "Extract value and unit (1/s, s⁻¹)."
        ),
        examples=[
            {"name": "Pre-shear rate", "numeric_value": 100, "unit": "1/s"},
            {"name": "Conditioning rate", "numeric_value": 50, "unit": "s⁻¹"},
        ],
    )

    pre_shear_duration: QuantityWithUnit | None = Field(
        None,
        description=(
            "Duration of pre-shear. Look for 'for', 'duration'. Extract value and unit (s, min)."
        ),
        examples=[
            {"name": "Pre-shear duration", "numeric_value": 60, "unit": "s"},
            {"name": "Conditioning time", "numeric_value": 2, "unit": "min"},
        ],
    )

    sweep_parameters: SweepParameters | None = Field(
        None,
        description=(
            "Sweep parameters for flow curves and sweeps. "
            "Extract x-axis quantity, start/end values, number of points."
        ),
        examples=[
            {
                "x_axis_quantity": "Shear rate",
                "x_start": {"numeric_value": 0.1, "unit": "1/s"},
                "x_end": {"numeric_value": 1000, "unit": "1/s"},
            }
        ],
    )

    @field_validator(
        "temperature_setpoint",
        "equilibration_time",
        "pre_shear_rate",
        "pre_shear_duration",
        mode="before",
    )
    @classmethod
    def convert_to_quantity(cls, v: Any) -> Any:
        """Convert simple numeric values to QuantityWithUnit."""
        return _to_quantity(v, default_name="Value")

    @field_validator("test_mode", mode="before")
    @classmethod
    def normalize_test_mode(cls, v: Any) -> Any:
        """Normalize test mode enum."""
        return _normalize_enum(TestMode, v)


# ============================================================================
# LAYER 6: TEST RUNS AND RESULTS
# ============================================================================


class RheologyTestRun(BaseModel):
    """
    Execution of a rheology test protocol on a slurry batch.
    Uniquely identified by run_id.
    """

    model_config = ConfigDict(graph_id_fields=["run_id"])

    run_id: str = Field(
        description=(
            "Unique identifier for this test run. "
            "Create a descriptive ID based on the test and sample. "
            "Format: 'RUN-[test-type]-[sample]' or use figure reference."
        ),
        examples=[
            "RUN-FLOW-CURVE-SAMPLE-A",
            "RUN-AMPLITUDE-SWEEP-LFP-001",
            "RUN-FIG-4A",
            "RUN-TABLE-3-ROW-1",
        ],
    )

    description: str | None = Field(
        None,
        description=(
            "Brief description of this test run. "
            "Summarize what was measured and under what conditions."
        ),
        examples=[
            "Flow curve measurement at 25°C",
            "Amplitude sweep to determine linear viscoelastic region",
            "Frequency sweep at 1% strain amplitude",
        ],
    )

    batch_reference: str | None = Field(
        None,
        description=(
            "Reference to the slurry batch tested. "
            "Look for sample names, batch codes, or identifiers. "
            "Extract as written in the text or figure caption."
        ),
        examples=["BATCH-LFP-001", "Sample A", "High loading slurry", "Formulation 1"],
    )

    rheometer_setup: RheometerSetup | None = edge(
        label="USES_RHEOMETER",
        default=None,
        description=(
            "Rheometer and geometry configuration used. Links to the instrument setup details."
        ),
    )

    protocol: TestProtocol | None = edge(
        label="FOLLOWS_PROTOCOL",
        default=None,
        description=("Test protocol followed for this run. Links to the protocol specification."),
    )

    dataset: Optional["RheologyDataset"] = edge(
        label="PRODUCES_DATASET",
        default=None,
        description=("Dataset produced by this test run. Links to the measurement results."),
    )


class RheologyCurve(BaseModel):
    """
    Single rheology curve (x vs y data series).
    Simplified representation of measurement data.
    Uniquely identified by curve_id.
    """

    model_config = ConfigDict(graph_id_fields=["curve_id"])

    curve_id: str = Field(
        description=(
            "Unique identifier for this curve. "
            "Create a descriptive ID based on the measurement type. "
            "Format: 'CURVE-[type]-[sample]' or use figure reference."
        ),
        examples=[
            "CURVE-VISCOSITY-VS-SHEAR-RATE",
            "CURVE-GPRIME-VS-FREQUENCY",
            "CURVE-FIG-3A",
            "CURVE-STRESS-STRAIN",
        ],
    )

    x_quantity: str = Field(
        description=(
            "Quantity on x-axis. "
            "Look for axis labels in figures or table headers. "
            "Common: 'Shear rate', 'Shear stress', 'Strain amplitude', 'Frequency', 'Time'."
        ),
        examples=["Shear rate", "Shear stress", "Strain amplitude", "Angular frequency", "Time"],
    )

    y_quantity: str = Field(
        description=(
            "Quantity on y-axis. "
            "Look for axis labels in figures or table headers. "
            "Common: 'Viscosity', 'Shear stress', 'G' (Storage modulus)', 'G'' (Loss modulus)', 'tan δ'."
        ),
        examples=[
            "Viscosity",
            "Shear stress",
            "Storage modulus (G')",
            "Loss modulus (G'')",
            "tan δ",
            "Complex viscosity",
        ],
    )

    x_unit: str | None = Field(
        None,
        description=(
            "Unit for x-axis values. "
            "Extract from axis labels or legends. "
            "Common: '1/s', 's⁻¹', 'Pa', '%', 'Hz', 'rad/s'."
        ),
        examples=["1/s", "s⁻¹", "Pa", "%", "Hz", "rad/s", "s"],
    )

    y_unit: str | None = Field(
        None,
        description=(
            "Unit for y-axis values. "
            "Extract from axis labels or legends. "
            "Common: 'Pa.s', 'mPa.s', 'Pa', 'kPa'."
        ),
        examples=["Pa.s", "mPa.s", "Pa", "kPa", "dimensionless"],
    )

    x_values: List[float] | None = Field(
        None,
        description=(
            "Array of x-axis data points. "
            "Extract numerical values from tables or digitized figures. "
            "Maintain the order as presented. "
            "Leave None if data not available in extractable form."
        ),
        examples=[[0.01, 0.1, 1.0, 10.0, 100.0], [1, 2, 5, 10, 20, 50, 100]],
    )

    y_values: List[float] | None = Field(
        None,
        description=(
            "Array of y-axis data points. "
            "Extract numerical values from tables or digitized figures. "
            "Must correspond to x_values (same length). "
            "Leave None if data not available in extractable form."
        ),
        examples=[[1.5, 1.2, 0.8, 0.5, 0.3], [100, 95, 85, 70, 50, 30, 20]],
    )

    description: str | None = Field(
        None,
        description=(
            "Description of the curve or measurement conditions. "
            "Extract from figure captions or text. "
            "Include any special conditions or observations."
        ),
        examples=[
            "Flow curve showing shear thinning behavior",
            "Storage modulus increases with frequency",
            "Measured at 25°C after 24h aging",
        ],
    )


class DerivedQuantity(BaseModel):
    """
    Derived rheological quantity (computed scalar).
    Examples: yield stress, viscosity at reference shear rate, crossover point.
    Component - deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    name: str = Field(
        description=(
            "Name of the derived quantity. "
            "Look for reported values like 'yield stress', 'viscosity at 100 s⁻¹', 'G'-G'' crossover'. "
            "Common: 'Yield stress', 'Apparent viscosity', 'Zero-shear viscosity', 'Crossover frequency', "
            "'Linear viscoelastic limit', 'Cohesive energy'."
        ),
        examples=[
            "Yield stress",
            "Apparent viscosity at 100 s⁻¹",
            "Zero-shear viscosity",
            "G'-G'' crossover frequency",
            "Linear viscoelastic strain limit",
            "Storage modulus at 1 Hz",
        ],
    )

    value: QuantityWithUnit | None = Field(
        None,
        description=("Value of the derived quantity. Extract the reported value with its unit."),
        examples=[
            {"name": "Yield stress", "numeric_value": 15.5, "unit": "Pa"},
            {"name": "Viscosity", "numeric_value": 1.2, "unit": "Pa.s"},
            {"name": "Crossover frequency", "numeric_value": 10, "unit": "Hz"},
        ],
    )

    method_description: str | None = Field(
        None,
        description=(
            "Method used to determine this quantity. "
            "Look for descriptions like 'determined from Herschel-Bulkley fit', 'extrapolated to zero shear rate'. "
            "Extract the calculation or determination method."
        ),
        examples=[
            "Determined from Herschel-Bulkley model fit",
            "Extrapolated from low shear rate region",
            "Identified as G'-G'' crossover point",
            "Calculated from stress-strain curve",
        ],
    )


class ModelFit(BaseModel):
    """
    Rheological model fit result.
    Examples: Herschel-Bulkley, Power law, Cross model.
    Component - deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    model_family: str = Field(
        description=(
            "Name of the rheological model. "
            "Look for model names like 'Herschel-Bulkley', 'Power law', 'Cross', 'Carreau', 'Bingham'. "
            "Common models for battery slurries: Herschel-Bulkley, Power law, Casson."
        ),
        examples=["Herschel-Bulkley", "Power law", "Cross", "Carreau", "Bingham", "Casson"],
    )

    parameters: List[QuantityWithUnit] = Field(
        default_factory=list,
        description=(
            "Model parameters with their fitted values. "
            "Look for parameter values like 'τ₀ = 15 Pa', 'K = 2.5 Pa.sⁿ', 'n = 0.45'. "
            "Common parameters: τ₀ (yield stress), K (consistency index), n (flow index), "
            "η₀ (zero-shear viscosity), η∞ (infinite-shear viscosity)."
        ),
        examples=[
            [
                {"name": "τ₀ (yield stress)", "numeric_value": 15.0, "unit": "Pa"},
                {"name": "K (consistency)", "numeric_value": 2.5, "unit": "Pa.sⁿ"},
                {"name": "n (flow index)", "numeric_value": 0.45, "unit": "dimensionless"},
            ]
        ],
    )

    goodness_of_fit: QuantityWithUnit | None = Field(
        None,
        description=(
            "Goodness of fit metric. "
            "Look for R², R, RMSE, or other fit quality indicators. "
            "Extract the metric name, value, and unit (if applicable)."
        ),
        examples=[
            {"name": "R²", "numeric_value": 0.995, "unit": "dimensionless"},
            {"name": "RMSE", "numeric_value": 0.05, "unit": "Pa"},
        ],
    )

    fit_range: str | None = Field(
        None,
        description=(
            "Range over which the model was fitted. "
            "Look for descriptions like 'fitted in range 0.1-100 s⁻¹', 'excluding low shear rate data'. "
            "Extract the range specification."
        ),
        examples=[
            "Fitted in shear rate range 0.1-100 s⁻¹",
            "Excluding low shear rate region (<0.01 s⁻¹)",
            "Applied to entire flow curve",
        ],
    )


class RheologyDataset(BaseModel):
    """
    Collection of rheology measurement results.
    Contains curves, derived quantities, and model fits.
    Uniquely identified by dataset_id.
    """

    model_config = ConfigDict(graph_id_fields=["dataset_id"])

    dataset_id: str = Field(
        description=(
            "Unique identifier for this dataset. "
            "Create a descriptive ID based on the measurement type and sample. "
            "Format: 'DATASET-[type]-[sample]' or use figure/table reference."
        ),
        examples=[
            "DATASET-FLOW-CURVE-LFP-001",
            "DATASET-OSCILLATION-SAMPLE-A",
            "DATASET-FIG-3",
            "DATASET-TABLE-2",
        ],
    )

    description: str | None = Field(
        None,
        description=(
            "Brief description of this dataset. Summarize what measurements are included."
        ),
        examples=[
            "Flow curve and oscillation data for LFP cathode slurry",
            "Rheological characterization at multiple solid loadings",
            "Temperature-dependent viscosity measurements",
        ],
    )

    curves: List[RheologyCurve] = edge(
        label="HAS_CURVE",
        default_factory=list,
        description=(
            "List of measurement curves in this dataset. "
            "Extract all curves from figures or tables. "
            "Each curve represents one x-y relationship."
        ),
    )

    derived_quantities: List[DerivedQuantity] = Field(
        default_factory=list,
        description=(
            "List of derived quantities reported. "
            "Extract computed values like yield stress, viscosity at reference shear rate. "
            "Look for reported values in text, tables, or figure captions."
        ),
        examples=[
            [
                {
                    "name": "Yield stress",
                    "value": {"name": "τ₀", "numeric_value": 15.5, "unit": "Pa"},
                    "method_description": "Herschel-Bulkley fit",
                },
                {
                    "name": "Viscosity at 100 s⁻¹",
                    "value": {"name": "η", "numeric_value": 0.8, "unit": "Pa.s"},
                },
            ]
        ],
    )

    model_fits: List[ModelFit] = Field(
        default_factory=list,
        description=(
            "List of rheological model fits. "
            "Extract fitted models with their parameters. "
            "Look for model equations and parameter values in text or tables."
        ),
        examples=[
            [
                {
                    "model_family": "Herschel-Bulkley",
                    "parameters": [
                        {"name": "τ₀", "numeric_value": 15.0, "unit": "Pa"},
                        {"name": "K", "numeric_value": 2.5, "unit": "Pa.sⁿ"},
                        {"name": "n", "numeric_value": 0.45, "unit": "dimensionless"},
                    ],
                    "goodness_of_fit": {"name": "R²", "numeric_value": 0.995},
                }
            ]
        ],
    )

    exclusion_notes: str | None = Field(
        None,
        description=(
            "Notes about excluded data or regions. "
            "Look for statements like 'slip observed at high shear rates', 'fracture at high strain'. "
            "Extract any data quality or exclusion notes."
        ),
        examples=[
            "Wall slip observed above 500 s⁻¹",
            "Sample fracture at strain >50%",
            "Low shear rate data excluded due to instrument limitations",
        ],
    )


# ============================================================================
# FORWARD REFERENCES RESOLUTION
# ============================================================================

# Update forward references for recursive/circular relationships
ScholarlyRheologyPaper.model_rebuild()
SlurryRheologyStudy.model_rebuild()
SlurryRheologyExperiment.model_rebuild()
RheologyTestRun.model_rebuild()
