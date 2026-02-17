"""
Improved Pydantic template for Slurry-Battery Rheology Research Papers.

Extracts structured data from battery electrode slurry rheology research papers,
focusing on steady shear and oscillatory rheology measurements.

Key improvements:
- Enhanced field descriptions with LOOK FOR / EXTRACT / EXAMPLES pattern.
- Strict typing (Enum, date, float) with robust validators.
- Explicit Entity vs. Component separation.
- Prevention of placeholder values ("N/A").
- Full restoration of all original measurement fields (sweeps, model fits, pre-shear).
"""

import re
from datetime import date, datetime
from enum import Enum
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def edge(label: str, **kwargs: Any) -> Any:
    """
    Helper function to create a Pydantic Field with edge metadata.
    The 'edge_label' defines the type of relationship in the knowledge graph.

    Args:
        label: Edge label in ALL_CAPS format (e.g., "HAS_STUDY")
        **kwargs: Additional Field parameters
    """
    # Extract default if provided, otherwise use ellipsis for required fields
    if "default" not in kwargs and "default_factory" not in kwargs:
        kwargs["default"] = ...
    return Field(json_schema_extra={"edge_label": label}, **kwargs)


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

        # Last attempt: direct value match or value creation
        try:
            return enum_cls(v)
        except ValueError:
            # Safe fallback to OTHER if present
            if "OTHER" in enum_cls.__members__:
                return enum_cls.OTHER
            # If no OTHER, return as is (Pydantic will raise validation error)
            return v

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

    model_config = ConfigDict(is_entity=False, extra="ignore")

    name: str | None = Field(
        None,
        description=(
            "Name of the measured property. "
            "LOOK FOR: Property names like 'Temperature', 'Viscosity', 'Shear rate'. "
            "EXTRACT: The full descriptive name as it appears. "
            "IMPORTANT: If not explicitly stated, leave None (auto-generated later). "
            "EXAMPLES: 'Temperature', 'Shear rate', 'Viscosity', 'Solid loading'"
        ),
        examples=["Temperature", "Shear rate", "Viscosity"],
    )

    numeric_value: float | None = Field(
        None,
        description=(
            "Single numerical value for the measurement. "
            "LOOK FOR: Numbers associated with units. "
            "EXTRACT: Extract only the number, removing any units or text. "
            "IMPORTANT: Use this for single-point measurements. "
            "EXAMPLES: 25.0, 1.6, 8.2, 2600"
        ),
        examples=[25.0, 1.6, 8.2],
    )

    numeric_value_min: float | None = Field(
        None,
        description=(
            "Minimum value for range measurements. "
            "LOOK FOR: Patterns like '80-90', '1.5 to 2.0'. "
            "EXTRACT: The lower bound of the range. "
            "EXAMPLES: 80.0, 1.5, 20.0"
        ),
        examples=[80.0, 1.5, 20.0],
    )

    numeric_value_max: float | None = Field(
        None,
        description=(
            "Maximum value for range measurements. "
            "LOOK FOR: Patterns like '80-90', '1.5 to 2.0'. "
            "EXTRACT: The upper bound of the range. "
            "EXAMPLES: 90.0, 2.0, 30.0"
        ),
        examples=[90.0, 2.0, 30.0],
    )

    text_value: str | None = Field(
        None,
        description=(
            "Textual value if not numerical. "
            "LOOK FOR: Qualitative descriptions. "
            "EXTRACT: As-is from the document. "
            "IMPORTANT: Do NOT use placeholders like 'N/A'. "
            "EXAMPLES: 'High', 'Low', 'Stable', 'Room temperature'"
        ),
        examples=["High", "Low", "Stable"],
    )

    unit: str | None = Field(
        None,
        description=(
            "Unit of measurement. "
            "LOOK FOR: Units like '°C', 'mPa.s', 'Hz', 'µm', 'wt%'. "
            "EXTRACT: Exactly as written, preserving symbols. "
            "EXAMPLES: '°C', 'mPa.s', 'Hz', 'µm', 'wt%'"
        ),
        examples=["°C", "mPa.s", "Hz", "µm", "wt%"],
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_scalar_or_string_to_dict(cls, v: Any) -> Any:
        """
        Accept scalar or string input from LLM and normalize to structured dict.
        - int | float -> {"numeric_value": float(v)}
        - str -> parse numeric (and optional unit); else {"text_value": v}
        - dict -> pass through unchanged
        """
        if isinstance(v, dict):
            return v
        if isinstance(v, int | float):
            return {"numeric_value": float(v)}
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return {"text_value": v}
            # Try leading number: optional minus, digits, optional dot and decimals, optional exponent
            num_match = re.match(r"^([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*(.*)$", s)
            if num_match:
                num_str, rest = num_match.group(1), num_match.group(2).strip()
                try:
                    num_val = float(num_str)
                except ValueError:
                    return {"text_value": v}
                out = {"numeric_value": num_val}
                if rest:
                    out["unit"] = rest
                return out
            # No leading number: treat as qualitative text
            return {"text_value": v}
        return v

    @field_validator("numeric_value", "numeric_value_min", "numeric_value_max", mode="before")
    @classmethod
    def coerce_numbers(cls, v: Any) -> Any:
        """Coerce string numbers to float."""
        if isinstance(v, str):
            clean = re.sub(r"[^\d\.\-eE]", "", v)
            try:
                return float(clean)
            except ValueError:
                return None
        return v

    @model_validator(mode="after")
    def validate_and_set_defaults(self) -> Self:
        """Auto-generate name if missing and handle ranges."""
        has_single = self.numeric_value is not None
        has_min = self.numeric_value_min is not None
        has_max = self.numeric_value_max is not None

        # Auto-generate name if missing
        if not self.name:
            if has_min and has_max:
                self.name = "Value range"
            elif has_single:
                self.name = "Value"
            elif self.text_value:
                self.name = "Property"
            else:
                self.name = "Measurement"

        # Allow implicit range: if numeric_value + min/max, treat as range
        if has_single and (has_min or has_max):
            if has_max and not has_min:
                self.numeric_value_min = self.numeric_value
                self.numeric_value = None
            elif has_min and not has_max:
                self.numeric_value_max = self.numeric_value
                self.numeric_value = None

        return self


# ============================================================================
# LAYER 1: SCHOLARLY DOCUMENT
# ============================================================================


class PersonIdentity(BaseModel):
    """
    Person identity component.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    full_name: str = Field(
        ...,
        description=(
            "Full name of the person. "
            "LOOK FOR: Author names in header, footnotes. "
            "EXTRACT: 'FirstName LastName' or as written. "
            "EXAMPLES: 'John Smith', 'Maria Garcia', 'A. Kumar'"
        ),
        examples=["John Smith", "Maria Garcia", "A. Kumar"],
    )

    orcid: str | None = Field(
        None,
        description=(
            "ORCID identifier if available. "
            "LOOK FOR: 'ORCID:', URL 'orcid.org/XXXX...'. "
            "EXTRACT: The numeric identifier (0000-0000-0000-0000). "
            "EXAMPLES: '0000-0002-1234-5678'"
        ),
        examples=["0000-0002-1234-5678"],
    )


class ScholarlyIdentifier(BaseModel):
    """
    Scholarly identifier component (DOI, arXiv, URL).
    Optional fields for partial extraction when identifiers are absent.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    scheme: str | None = Field(
        None,
        description=(
            "Type of identifier. "
            "LOOK FOR: 'DOI:', 'arXiv:', 'URL:'. "
            "EXTRACT: Omit when not in document. "
            "EXAMPLES: 'DOI', 'arXiv', 'URL', 'PMID'"
        ),
        examples=["DOI", "arXiv", "URL"],
    )

    value: str | None = Field(
        None,
        description=(
            "The identifier value. "
            "EXTRACT: Full ID string or URL. Omit when not in document. "
            "EXAMPLES: '10.1016/j.jpowsour.2023.233456', 'https://example.com'"
        ),
        examples=["10.1016/j.jpowsour.2023.233456"],
    )


class ScholarlyRheologyPaper(BaseModel):
    """
    Root entity: The research paper itself.
    Uniquely identified by title.
    """

    model_config = ConfigDict(graph_id_fields=["title"])

    title: str = Field(
        description=(
            "Full title of the research paper. "
            "LOOK FOR: Large bold text at the top of the first page. "
            "EXTRACT: Exactly as written, including subtitles. "
            "EXAMPLES: 'Rheological Properties of LiFePO4 Cathode Slurries'"
        ),
        examples=["Rheological Properties of LiFePO4 Cathode Slurries"],
    )

    authors: List[PersonIdentity] = Field(
        default_factory=list,
        description=(
            "List ALL paper authors. "
            "LOOK FOR: Every name on the title/author line (comma-separated or on the same line). "
            "EXTRACT: One object per author; do not stop after the first name. "
            "EXAMPLES: [{'full_name': 'John Doe'}, {'full_name': 'Jane Smith'}]"
        ),
    )

    institutions: List[str] = Field(
        default_factory=list,
        description=(
            "Research institutions or affiliations. "
            "LOOK FOR: Affiliations below author names or in footnotes. "
            "EXTRACT: Full institution names. "
            "EXAMPLES: ['MIT', 'Stanford University', 'CNRS France']"
        ),
        examples=[["MIT", "Stanford University"], ["University of Tokyo"]],
    )

    publication_date: date | None = Field(
        None,
        description=(
            "Publication or submission date. "
            "LOOK FOR: 'Published:', 'Received:', date stamps. "
            "EXTRACT: Parse to YYYY-MM-DD. "
            "IMPORTANT: If only year is found, use YYYY-01-01. "
            "EXAMPLES: '2024-01-15', '2023-11-20'"
        ),
        examples=["2024-01-15"],
    )

    identifiers: List[ScholarlyIdentifier] = Field(
        default_factory=list,
        description=(
            "Document identifiers. "
            "LOOK FOR: DOIs, URLs, arXiv IDs in header/footer. "
            "EXAMPLES: [{'scheme': 'DOI', 'value': '10.1016/...'}]"
        ),
    )

    abstract: str | None = Field(
        None,
        description=(
            "Paper abstract text. "
            "LOOK FOR: Section labeled 'Abstract'. "
            "EXTRACT: The full paragraph text."
        ),
    )

    keywords: List[str] = Field(
        default_factory=list,
        description=(
            "Paper keywords. "
            "LOOK FOR: 'Keywords:', 'Key words:'. "
            "EXTRACT: List of strings. "
            "EXAMPLES: ['rheology', 'battery slurry', 'yield stress']"
        ),
        examples=[["rheology", "battery slurry"]],
    )

    studies: List["SlurryRheologyStudy"] = edge(
        label="HAS_STUDY",
        default_factory=list,
        description=(
            "List of experimental studies in the paper. "
            "Each study represents a distinct experimental campaign or section."
        ),
    )

    @model_validator(mode="after")
    def deduplicate_authors_by_name(self) -> Self:
        """Keep first occurrence of each author per full_name (removes duplicates from chunked extraction)."""
        if not self.authors:
            return self
        seen: set[str] = set()
        unique: list[PersonIdentity] = []
        for a in self.authors:
            key = (getattr(a, "full_name", None) or "").strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)
        object.__setattr__(self, "authors", unique)
        return self

    @field_validator("publication_date", mode="before")
    @classmethod
    def parse_date(cls, v: Any) -> Any:
        """Parse various date string formats to date object."""
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            # Try YYYY-MM-DD
            try:
                return date.fromisoformat(v)
            except ValueError:
                pass
            # Try simple Year
            if re.match(r"^\d{4}$", v.strip()):
                return date(int(v), 1, 1)
            # Try finding a year in the string as fallback
            match = re.search(r"(\d{4})", v)
            if match:
                return date(int(match.group(1)), 1, 1)
        return None

    @field_validator("keywords", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> Any:
        """Ensure keywords are a list."""
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v


# ============================================================================
# LAYER 2: STUDY AND EXPERIMENT
# ============================================================================


class SlurryRheologyStudy(BaseModel):
    """
    Coherent experimental campaign (e.g., 'Effect of Binder MW').
    Uniquely identified by study_id.
    """

    model_config = ConfigDict(graph_id_fields=["study_id"])

    study_id: str = Field(
        description=(
            "Short, stable, human-readable identifier for this study. "
            "EXTRACT: Prefer descriptive IDs from the study's objective or content (e.g. 'STUDY-TEMPERATURE-DEPENDENCE', 'Phenomenological fitting'), "
            "or document-derived labels that are meaningful on their own (e.g. section number '3.1', or 'FIG-4' when the study is clearly about that figure). "
            "Avoid using only a section letter or Roman numeral (e.g. 'C', 'V', 'VI') as the study_id; prefer a descriptive label or combine with topic (e.g. 'IV-A-Temperature-dependence'). "
            "EXAMPLES: 'STUDY-BINDER-MW', 'STUDY-TEMPERATURE-DEPENDENCE', 'Phenomenological fitting', 'FIG-4'"
        ),
        examples=[
            "STUDY-BINDER-MW",
            "STUDY-TEMPERATURE-DEPENDENCE",
            "Phenomenological fitting",
            "FIG-4",
        ],
    )

    objective: str | None = Field(
        None,
        description=(
            "Goal of this specific study. "
            "LOOK FOR: 'We investigate...', 'The objective is...'. "
            "EXTRACT: The core research question. "
            "EXAMPLES: 'Investigate effect of binder molecular weight on viscosity'"
        ),
        examples=["Investigate effect of binder molecular weight"],
    )

    experiments: List["SlurryRheologyExperiment"] = edge(
        label="HAS_EXPERIMENT",
        default_factory=list,
        description="List of specific experiments (sample + condition comparisons) in this study.",
    )


class SlurryRheologyExperiment(BaseModel):
    """
    Unit of scientific comparison (e.g., 'Sample A vs Sample B').
    Uniquely identified by experiment_id.
    """

    model_config = ConfigDict(graph_id_fields=["experiment_id"])

    experiment_id: str = Field(
        description=(
            "Unique identifier for this experiment. "
            "EXTRACT: Prefer figure/table reference from document (e.g. 'Fig-3A', 'Table-2'). "
            "Else create from conditions (e.g. 'EXP-LFP-PVDF-20VOL'). "
            "EXAMPLES: 'Fig-3A', 'EXP-LFP-PVDF-20VOL', 'EXP-FIG-3A'"
        ),
        examples=["Fig-3A", "EXP-LFP-PVDF-20VOL", "EXP-FIG-3A"],
    )

    description: str | None = Field(
        None,
        description=(
            "Description of experimental conditions. "
            "LOOK FOR: Figure captions, table headers. "
            "EXTRACT: Summary of what is being tested. "
            "EXAMPLES: 'LFP slurry with PVDF binder at 20 vol% solid loading'"
        ),
        examples=["LFP slurry with PVDF binder at 20 vol% solid loading"],
    )

    slurry_batch: Optional["BatterySlurryBatch"] = edge(
        label="USES_BATCH",
        default=None,
        description="The specific slurry batch (material + preparation) used.",
    )

    rheometry_runs: List["RheologyTestRun"] = edge(
        label="HAS_TEST_RUN",
        default_factory=list,
        description="List of rheology tests performed on this slurry.",
    )

    reported_findings: List[str] = Field(
        default_factory=list,
        description=(
            "Key findings/observations. "
            "LOOK FOR: 'Results show...', 'We observed...'. "
            "EXTRACT: List of finding strings. "
            "EXAMPLES: ['Shear thinning behavior observed', 'Yield stress increases with solids']"
        ),
        examples=[["Shear thinning behavior observed"]],
    )


# ============================================================================
# LAYER 3: FORMULATION
# ============================================================================


class ComponentRole(str, Enum):
    """Role of component in battery slurry."""

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
            "Role of this component. "
            "LOOK FOR: 'active material', 'binder', 'conductive additive'. "
            "EXTRACT: Map to nearest enum value. "
            "EXAMPLES: 'Active Material', 'Polymer Binder'"
        ),
        examples=["Active Material", "Polymer Binder", "Conductive Additive"],
    )

    material_name: str = Field(
        description=(
            "Material name or formula. "
            "LOOK FOR: Chemical formulas, trade names. "
            "EXTRACT: Full name. "
            "EXAMPLES: 'LiFePO4', 'PVDF', 'Carbon black Super C65'"
        ),
        examples=["LiFePO4", "PVDF", "Carbon black Super C65"],
    )

    material_supplier: str | None = Field(
        None,
        description="Material supplier (e.g., Sigma-Aldrich, Timcal).",
        examples=["Sigma-Aldrich", "Timcal"],
    )

    amount_value: float | None = Field(
        None,
        description=(
            "Numeric amount. "
            "LOOK FOR: Numbers in composition tables. "
            "EXTRACT: Number only. "
            "If the text gives alternatives (e.g. '5 or 10 wt%'), create separate formulation/component entries for each variant or extract a range; do not omit. "
            "EXAMPLES: 85.0, 10.0"
        ),
        examples=[85.0, 10.0],
    )

    amount_unit: str | None = Field(
        None,
        description=(
            "Unit for the amount. "
            "LOOK FOR: 'wt%', 'vol%', 'g'. "
            "EXTRACT: Exactly as written. "
            "EXAMPLES: 'wt%', 'vol%', 'parts'"
        ),
        examples=["wt%", "vol%", "g"],
    )

    particle_size: QuantityWithUnit | None = Field(
        None,
        description=(
            "Particle size info (D50, median, mean diameter, etc.). "
            "LOOK FOR: 'D50', 'particle size', 'median of X nm', 'mean particle size of X', "
            "'particle size of X', 'diameter'. "
            "EXTRACT: As quantity (numeric_value + unit) or text_value. "
            "EXAMPLES: {'name': 'D50', 'numeric_value': 5.0, 'unit': 'µm'}, "
            "'median of 130 nm' -> numeric_value 0.13, unit 'µm', or text_value 'median 130 nm'"
        ),
        examples=[{"name": "D50", "numeric_value": 0.13, "unit": "µm"}],
    )

    molecular_weight: QuantityWithUnit | None = Field(
        None,
        description=(
            "Molecular weight (for polymers). "
            "LOOK FOR: 'Mw', 'MW'. "
            "EXTRACT: As quantity. "
            "EXAMPLES: {'name': 'Mw', 'numeric_value': 500, 'unit': 'kDa'}"
        ),
    )

    @field_validator("component_role", mode="before")
    @classmethod
    def normalize_role(cls, v: Any) -> Any:
        return _normalize_enum(ComponentRole, v)

    @model_validator(mode="after")
    def clear_amount_if_property_unit(self) -> Self:
        """Clear amount_value/amount_unit when unit indicates a property (temp, viscosity), not a quantity."""
        unit = self.amount_unit
        if not unit or not isinstance(unit, str):
            return self
        normalized = re.sub(r"[\s·]", "", unit.lower())
        forbidden_substrings = ("°c", "k", "pa.s", "pas", "mpa.s", "mpas")
        is_pressure_pa = normalized == "pa" or (len(normalized) > 2 and normalized.endswith("pa"))
        if any(f in normalized for f in forbidden_substrings) or is_pressure_pa:
            object.__setattr__(self, "amount_value", None)
            object.__setattr__(self, "amount_unit", None)
        return self


class SlurryFormulation(BaseModel):
    """
    The recipe used for the slurry.
    Uniquely identified by formulation_id.
    """

    model_config = ConfigDict(graph_id_fields=["formulation_id"])

    formulation_id: str = Field(
        description=(
            "Unique identifier for formulation. "
            "EXTRACT: Prefer table/section reference (e.g. 'FORM-TABLE-1', 'Table-1'). "
            "Else 'FORM-[short description]' (e.g. 'FORM-LFP-90-5-5'). "
            "EXAMPLES: 'FORM-TABLE-1', 'FORM-LFP-90-5-5'"
        ),
        examples=["FORM-TABLE-1", "FORM-LFP-90-5-5"],
    )

    description: str | None = Field(
        None,
        description="Brief description of the formulation (e.g., '90:5:5 LFP:CB:PVDF').",
        examples=["90:5:5 LFP:CB:PVDF cathode slurry"],
    )

    components: List[SlurryComponent] = edge(
        label="HAS_COMPONENT",
        default_factory=list,
        description=(
            "List of ingredients in this formulation. "
            "Include a component for every distinct material mentioned (e.g. solvent, binder, active material, conductive additive)."
        ),
    )

    target_solid_loading: QuantityWithUnit | None = Field(
        None,
        description="Target solid content (e.g., 50 wt%).",
    )


# ============================================================================
# LAYER 4: BATCH AND PREPARATION
# ============================================================================


class StepType(str, Enum):
    """Type of slurry preparation step."""

    DRY_PREMIX = "Dry Premix"
    BINDER_DISSOLUTION = "Binder Dissolution"
    DISPERSION = "Dispersion"
    MIXING = "Mixing"
    DEAGGLOMERATION = "Deagglomeration"
    REST = "Rest"
    DEGASSING = "Degassing"
    OTHER = "Other"


class PreparationStep(BaseModel):
    """
    Single step in mixing process.
    Uniquely identified by step_id.
    """

    model_config = ConfigDict(graph_id_fields=["step_id"])

    step_id: str = Field(
        description=(
            "Unique ID for the step. "
            "EXTRACT: Prefer step number from document (e.g. '1', 'Step-1'). "
            "Else 'STEP-[number]-[type]' (e.g. 'STEP-1-DISSOLUTION'). "
            "EXAMPLES: 'Step-1', 'STEP-1-DISSOLUTION', 'STEP-2-MIXING'"
        ),
        examples=["Step-1", "STEP-1-DISSOLUTION", "STEP-2-MIXING"],
    )

    step_type: StepType = Field(
        StepType.OTHER,
        description="Type of step (Mixing, Rest, etc.).",
        examples=["Mixing", "Rest"],
    )

    description: str | None = Field(
        None,
        description="Full text description of the step procedure.",
        examples=["Mixed at 2000 rpm for 30 min"],
    )

    equipment_name: str | None = Field(
        None,
        description="Equipment used (e.g., 'Planetary mixer').",
        examples=["Planetary mixer", "Thinky mixer"],
    )

    equipment_model: str | None = Field(
        None,
        description="Equipment model number (e.g., 'ARE-310').",
        examples=["ARE-310", "T25 digital"],
    )

    equipment_vendor: str | None = Field(
        None,
        description="Equipment manufacturer (e.g., 'Thinky', 'IKA').",
        examples=["Thinky", "IKA"],
    )

    parameters: List[QuantityWithUnit] = Field(
        default_factory=list,
        description="Process parameters (Speed, Time, Temp).",
    )

    @field_validator("step_type", mode="before")
    @classmethod
    def normalize_step(cls, v: Any) -> Any:
        if v is None:
            return StepType.OTHER
        return _normalize_enum(StepType, v)


class BatterySlurryBatch(BaseModel):
    """
    Physical batch produced.
    Uniquely identified by batch_id.
    """

    model_config = ConfigDict(graph_id_fields=["batch_id"])

    batch_id: str = Field(
        description=(
            "Unique ID for the batch. "
            "EXTRACT: Prefer label from document (e.g. 'Batch 1', 'Sample A'). "
            "Else 'BATCH-[short label]' (e.g. 'BATCH-LFP-001'). "
            "EXAMPLES: 'Batch-1', 'BATCH-LFP-001'"
        ),
        examples=["Batch-1", "BATCH-LFP-001"],
    )

    formulation: SlurryFormulation | None = edge(
        label="HAS_FORMULATION",
        default=None,
        description="Link to the recipe used.",
    )

    preparation_history: List[PreparationStep] = edge(
        label="HAS_PREPARATION_STEP",
        default_factory=list,
        description="Ordered list of steps to make this batch.",
    )

    post_mix_age_time: QuantityWithUnit | None = Field(
        None,
        description="Rest time before testing.",
    )

    storage_temperature: QuantityWithUnit | None = Field(
        None,
        description="Storage temperature during aging.",
    )


# ============================================================================
# LAYER 5: RHEOMETRY SETUP
# ============================================================================


class GeometryType(str, Enum):
    PLATE_PLATE = "Plate-Plate"
    CONE_PLATE = "Cone-Plate"
    COUETTE = "Couette"
    VANE = "Vane"
    OTHER = "Other"


class TestMode(str, Enum):
    FLOW_CURVE = "Steady Shear Flow Curve"
    AMPLITUDE_SWEEP = "Oscillation Amplitude Sweep"
    FREQUENCY_SWEEP = "Oscillation Frequency Sweep"
    CREEP = "Creep Recovery"
    OTHER = "Other"


class RheometerSetup(BaseModel):
    """
    Instrument configuration.
    Uniquely identified by model and geometry.
    """

    model_config = ConfigDict(graph_id_fields=["instrument_model", "geometry_type"])

    instrument_vendor: str | None = Field(
        None,
        description="Manufacturer (e.g., TA Instruments).",
        examples=["TA Instruments", "Anton Paar"],
    )

    instrument_model: str = Field(
        description="Model name (e.g., DHR-3).",
        examples=["DHR-3", "MCR 302"],
    )

    geometry_type: GeometryType = Field(
        description=(
            "Geometry used. Map 'parallel plate', 'parallel disk', or 'plate-plate' to 'Plate-Plate'. "
            "Do not use Other when the text matches a known type."
        ),
        examples=["Plate-Plate", "Cone-Plate"],
    )

    gap: QuantityWithUnit | None = Field(
        None,
        description="Gap setting.",
    )

    tool_diameter: QuantityWithUnit | None = Field(
        None,
        description=(
            "Diameter of the measuring tool. "
            "LOOK FOR: 'diameter', 'Ø'. "
            "EXTRACT: As quantity. "
            "EXAMPLES: {'name': 'Diameter', 'numeric_value': 40, 'unit': 'mm'}"
        ),
    )

    surface_treatment: str | None = Field(
        None,
        description="Surface finish (e.g., Sandblasted, Roughened).",
        examples=["Sandblasted", "Serrated"],
    )

    temperature_control: str | None = Field(
        None,
        description="Temperature control system (e.g., Peltier).",
        examples=["Peltier plate", "Water bath"],
    )

    @field_validator("geometry_type", mode="before")
    @classmethod
    def normalize_geo(cls, v: Any) -> Any:
        if isinstance(v, str) and "parallel" in v.lower() and "plate" in v.lower():
            return GeometryType.PLATE_PLATE
        return _normalize_enum(GeometryType, v)


class SweepParameters(BaseModel):
    """
    Parameters for sweep tests (flow curve, amplitude sweep, frequency sweep).
    Component - deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    x_axis_quantity: str = Field(
        description=(
            "Quantity varied on x-axis. "
            "LOOK FOR: 'shear rate', 'stress', 'frequency'. "
            "EXAMPLES: 'Shear rate', 'Angular frequency'"
        ),
        examples=["Shear rate", "Angular frequency"],
    )

    x_start: QuantityWithUnit | None = Field(
        None,
        description="Starting value of sweep.",
    )

    x_end: QuantityWithUnit | None = Field(
        None,
        description="Ending value of sweep.",
    )

    num_points: int | None = Field(
        None,
        description="Number of data points measured.",
        examples=[10, 50, 100],
    )

    sweep_direction: str | None = Field(
        None,
        description="Direction (LowToHigh, HighToLow).",
        examples=["LowToHigh", "HighToLow"],
    )


class TestProtocol(BaseModel):
    """
    Rheology test settings.
    Uniquely identified by protocol_id.
    """

    model_config = ConfigDict(graph_id_fields=["protocol_id"])

    protocol_id: str = Field(
        description=(
            "Unique ID for the protocol. "
            "EXTRACT: Prefer test name from document; else 'PROTOCOL-[type]'. "
            "EXAMPLES: 'Flow curve', 'PROTOCOL-FLOW-CURVE'"
        ),
        examples=["Flow curve", "PROTOCOL-FLOW-CURVE"],
    )

    test_mode: TestMode | None = Field(
        None,
        description="Test mode (Flow curve, Sweep).",
        examples=["Steady Shear Flow Curve"],
    )

    temperature_setpoint: QuantityWithUnit | None = Field(
        None,
        description="Test temperature. Look in Methods for temperature values.",
    )

    equilibration_time: QuantityWithUnit | None = Field(
        None,
        description=(
            "Wait time before test starts. "
            "Look in Methods for 'equilibration', 'equilibrat'; extract values even if mid-paragraph."
        ),
    )

    pre_shear_rate: QuantityWithUnit | None = Field(
        None,
        description=(
            "Conditioning shear rate applied before test. "
            "Look in Methods for 'pre-shear', 'shear rate'; extract values even if mid-paragraph."
        ),
    )

    pre_shear_duration: QuantityWithUnit | None = Field(
        None,
        description=(
            "Duration of pre-shear. "
            "Look in Methods for 'pre-sheared for', 'duration'; extract values even if mid-paragraph."
        ),
    )

    sweep_parameters: SweepParameters | None = Field(
        None,
        description="Parameters if this is a sweep test.",
    )

    @field_validator("test_mode", mode="before")
    @classmethod
    def normalize_mode(cls, v: Any) -> Any:
        return _normalize_enum(TestMode, v)


# ============================================================================
# LAYER 6: TEST RUNS AND RESULTS
# ============================================================================


class RheologyCurve(BaseModel):
    """
    X-Y Data series.
    Uniquely identified by curve_id.
    """

    model_config = ConfigDict(graph_id_fields=["curve_id"])

    curve_id: str = Field(
        description=(
            "Unique ID for the curve. "
            "EXTRACT: Prefer figure/legend label (e.g. 'Fig-2a', 'Sample A'). "
            "Else 'CURVE-[type]-[sample]'. "
            "Look in figure captions and data tables for curve identifiers. "
            "EXAMPLES: 'Fig-2a', 'CURVE-VISCOSITY-SAMPLE-A'"
        ),
        examples=["Fig-2a", "CURVE-VISCOSITY-SAMPLE-A"],
    )

    x_quantity: str | None = Field(
        None,
        description=(
            "X-axis label (e.g., Shear rate). "
            "Look in figure captions and axis labels for the quantity name."
        ),
        examples=["Shear rate", "Frequency"],
    )

    y_quantity: str | None = Field(
        None,
        description=(
            "Y-axis label (e.g., Viscosity). "
            "Look in figure captions and axis labels for the quantity name."
        ),
        examples=["Viscosity", "Storage modulus"],
    )

    x_unit: str | None = Field(
        None,
        description="Unit for X-axis.",
        examples=["1/s", "Hz"],
    )

    y_unit: str | None = Field(
        None,
        description="Unit for Y-axis.",
        examples=["Pa.s", "Pa"],
    )

    x_values: List[float] | None = Field(
        None,
        description="List of X numerical values.",
        examples=[[0.1, 1.0, 10.0]],
    )

    y_values: List[float] | None = Field(
        None,
        description="List of Y numerical values (must match X length).",
        examples=[[10.0, 5.0, 2.0]],
    )

    description: str | None = Field(
        None,
        description="Brief description of the curve.",
        examples=["Viscosity flow curve at 25C"],
    )


class DerivedQuantity(BaseModel):
    """
    Computed scalar (e.g., Yield Stress).
    Component.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    name: str = Field(
        description="Name of quantity (e.g., Yield stress).",
        examples=["Yield stress", "Zero-shear viscosity"],
    )

    value: QuantityWithUnit | None = Field(
        None,
        description=(
            "The calculated value. "
            "LOOK FOR: Values in tables, figure captions, and result sections. "
            "EXTRACT: Numeric value and unit (e.g. 198 Pa)."
        ),
    )

    method: str | None = Field(
        None,
        description="Method used (e.g., Herschel-Bulkley fit).",
        examples=["Herschel-Bulkley fit", "Crossover point"],
    )


class ModelFit(BaseModel):
    """
    Rheological model fit result.
    Component.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    model_family: str = Field(
        description="Name of model (Herschel-Bulkley, Power law).",
        examples=["Herschel-Bulkley", "Power law"],
    )

    parameters: List[QuantityWithUnit] = Field(
        default_factory=list,
        description="Fitted parameters (yield stress, K, n).",
    )

    goodness_of_fit: QuantityWithUnit | None = Field(
        None,
        description="Fit quality (R2, RMSE).",
    )

    fit_range: str | None = Field(
        None,
        description="Data range used for fitting.",
        examples=["0.1-100 s-1"],
    )


class RheologyDataset(BaseModel):
    """
    Container for results.
    Uniquely identified by dataset_id.
    """

    model_config = ConfigDict(graph_id_fields=["dataset_id"])

    dataset_id: str = Field(
        description=(
            "Unique ID for the dataset. "
            "EXTRACT: Prefer sample/figure reference from document; else 'DATASET-[sample]'. "
            "EXAMPLES: 'Sample-A', 'DATASET-SAMPLE-A'"
        ),
        examples=["Sample-A", "DATASET-SAMPLE-A"],
    )

    description: str | None = Field(
        None,
        description="Description of dataset contents.",
    )

    curves: List[RheologyCurve] = edge(
        label="HAS_CURVE",
        default_factory=list,
        description="Measured curves.",
    )

    derived_quantities: List[DerivedQuantity] = Field(
        default_factory=list,
        description="Scalar results derived from data.",
    )

    model_fits: List[ModelFit] = Field(
        default_factory=list,
        description="Rheological model fits applied to data.",
    )

    exclusion_notes: str | None = Field(
        None,
        description=(
            "Notes about excluded data (slip, fracture). "
            "EXTRACT: Any mentioned data quality issues."
        ),
        examples=["Wall slip observed above 500 s-1"],
    )


class RheologyTestRun(BaseModel):
    """
    Execution of a test.
    Uniquely identified by run_id.
    """

    model_config = ConfigDict(graph_id_fields=["run_id"])

    run_id: str = Field(
        description=(
            "Unique ID for the test run. "
            "EXTRACT: Prefer label from document (e.g. 'Run 1', 'Test A'); else 'RUN-[test]-[sample]'. "
            "EXAMPLES: 'Run-1', 'RUN-FLOW-SAMPLE-A'"
        ),
        examples=["Run-1", "RUN-FLOW-SAMPLE-A"],
    )

    description: str | None = Field(
        None,
        description="Brief description of the test run.",
    )

    batch_reference: str | None = Field(
        None,
        description="Text reference to the batch/sample tested.",
    )

    rheometer_setup: RheometerSetup | None = edge(
        label="USES_RHEOMETER",
        default=None,
        description="Link to instrument used.",
    )

    protocol: TestProtocol | None = edge(
        label="FOLLOWS_PROTOCOL",
        default=None,
        description="Link to protocol used.",
    )

    dataset: RheologyDataset | None = edge(
        label="PRODUCES_DATASET",
        default=None,
        description="Link to results.",
    )


# ============================================================================
# FORWARD REFERENCES RESOLUTION
# ============================================================================

ScholarlyRheologyPaper.model_rebuild()
SlurryRheologyStudy.model_rebuild()
SlurryRheologyExperiment.model_rebuild()
RheologyTestRun.model_rebuild()
