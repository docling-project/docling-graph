"""
Pydantic templates for French ID Card extraction.

These models include descriptions and concrete examples in each field to guide
the language model, improving the accuracy and consistency of the extracted data.
The schema is designed to be converted into a knowledge graph.
"""

import re
from datetime import date
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- Edge Helper Function ---


def edge(label: str, default: Any = None, **kwargs: Any) -> Any:
    """
    Helper function to create a Pydantic Field with edge metadata.
    The 'edge_label' defines the type of relationship in the graph.
    Use default=None or default_factory=list for optional/list edges.
    """
    json_schema_extra = dict(kwargs.pop("json_schema_extra", {}) or {})
    json_schema_extra["edge_label"] = label
    if "default_factory" in kwargs:
        default_factory = kwargs.pop("default_factory")
        return Field(default_factory=default_factory, json_schema_extra=json_schema_extra, **kwargs)
    if default is not None or "default" in kwargs:
        return Field(
            default=kwargs.pop("default", default), json_schema_extra=json_schema_extra, **kwargs
        )
    return Field(..., json_schema_extra=json_schema_extra, **kwargs)


# --- Reusable Component: Address ---


class Address(BaseModel):
    """
    Represents a physical address (component).
    In delta extraction, nested address payloads may be flattened.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    street_address: str | None = Field(
        None,
        description="Street name and number",
        examples=["123 Rue de la Paix", "90 Boulevard Voltaire"],
    )

    city: str | None = Field(None, description="City", examples=["Paris", "Lyon"])

    state_or_province: str | None = Field(
        None, description="State, province, or region", examples=["Île-de-France"]
    )

    postal_code: str | None = Field(
        None, description="Postal or ZIP code", examples=["75001", "69002"]
    )

    country: str | None = Field(None, description="Country", examples=["France"])

    def __str__(self) -> str:
        parts = [
            self.street_address,
            self.city,
            self.state_or_province,
            self.postal_code,
            self.country,
        ]
        return ", ".join(p for p in parts if p)


# --- Reusable Entity: Person ---


class Person(BaseModel):
    """
    A generic model for a person.
    Identity uses last_name (required) plus given_names and date_of_birth for stable deduplication.
    """

    model_config = ConfigDict(
        graph_id_fields=["last_name", "given_names", "date_of_birth"],
        extra="ignore",
        populate_by_name=True,
    )

    given_names: List[str] | None = Field(
        default=None,
        description=(
            "List of given names (first names), in order. "
            "LOOK FOR: First name(s) on the document. EXTRACT: As list. "
            "EXAMPLES: ['Pierre'], ['Pierre', 'Louis'], ['Marie', 'Claire']"
        ),
        examples=[["Pierre"], ["Pierre", "Louis"], ["Pierre", "Louis", "André"]],
    )

    last_name: str = Field(
        ...,
        description=(
            "The person's family name (surname). Required for identity. "
            "LOOK FOR: 'Nom', 'Family name', 'Surname' on the document. "
            "EXAMPLES: 'Dupont', 'Martin', 'Bernard'"
        ),
        examples=["Dupont", "Martin", "Bernard"],
    )

    alternate_name: str | None = Field(
        None, description="The person's alternate name", examples=["Doe", "MJ"]
    )

    date_of_birth: date | None = Field(
        None,
        description=(
            "The cardholder's date of birth. "
            "LOOK FOR: 'Date of birth', 'Date de naiss.', 'Naissance'. "
            "EXTRACT: Parse DD MM YYYY or DDMMYYYY and normalize to YYYY-MM-DD. "
            "EXAMPLES: '1990-05-15', '1985-12-01'"
        ),
        examples=["1990-05-15", "1985-12-01"],
    )

    place_of_birth: str | None = Field(
        None, description="City and/or country of birth", examples=["Paris", "Marseille (France)"]
    )

    gender: str | None = Field(
        None, description="Gender or sex of the person", examples=["F", "M", "Female", "Male"]
    )

    # --- Edge Definition ---
    lives_at: Address | None = edge(
        label="LIVES_AT",
        default=None,
        description="Physical address (e.g., home address). Omit when not present in document.",
    )

    # --- Validators ---
    @field_validator("given_names", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> Any:
        """Ensure given_names is always a list."""
        if isinstance(v, str):
            # Handle comma-separated or space-separated names
            if "," in v:
                return [name.strip() for name in v.split(",")]
            return [v]
        return v

    @field_validator("lives_at", mode="before")
    @classmethod
    def parse_address(cls, v: Any) -> Any:
        """
        Accept both Address objects and strings.
        If string, attempt to parse into Address structure.
        """
        if v is None or isinstance(v, dict):
            return v  # Let Pydantic handle dict -> Address
        if isinstance(v, str):
            # Attempt to parse the address string
            parts = [p.strip() for p in v.split(",")]
            # Basic heuristic parsing
            address_dict = {
                "street_address": parts[0] if len(parts) > 0 else None,
                "city": None,
                "postal_code": None,
                "country": parts[-1] if len(parts) > 1 else None,
            }
            # Try to extract postal code and city
            if len(parts) >= 2:
                for part in parts:
                    postal_match = re.search(r"\b(\d{5})\s+(.+)", part)
                    if postal_match:
                        address_dict["postal_code"] = postal_match.group(1)
                        address_dict["city"] = postal_match.group(2)
                        break
            return address_dict
        return v

    def __str__(self) -> str:
        first_names = " ".join(self.given_names) if self.given_names else ""
        parts = [first_names, self.last_name]
        return " ".join(p for p in parts if p) or "Unknown"


# --- Root Document Model: IDCard ---


class IDCard(BaseModel):
    """
    A model for an identification document.
    It is uniquely identified by its document number.
    """

    model_config = ConfigDict(
        graph_id_fields=["document_number"], extra="ignore", populate_by_name=True
    )

    document_number: str = Field(
        ...,
        description="The unique identifier for the document",
        examples=["23AB12345", "19XF56789", "1234567890"],
    )

    issuing_country: str | None = Field(
        None,
        description="The country that issued the document (e.g., 'France', 'République Française')",
        examples=["France", "USA", "Deutschland"],
    )

    issue_date: date | None = Field(
        None,
        description=(
            "Date the document was issued.",
            "Look for text like 'Date of Issue', 'Date de délivrance', or similar.",
            "The model should parse dates like 'DD MM YYYY' or 'DDMMYYYY' and normalize them to YYYY-MM-DD format.",
        ),
        examples=["2023-10-20"],
    )

    expiry_date: date | None = Field(
        None,
        description=(
            "Date the document expires.",
            "Look for text like 'Expiry Date', 'Date d'expiration', or similar.",
            "The model should parse dates like 'DD MM YYYY' or 'DDMMYYYY' and normalize them to YYYY-MM-DD format.",
        ),
        examples=["2033-10-19"],
    )

    # --- Edge Definition ---
    holder: Person | None = edge(
        label="BELONGS_TO",
        default=None,
        description=(
            "The person this ID card belongs to. "
            "EXTRACT: When holder info is present. Omit when not in extracted batch."
        ),
    )

    def __str__(self) -> str:
        return f"{self.issuing_country} {self.document_number}"
