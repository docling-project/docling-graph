"""
Pydantic templates for French ID Card extraction.

These models include descriptions and concrete examples in each field to guide
the language model, improving the accuracy and consistency of the extracted data.
The schema is designed to be converted into a knowledge graph.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, Any, List
from datetime import date
import re

# --- Edge Helper Function ---
def Edge(label: str, **kwargs: Any) -> Any:
    """
    Helper function to create a Pydantic Field with edge metadata.
    The 'edge_label' defines the type of relationship in the graph.
    """
    return Field(..., json_schema_extra={'edge_label': label}, **kwargs)

# --- Reusable Component: Address ---
class Address(BaseModel):
    """
    A flexible, generic model for a physical address.
    It's treated as a component, so it has no graph_id_fields.
    Its ID will be a hash of its content, making it unique to its context.
    """
    street_address: Optional[str] = Field(
        None,
        description="Street name and number",
        examples=["123 Rue de la Paix", "90 Boulevard Voltaire"]
    )
    city: Optional[str] = Field(
        None,
        description="City",
        examples=["Paris", "Lyon"]
    )
    state_or_province: Optional[str] = Field(
        None,
        description="State, province, or region",
        examples=["Île-de-France"]
    )
    postal_code: Optional[str] = Field(
        None,
        description="Postal or ZIP code",
        examples=["75001", "69002"]
    )
    country: Optional[str] = Field(
        None,
        description="Country",
        examples=["France"]
    )

    def __str__(self):
        parts = [self.street_address, self.city, self.state_or_province, self.postal_code, self.country]
        return ", ".join(p for p in parts if p)

# --- Reusable Entity: Person ---
class Person(BaseModel):
    """
    A generic model for a person.
    A person is uniquely identified by their full name and date of birth.
    """
    model_config = ConfigDict(graph_id_fields=['given_names', 'last_name', 'date_of_birth'])
    
    given_names: Optional[List[str]] = Field(
        default=None,
        description="List of given names (first names usually seperated with a comma) of the person, in order",
        examples=[["Pierre"], ["Pierre", "Louis"], ["Pierre", "Louis", "André"]]
    )
    last_name: Optional[str] = Field(
        None,
        description="The person's family name (surname)",
        examples=["Dupont", "Martin"]
    )
    alternate_name: Optional[str] = Field(
        None,
        description="The person's alterante name",
        examples=["Doe", "MJ"]
    )
    date_of_birth: Optional[date] = Field(
        None,
        description=(
            "The cardholder's date of birth.",
            "Look for text like 'Date of birth', 'Date de naiss.', or similar.",
            "The model should parse dates like 'DD MM YYYY' or 'DDMMYYYY' and normalize them to YYYY-MM-DD format."
        ),
        examples=["1990-05-15"]
    )
    place_of_birth: Optional[str] = Field(
        None,
        description="City and/or country of birth",
        examples=["Paris", "Marseille (France)"]
    )
    gender: Optional[str] = Field(
        None,
        description="Gender or sex of the person",
        examples=["F", "M", "Female", "Male"]
    )
    
    # --- Edge Definition ---
    lives_at: Optional[Address] = Edge(
        label="LIVES_AT",
        description="Physical address (e.g., home address)"
    )

    # --- Validator ---
    @field_validator('given_names', mode='before')
    def ensure_list(cls, v):
        """Ensure given_names is always a list."""
        if isinstance(v, str):
            return [v]
        return v
    
    @field_validator('lives_at', mode='before')
    @classmethod
    def parse_address(cls, v):
        """
        Accept both Address objects and strings.
        If string, attempt to parse into Address structure.
        """
        if v is None or isinstance(v, dict):
            return v  # Let Pydantic handle dict -> Address
        
        if isinstance(v, str):
            # Attempt to parse the address string
            # Pattern: "street, additional, postal_code city, country"
            parts = [p.strip() for p in v.split(',')]
            
            # Basic heuristic parsing
            address_dict = {
                'street_address': parts[0] if len(parts) > 0 else None,
                'city': None,
                'postal_code': None,
                'country': parts[-1] if len(parts) > 1 else None
            }
            
            # Try to extract postal code and city
            if len(parts) >= 2:
                # Look for postal code pattern (5 digits for France)
                for part in parts:
                    postal_match = re.search(r'\b(\d{5})\s+(.+)', part)
                    if postal_match:
                        address_dict['postal_code'] = postal_match.group(1)
                        address_dict['city'] = postal_match.group(2)
                        break
                                
            return address_dict        
        return v
    
    def __str__(self):
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p)

# --- Root Document Model: IDCard ---
class IDCard(BaseModel):
    """
    A model for an identification document.
    It is uniquely identified by its document number.
    """
    model_config = ConfigDict(graph_id_fields=['document_number'])
    
    document_type: str = Field(
        "ID Card",
        description="Type of document (e.g., ID Card, Passport, Driver's License)",
        examples=["ID Card", "Passeport"]
    )
    document_number: str = Field(
        ...,
        description="The unique identifier for the document",
        examples=["23AB12345", "19XF56789"]
    )
    issuing_country: Optional[str] = Field(
        None,
        description="The country that issued the document (e.g., 'France', 'République Française')",
        examples=["France", "USA"]
    )
    issue_date: Optional[date] = Field(
        None,
        description=(
            "Date the document was issued.",
            "Look for text like 'Date of Issue', 'Date de délivrence', or similar.",
            "The model should parse dates like 'DD MM YYYY' or 'DDMMYYYY' and normalize them to YYYY-MM-DD format."
        ),
        examples=["2023-10-20"]
    )
    expiry_date: Optional[date] = Field(
        None,
        description=(
            "Date the document expires.",
            "Look for text like 'Expiry Date', 'Date d'expir.', or similar.",
            "The model should parse dates like 'DD MM YYYY' or 'DDMMYYYY' and normalize them to YYYY-MM-DD format."
        ),
        examples=["2033-10-19"]
    )

    # --- Edge Definition ---
    holder: Person = Edge(
        label="BELONGS_TO",
        description="The person this ID card belongs to"
    )

    def __str__(self):
        return f"{self.document_type} {self.document_number}"
