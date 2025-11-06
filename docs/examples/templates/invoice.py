"""
Pydantic models defining the schema for invoice data extraction.

These models include descriptions and concrete examples in each field to guide
the language model, improving the accuracy and consistency of the extracted data.
The schema is designed to be converted into a knowledge graph.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# A special object to define graph edges
def edge(label: str, **kwargs: Any) -> Any:
    """Helper function to create a Pydantic Field with edge metadata."""
    # Note: The '...' makes the field required. Use default=None for optional.
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)


# --- Node Definitions ---


class Address(BaseModel):
    """Represents a physical address entity."""

    model_config = ConfigDict(is_entity=False)

    street: str = Field(
        description="Street name and number", examples=["Marktgasse 28", "Rue du Lac 1268"]
    )
    postal_code: str = Field(description="Postal or ZIP code", examples=["9400", "2501"])
    city: str = Field(description="City or town name", examples=["Rorschach", "Biel"])
    country: Optional[str] = Field(
        default=None, description="Country, preferably as a two-letter code.", examples=["CH"]
    )


class Issuer(BaseModel):
    """Represents a company or organization entity that issues the invoice."""

    name: str = Field(
        description="The legal name of the organization", examples=["Robert Schneider AG", "Slack"]
    )
    phone: Optional[str] = Field(
        default=None, description="Contact phone number", examples=["059/987 6540"]
    )
    email: Optional[str] = Field(
        default=None, description="Contact email address", examples=["robert@rschneider.ch"]
    )
    website: Optional[str] = Field(
        default=None, description="Company website URL", examples=["www.rschneider.ch"]
    )

    # --- Edge Definition ---
    located_at: Address = edge(label="LOCATED_AT")


class Client(BaseModel):
    """Represents an individual person or entity receiving the invoice."""

    name: str = Field(
        description="Full name of the person or client entity",
        examples=["Pia Rutschmann", "MineralTree"],
    )
    phone: Optional[str] = Field(
        default=None, description="Client phone number", examples=["059/987 6540"]
    )
    email: Optional[str] = Field(
        default=None, description="Client email address", examples=["client@client.com"]
    )
    website: Optional[str] = Field(
        default=None, description="Client website URL", examples=["www.client.ch"]
    )

    # --- Edge Definition ---
    lives_at: Address = edge(label="LIVES_AT")


class LineItem(BaseModel):
    """Represents a single line item within the invoice."""

    description: str = Field(
        description="Description of the service or product",
        examples=["Garden work", "Disposal of cuttings", "Business+ Monthly User License"],
    )
    quantity: float = Field(description="The quantity of the item", examples=[28.0, 1.0, 115.0])
    unit: Optional[str] = Field(
        description="The unit of measurement for the quantity",
        examples=["Std.", "pcs", "hours", "user"],
    )
    unit_price: float = Field(description="The price per unit", examples=[120.00, 307.35, 15.00])
    total: float = Field(
        description="The total price for this line item (quantity * unit_price)",
        examples=[3360.00, 307.35, 1725.00],
    )


# --- Central Node with Edges ---


class Invoice(BaseModel):
    """The central node representing the entire invoice document."""

    bill_no: str = Field(
        description="The unique invoice identifier or bill number", examples=["3139", "1223113"]
    )
    date: str = Field(
        description="Date the invoice was issued, preferably in YYYY-MM-DD format",
        examples=["01.07.2020", "2023-12-01"],
    )
    currency: str = Field(
        description="The currency of the invoice amounts (e.g., 'CHF', 'USD', 'EUR')",
        examples=["CHF", "USD"],
    )
    subtotal: float = Field(
        description="The total amount before tax or other fees", examples=[3667.35, 1725.0]
    )
    vat_rate: Optional[float] = Field(
        default=None,
        description="The numeric Value Added Tax rate as a percentage without the '%' symbol",
        examples=[7.7, 0.0],
    )
    vat_amount: float = Field(description="The total amount of VAT charged", examples=[282.40, 0.0])
    total: float = Field(
        description="The final, total amount to be paid", examples=[3949.75, 1725.0]
    )

    # --- Edge Definitions ---
    issued_by: Issuer = edge(label="ISSUED_BY")
    sent_to: Client = edge(label="SENT_TO")
    contains_items: List[LineItem] = edge(label="CONTAINS_ITEM")
