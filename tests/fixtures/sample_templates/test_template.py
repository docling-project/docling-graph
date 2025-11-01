"""
Sample Pydantic template for testing.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SampleInvoice(BaseModel):
    """Sample invoice template for testing."""

    invoice_number: str = Field(description="Invoice number")
    date: str = Field(description="Invoice date")
    total_amount: float = Field(description="Total amount")
    vendor_name: str = Field(description="Vendor name")
    items: List[str] = Field(default_factory=list, description="Line items")

    model_config = {"graph_id_fields": ["invoice_number"]}


class SamplePerson(BaseModel):
    """Sample person template for testing."""

    first_name: str = Field(description="First name")
    last_name: str = Field(description="Last name")
    age: Optional[int] = Field(default=None, description="Age")
    email: str = Field(description="Email address")

    model_config = {"graph_id_fields": ["email"]}


class SampleCompany(BaseModel):
    """Sample company template for testing."""

    company_name: str = Field(description="Company name")
    industry: str = Field(description="Industry")
    founded_year: int = Field(description="Year founded")
    employees: List[SamplePerson] = Field(default_factory=list)

    model_config = {"graph_id_fields": ["company_name"]}
