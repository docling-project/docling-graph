"""A narrow, year-scoped extraction schema for Pfizer revenue analysis."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RevenueCategory(BaseModel):
    """One current-year business revenue category, as named in the filing."""

    model_config = ConfigDict(graph_id_fields=["fiscal_year", "name"], extra="ignore")

    fiscal_year: int = Field(
        ..., description="The report's fiscal year, not a comparative prior year."
    )
    name: str = Field(
        ..., description="Revenue category or business segment name exactly as reported."
    )
    amount: float | None = Field(
        None, description="Current fiscal year's revenue as printed, without rescaling."
    )
    unit: str | None = Field(
        None, description="Currency and scale, e.g. USD millions. Required when amount is present."
    )
    evidence: str | None = Field(
        None, description="Short source phrase or table heading supporting the figure."
    )


class PfizerAnnualReport(BaseModel):
    """Pfizer 10-K revenue breakdown for one fiscal year.

    Extract current-year, non-overlapping business/segment revenue categories.
    Do not include geography breakdowns or comparative prior-year columns as
    additional categories; those would double-count the same revenue.
    """

    model_config = ConfigDict(graph_id_fields=["company_name", "fiscal_year"], extra="ignore")

    company_name: str = Field(..., description="Reporting company's legal name, e.g. Pfizer Inc.")
    fiscal_year: int = Field(..., description="Current fiscal year of this Form 10-K.")
    total_revenue: float | None = Field(
        None, description="Consolidated current-year total revenue as printed."
    )
    total_revenue_unit: str | None = Field(None, description="Currency and scale of total_revenue.")
    revenue_categories: list[RevenueCategory] = Field(
        default_factory=list,
        description="Named current-year revenue types/segments with amounts; omit missing amounts.",
        json_schema_extra={"edge_label": "HAS_REVENUE_CATEGORY"},
    )
