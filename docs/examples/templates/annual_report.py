"""
Annual Report extraction template.

Extracts a graph-ready structure from corporate annual reports / 10-K style
filings (e.g. IBM's Form 10-K Exhibit 13: Management Discussion, Consolidated
Financial Statements, Notes, governance and stockholder information).

Annual reports are long (100+ pages) and repeat the same facts in several
places — a segment's revenue appears in the MD&A narrative, in a "Year in
Review" table AND again in the segment note; a named acquisition is
mentioned in the shareholder letter, in the MD&A and again with full detail
in the acquisitions note. This template is designed for the `dense`
extraction contract: entities that are genuinely discussed in multiple
places (segments, people, acquisitions, the auditor) are modeled with
short, stable, document-derived identities so scattered mentions are
discovered once and filled with scoped context from every place they were
observed, instead of fragmenting into duplicate nodes. Purely tabular,
self-contained facts (the statements themselves, non-GAAP reconciliations)
are kept as components nested on the root, since the authoritative table is
a complete source on its own.

Key entities:
- AnnualReport (root): the filing itself, identified by company + fiscal year.
- BusinessSegment: a reportable segment (e.g. Software, Consulting).
- Person: minimal shared identity behind both BoardMember and ExecutiveOfficer.
- BoardMember / ExecutiveOfficer: per-report roles, each linked to a Person.
- Acquisition / Divestiture: M&A activity disclosed in the filing.
- Auditor: the independent registered public accounting firm.
- Partnership: named ecosystem/technology partners.

Key relationships:
- AnnualReport --HAS_SEGMENT--> BusinessSegment
- AnnualReport --HAS_BOARD_MEMBER--> BoardMember --IS_PERSON--> Person
- AnnualReport --HAS_EXECUTIVE_OFFICER--> ExecutiveOfficer --IS_PERSON--> Person (same
  Person node when an individual holds both roles)
- AnnualReport --ACQUIRED--> Acquisition --ASSIGNED_TO_SEGMENT--> BusinessSegment
- AnnualReport --DIVESTED--> Divestiture
- AnnualReport --AUDITED_BY--> Auditor
- AnnualReport --PARTNERS_WITH--> Partnership
"""

import re
from datetime import date
from enum import Enum
from typing import Any, List, Type

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def edge(label: str, default: Any = None, *, reference: bool = False, **kwargs: Any) -> Any:
    """
    Helper function to create a Pydantic Field with edge metadata.
    The 'edge_label' defines the type of relationship in the knowledge graph.
    Edges are optional by default (default=None) so a missing relationship
    never fails validation of the whole document; use default_factory=list
    for list edges.

    ``reference=True`` marks an identity-only link (graph_reference): the field
    carries id-only references to an entity whose detail lives elsewhere (or
    that is identity-only by design, like Person). Dense extraction fills such
    fields inside the PARENT's own fill call instead of discovering them as
    separate instances.
    """
    json_schema_extra = dict(kwargs.pop("json_schema_extra", {}) or {})
    json_schema_extra["edge_label"] = label
    if reference:
        json_schema_extra["graph_reference"] = True
    if "default_factory" in kwargs:
        default_factory = kwargs.pop("default_factory")
        return Field(default_factory=default_factory, json_schema_extra=json_schema_extra, **kwargs)
    return Field(
        default=kwargs.pop("default", default), json_schema_extra=json_schema_extra, **kwargs
    )


def _normalize_enum(enum_cls: Type[Enum], v: Any) -> Any:
    """
    Normalize enum values to handle various input formats.
    Accepts enum instances, value strings, or member names with flexible formatting.
    Falls back to 'OTHER' member if present.
    """
    if isinstance(v, enum_cls):
        return v
    if isinstance(v, str):
        key = re.sub(r"[^A-Za-z0-9]+", "", v).lower()
        mapping = {}
        for member in enum_cls:
            mapping[re.sub(r"[^A-Za-z0-9]+", "", member.name).lower()] = member
            mapping[re.sub(r"[^A-Za-z0-9]+", "", member.value).lower()] = member
        if key in mapping:
            return mapping[key]
        try:
            return enum_cls(v)
        except ValueError:
            if "OTHER" in enum_cls.__members__:
                return enum_cls.OTHER
            return v
    return v


def _coerce_float(v: Any) -> Any:
    """
    Coerce formatted numeric strings ('$1,234.5', '10.6%', '(500)') to float.
    Lenient by design: unparsable strings are passed through so Pydantic's own
    validation error surfaces rather than silently dropping data.
    """
    if isinstance(v, str):
        cleaned = v.strip()
        blank_placeholders = {"-", chr(0x2013), chr(0x2014), "N/A", "NM"}  # -, en dash, em dash
        if not cleaned or cleaned in blank_placeholders:
            return None
        is_negative = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = re.sub(r"[,$%\s]", "", cleaned).strip("()")
        try:
            value = float(cleaned)
            return -value if is_negative else value
        except ValueError:
            return v
    return v


# =============================================================================
# ENUMS
# =============================================================================


class FilingType(str, Enum):
    """Type/format of the annual filing."""

    FORM_10K = "Form 10-K"
    ANNUAL_REPORT_TO_SHAREHOLDERS = "Annual Report to Shareholders"
    FORM_20F = "Form 20-F"
    INTEGRATED_REPORT = "Integrated Report"
    OTHER = "Other"


class AcquisitionStatus(str, Enum):
    """Lifecycle status of a disclosed acquisition."""

    COMPLETED = "Completed"
    ANNOUNCED = "Announced"
    PENDING = "Pending"
    TERMINATED = "Terminated"


# =============================================================================
# COMPONENTS (value objects, is_entity=False)
# =============================================================================


class Address(BaseModel):
    """Physical address component (headquarters). Deduplicated by content."""

    model_config = ConfigDict(is_entity=False, extra="ignore")

    street_address: str | None = Field(
        None,
        description="Street name and number. LOOK FOR: company address in the cover page or 'Stockholder Information' section.",
        examples=["New Orchard Road", "One Apple Park Way"],
    )
    city: str | None = Field(None, description="City name.", examples=["Armonk", "Cupertino"])
    state_or_province: str | None = Field(
        None, description="State or province.", examples=["New York", "California"]
    )
    postal_code: str | None = Field(None, description="Postal or ZIP code.", examples=["10504"])
    country: str | None = Field(None, description="Country name.", examples=["United States"])


class Workforce(BaseModel):
    """Employee headcount snapshot (Human Capital section). Deduplicated by content."""

    model_config = ConfigDict(is_entity=False, extra="ignore")

    employee_count: float | None = Field(
        None,
        description=(
            "Total workforce headcount. "
            "LOOK FOR: 'Human Capital' / 'Employees' table. "
            "EXTRACT: The number exactly as printed; do not multiply or rescale it — "
            "record whatever scale the table uses in count_unit instead. "
            "EXAMPLES: 264.3, 286900"
        ),
        examples=[264.3, 286900.0],
    )
    count_unit: str | None = Field(
        None,
        description="Unit the headcount is expressed in, as implied by the table header.",
        examples=["thousands", "actual headcount"],
    )

    @field_validator("employee_count", mode="before")
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


class FinancialLineItem(BaseModel):
    """
    Generic financial line item — an extension point for figures that don't
    have a dedicated named field on the enclosing statement summary.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    label: str | None = Field(
        None,
        description="The line item's caption exactly as printed in the statement or note.",
        examples=["Prepaid pension assets", "Operating lease liabilities", "Contract assets"],
    )
    value: float | None = Field(
        None, description="Current-period value, copied digit-for-digit.", examples=[7544.0, 800.0]
    )
    prior_year_value: float | None = Field(
        None, description="Prior-period value for the same caption, if shown in the same table."
    )
    unit: str | None = Field(
        None,
        description="Unit/scale for this figure if different from the document's default.",
        examples=["$ in millions", "%"],
    )

    @field_validator("value", "prior_year_value", mode="before")
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


class RevenueLineItem(BaseModel):
    """
    Revenue sub-category within a business segment (e.g. a product/offering
    line inside 'Software'). Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    name: str | None = Field(
        None,
        description=(
            "Sub-category name as printed in the segment revenue breakdown table. "
            "EXAMPLES: 'Hybrid Cloud', 'Automation', 'Data', 'Transaction Processing', "
            "'Strategy and Technology', 'Intelligent Operations'"
        ),
        examples=["Hybrid Cloud", "Automation", "Strategy and Technology"],
    )
    revenue: float | None = Field(None, description="Current-year revenue for this sub-category.")
    revenue_prior_year: float | None = Field(
        None, description="Prior-year revenue for this sub-category, if shown."
    )
    yoy_change_pct: float | None = Field(
        None,
        description="Year-to-year percent change as printed (e.g. '12.9%' -> 12.9). Do not compute it yourself.",
    )

    @field_validator("revenue", "revenue_prior_year", "yoy_change_pct", mode="before")
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


class GeographicRevenue(BaseModel):
    """Revenue attributed to a geographic region or country. Deduplicated by content."""

    model_config = ConfigDict(is_entity=False, extra="ignore")

    region_name: str | None = Field(
        None,
        description=(
            "Region or country name as printed in the geographic breakdown table. "
            "EXAMPLES: 'Americas', 'Europe/Middle East/Africa', 'Asia Pacific', 'United States'"
        ),
        examples=["Americas", "Europe/Middle East/Africa", "United States"],
    )
    revenue: float | None = Field(
        None, description="Current-year revenue attributed to this region."
    )
    revenue_prior_year: float | None = Field(
        None,
        description="Prior-year revenue attributed to this region, if shown in the same table.",
    )

    @field_validator("revenue", "revenue_prior_year", mode="before")
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


class IndexComparison(BaseModel):
    """
    Cumulative total shareholder return vs. a benchmark index (performance
    graph). Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    index_name: str | None = Field(
        None,
        description="Name of the company's stock or benchmark index in the performance graph.",
        examples=["S&P 500", "S&P Information Technology", "NASDAQ Composite"],
    )
    one_year_cumulative_return_pct: float | None = Field(
        None,
        description=(
            "One-year cumulative return, expressed as the ending index value of a $100 "
            "investment (e.g. '$138.26' -> 138.26). Copy verbatim; do not annualize or rescale."
        ),
    )
    five_year_cumulative_return_pct: float | None = Field(
        None, description="Five-year cumulative return, same $100-basis convention as above."
    )

    @field_validator(
        "one_year_cumulative_return_pct", "five_year_cumulative_return_pct", mode="before"
    )
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


class ExecutiveCommentary(BaseModel):
    """Letter to shareholders (CEO/Chair commentary). Deduplicated by content."""

    model_config = ConfigDict(is_entity=False, extra="ignore")

    author_name: str | None = Field(
        None, description="Name of the letter's signer.", examples=["Arvind Krishna"]
    )
    author_title: str | None = Field(
        None,
        description="Signer's title as printed under their name.",
        examples=["Chairman, President and Chief Executive Officer"],
    )
    summary: str | None = Field(
        None,
        description="A concise (2-4 sentence) summary of the letter's core message, in your own words.",
    )
    strategic_priorities: List[str] = Field(
        default_factory=list,
        description=(
            "Distinct strategic themes or priorities the letter emphasizes. "
            "EXAMPLES: 'Hybrid cloud and AI', 'Quantum computing investment', "
            "'Productivity initiatives', 'Capital return to shareholders'"
        ),
        examples=[["Hybrid cloud and AI", "Quantum computing investment"]],
    )

    @model_validator(mode="after")
    def _dedupe_priorities(self) -> Self:
        """Keep first occurrence per normalized priority (removes chunk-duplicate echoes)."""
        if not self.strategic_priorities:
            return self
        seen: set[str] = set()
        unique: list[str] = []
        for p in self.strategic_priorities:
            key = p.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(p)
        object.__setattr__(self, "strategic_priorities", unique)
        return self


class IncomeStatementSummary(BaseModel):
    """
    Consolidated income statement (statement of operations). Singleton
    per report; the authoritative table is self-contained, so this is
    modeled as a component rather than an independently-discovered entity.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    services_revenue: float | None = Field(None, description="Revenue from services.")
    sales_revenue: float | None = Field(None, description="Revenue from product/hardware sales.")
    financing_revenue: float | None = Field(
        None, description="Revenue from financing arrangements."
    )
    total_revenue: float | None = Field(None, description="Total revenue, current period.")
    total_revenue_prior_year: float | None = Field(None, description="Total revenue, prior period.")
    total_cost: float | None = Field(None, description="Total cost of revenue.")
    gross_profit: float | None = Field(
        None, description="Total revenue minus total cost, current period."
    )
    gross_profit_prior_year: float | None = Field(None, description="Gross profit, prior period.")
    gross_margin_pct: float | None = Field(
        None, description="Gross profit margin as printed (e.g. '58.2%' -> 58.2)."
    )
    sga_expense: float | None = Field(
        None, description="Selling, general and administrative expense."
    )
    rd_expense: float | None = Field(None, description="Research and development expense.")
    other_income_and_expense: float | None = Field(
        None, description="Net other (income) and expense line, current period."
    )
    interest_expense: float | None = Field(None, description="Interest expense.")
    total_expense: float | None = Field(None, description="Total expense and other (income).")
    pretax_income: float | None = Field(
        None, description="Income from continuing operations before income taxes, current period."
    )
    pretax_income_prior_year: float | None = Field(None, description="Pretax income, prior period.")
    income_tax_provision: float | None = Field(
        None, description="Provision for/(benefit from) income taxes from continuing operations."
    )
    income_from_continuing_operations: float | None = Field(
        None, description="Income from continuing operations, current period."
    )
    income_from_discontinued_operations: float | None = Field(
        None, description="Income/(loss) from discontinued operations, net of tax."
    )
    net_income: float | None = Field(None, description="Net income, current period.")
    net_income_prior_year: float | None = Field(None, description="Net income, prior period.")
    diluted_eps_continuing_ops: float | None = Field(
        None, description="Diluted earnings per share from continuing operations."
    )
    diluted_eps_total: float | None = Field(
        None, description="Total diluted earnings per share, current period."
    )
    diluted_eps_total_prior_year: float | None = Field(
        None, description="Total diluted earnings per share, prior period."
    )
    basic_eps_total: float | None = Field(None, description="Total basic earnings per share.")
    weighted_avg_diluted_shares: float | None = Field(
        None, description="Weighted-average diluted shares outstanding."
    )
    other_comprehensive_income_net_of_tax: float | None = Field(
        None, description="Other comprehensive income/(loss), net of tax."
    )
    total_comprehensive_income: float | None = Field(
        None, description="Total comprehensive income."
    )
    line_items: List[FinancialLineItem] = Field(
        default_factory=list,
        description="Any other income-statement or comprehensive-income line not captured above.",
    )

    @field_validator(
        "services_revenue",
        "sales_revenue",
        "financing_revenue",
        "total_revenue",
        "total_revenue_prior_year",
        "total_cost",
        "gross_profit",
        "gross_profit_prior_year",
        "gross_margin_pct",
        "sga_expense",
        "rd_expense",
        "other_income_and_expense",
        "interest_expense",
        "total_expense",
        "pretax_income",
        "pretax_income_prior_year",
        "income_tax_provision",
        "income_from_continuing_operations",
        "income_from_discontinued_operations",
        "net_income",
        "net_income_prior_year",
        "diluted_eps_continuing_ops",
        "diluted_eps_total",
        "diluted_eps_total_prior_year",
        "basic_eps_total",
        "weighted_avg_diluted_shares",
        "other_comprehensive_income_net_of_tax",
        "total_comprehensive_income",
        mode="before",
    )
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


class BalanceSheetSummary(BaseModel):
    """
    Consolidated balance sheet. Singleton per report; modeled as a
    component since the authoritative table is a complete, self-contained source.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    as_of_date: date | None = Field(
        None, description="Balance sheet date, normalized to YYYY-MM-DD.", examples=["2025-12-31"]
    )
    cash_and_equivalents: float | None = Field(None, description="Cash and cash equivalents.")
    restricted_cash: float | None = Field(None, description="Restricted cash.")
    marketable_securities: float | None = Field(None, description="Marketable securities.")
    accounts_receivable_net: float | None = Field(
        None, description="Notes and accounts receivable, net of allowances."
    )
    inventory: float | None = Field(None, description="Inventory.")
    total_current_assets: float | None = Field(None, description="Total current assets.")
    property_plant_equipment_net: float | None = Field(
        None, description="Property, plant and equipment, net of accumulated depreciation."
    )
    operating_lease_right_of_use_assets: float | None = Field(
        None, description="Operating right-of-use assets, net."
    )
    goodwill: float | None = Field(None, description="Goodwill.")
    intangible_assets_net: float | None = Field(None, description="Intangible assets, net.")
    total_assets: float | None = Field(None, description="Total assets, current period.")
    total_assets_prior_year: float | None = Field(None, description="Total assets, prior period.")
    short_term_debt: float | None = Field(None, description="Short-term debt.")
    long_term_debt: float | None = Field(None, description="Long-term debt.")
    accounts_payable: float | None = Field(None, description="Accounts payable.")
    deferred_income_current: float | None = Field(
        None, description="Deferred income, current portion."
    )
    deferred_income_noncurrent: float | None = Field(
        None, description="Deferred income, noncurrent portion."
    )
    total_current_liabilities: float | None = Field(None, description="Total current liabilities.")
    total_liabilities: float | None = Field(None, description="Total liabilities, current period.")
    total_liabilities_prior_year: float | None = Field(
        None, description="Total liabilities, prior period."
    )
    common_stock_and_apic: float | None = Field(
        None, description="Common stock, par value, and additional paid-in capital."
    )
    retained_earnings: float | None = Field(None, description="Retained earnings.")
    treasury_stock: float | None = Field(None, description="Treasury stock, at cost.")
    accumulated_other_comprehensive_income: float | None = Field(
        None, description="Accumulated other comprehensive income/(loss)."
    )
    noncontrolling_interests: float | None = Field(None, description="Noncontrolling interests.")
    total_equity: float | None = Field(None, description="Total equity, current period.")
    total_equity_prior_year: float | None = Field(None, description="Total equity, prior period.")
    line_items: List[FinancialLineItem] = Field(
        default_factory=list, description="Any other balance-sheet line not captured above."
    )

    @field_validator(
        "cash_and_equivalents",
        "restricted_cash",
        "marketable_securities",
        "accounts_receivable_net",
        "inventory",
        "total_current_assets",
        "property_plant_equipment_net",
        "operating_lease_right_of_use_assets",
        "goodwill",
        "intangible_assets_net",
        "total_assets",
        "total_assets_prior_year",
        "short_term_debt",
        "long_term_debt",
        "accounts_payable",
        "deferred_income_current",
        "deferred_income_noncurrent",
        "total_current_liabilities",
        "total_liabilities",
        "total_liabilities_prior_year",
        "common_stock_and_apic",
        "retained_earnings",
        "treasury_stock",
        "accumulated_other_comprehensive_income",
        "noncontrolling_interests",
        "total_equity",
        "total_equity_prior_year",
        mode="before",
    )
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


class CashFlowSummary(BaseModel):
    """
    Consolidated statement of cash flows. Singleton per report; modeled as
    a component since the authoritative table is a complete, self-contained source.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    depreciation: float | None = Field(None, description="Depreciation expense.")
    amortization_of_intangibles_and_software: float | None = Field(
        None, description="Amortization of capitalized software and acquired intangible assets."
    )
    stock_based_compensation: float | None = Field(
        None, description="Stock-based compensation expense."
    )
    deferred_taxes: float | None = Field(None, description="Deferred tax adjustment.")
    net_cash_from_operations: float | None = Field(
        None, description="Net cash provided by operating activities, current period."
    )
    net_cash_from_operations_prior_year: float | None = Field(
        None, description="Net cash from operating activities, prior period."
    )
    capital_expenditures: float | None = Field(
        None, description="Payments for property, plant and equipment."
    )
    investment_in_software: float | None = Field(
        None, description="Capitalized investment in software."
    )
    acquisitions_net_of_cash_acquired: float | None = Field(
        None, description="Cash paid for acquisition of businesses, net of cash acquired."
    )
    net_cash_from_investing: float | None = Field(
        None, description="Net cash provided by/(used in) investing activities, current period."
    )
    net_cash_from_investing_prior_year: float | None = Field(
        None, description="Net cash from investing activities, prior period."
    )
    debt_proceeds: float | None = Field(None, description="Proceeds from new debt issuance.")
    debt_repayments: float | None = Field(None, description="Payments to settle debt.")
    share_repurchases: float | None = Field(
        None,
        description="Common stock repurchases (e.g. for tax withholdings or buyback programs).",
    )
    proceeds_from_share_issuance: float | None = Field(
        None, description="Proceeds from issuance of shares."
    )
    dividends_paid: float | None = Field(None, description="Cash dividends paid.")
    net_cash_from_financing: float | None = Field(
        None, description="Net cash provided by/(used in) financing activities, current period."
    )
    net_cash_from_financing_prior_year: float | None = Field(
        None, description="Net cash from financing activities, prior period."
    )
    net_change_in_cash: float | None = Field(
        None, description="Net change in cash, cash equivalents and restricted cash."
    )
    cash_at_period_end: float | None = Field(
        None, description="Cash, cash equivalents and restricted cash at period end."
    )
    line_items: List[FinancialLineItem] = Field(
        default_factory=list, description="Any other cash-flow line not captured above."
    )

    @field_validator(
        "depreciation",
        "amortization_of_intangibles_and_software",
        "stock_based_compensation",
        "deferred_taxes",
        "net_cash_from_operations",
        "net_cash_from_operations_prior_year",
        "capital_expenditures",
        "investment_in_software",
        "acquisitions_net_of_cash_acquired",
        "net_cash_from_investing",
        "net_cash_from_investing_prior_year",
        "debt_proceeds",
        "debt_repayments",
        "share_repurchases",
        "proceeds_from_share_issuance",
        "dividends_paid",
        "net_cash_from_financing",
        "net_cash_from_financing_prior_year",
        "net_change_in_cash",
        "cash_at_period_end",
        mode="before",
    )
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


class PerformanceMetrics(BaseModel):
    """
    Non-GAAP and operational key performance indicators the company
    highlights alongside GAAP results. Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore")

    operating_pretax_income: float | None = Field(
        None, description="Operating (non-GAAP) pre-tax income from continuing operations."
    )
    operating_pretax_margin_pct: float | None = Field(
        None, description="Operating (non-GAAP) pre-tax margin, as printed."
    )
    operating_net_income: float | None = Field(
        None, description="Operating (non-GAAP) net earnings."
    )
    operating_diluted_eps: float | None = Field(
        None, description="Diluted operating (non-GAAP) earnings per share."
    )
    free_cash_flow: float | None = Field(
        None,
        description="Free cash flow, as explicitly stated by the company (do not derive it yourself).",
    )
    constant_currency_revenue_growth_pct: float | None = Field(
        None, description="Total revenue year-to-year growth rate adjusted for currency."
    )
    remaining_performance_obligations: float | None = Field(
        None,
        description="Remaining performance obligations (RPO), the backlog of unrecognized contract revenue.",
    )

    @field_validator(
        "operating_pretax_income",
        "operating_pretax_margin_pct",
        "operating_net_income",
        "operating_diluted_eps",
        "free_cash_flow",
        "constant_currency_revenue_growth_pct",
        "remaining_performance_obligations",
        mode="before",
    )
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


class ShareholderReturns(BaseModel):
    """Capital returned to shareholders via dividends and buybacks. Deduplicated by content."""

    model_config = ConfigDict(is_entity=False, extra="ignore")

    dividends_paid_total: float | None = Field(
        None, description="Total cash dividends paid during the period."
    )
    dividend_per_share_quarterly: float | None = Field(
        None, description="Most recently declared quarterly dividend per common share."
    )
    share_repurchases_total: float | None = Field(
        None, description="Total value of common stock repurchased during the period."
    )
    dividend_declaration_date: date | None = Field(
        None,
        description="Date the most recent dividend was declared/announced, normalized to YYYY-MM-DD.",
    )

    @field_validator(
        "dividends_paid_total",
        "dividend_per_share_quarterly",
        "share_repurchases_total",
        mode="before",
    )
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


# =============================================================================
# ENTITIES
# =============================================================================


class Person(BaseModel):
    """
    A named individual. Uniquely identified by full name; kept deliberately
    minimal (identity only) so the same node is safely shared everywhere
    this person is referenced. Role-specific data (title, affiliation) lives
    on BoardMember / ExecutiveOfficer instead of here: when the same person
    is discovered via two different parent lists, duplicate instances merge
    into one node whose conflicting attribute values resolve first-seen-wins
    — role data on the shared node would collide unpredictably for a person
    holding both a board seat and an executive role, while separate role
    entities keep each role's data intact.
    """

    model_config = ConfigDict(graph_id_fields=["full_name"], extra="ignore")

    full_name: str = Field(
        ...,
        description=(
            "The person's full name as printed. "
            "LOOK FOR: 'Board of Directors and Senior Leadership' section, or names "
            "signing the shareholder letter."
        ),
        examples=["Arvind Krishna", "James J. Kavanaugh", "Thomas Buberl"],
    )


class BusinessSegment(BaseModel):
    """
    ONE of the 3-6 REPORTABLE segments named in the segment note (e.g. Software,
    Consulting, Infrastructure). Geographic regions, products/offerings, revenue
    sub-lines and other table rows are NOT segments. There are only a handful.
    Deeper detail: geographies belong in `revenue_by_geography`, offering
    breakdowns in `revenue_lines`; both segment mentions (MD&A narrative and the
    dedicated segment note) describe the same segment and consolidate into one
    node.
    """

    model_config = ConfigDict(graph_id_fields=["name"], extra="ignore")

    name: str = Field(
        ...,
        description=(
            "Segment name exactly as used as a table/section heading. "
            "LOOK FOR: 'Segment Details' in Year in Review, and the segment note "
            "in the financial statement notes. "
            "EXAMPLES: 'Software', 'Consulting', 'Infrastructure', 'Financing', 'Other'"
        ),
        examples=["Software", "Consulting", "Infrastructure"],
    )
    description: str | None = Field(
        None, description="One or two sentences on what this segment covers, if stated."
    )
    revenue: float | None = Field(None, description="Segment revenue, current period.")
    revenue_prior_year: float | None = Field(None, description="Segment revenue, prior period.")
    yoy_revenue_change_pct: float | None = Field(
        None, description="Year-to-year revenue percent change, as printed."
    )
    gross_margin_pct: float | None = Field(
        None, description="Segment gross margin, current period."
    )
    gross_margin_pct_prior_year: float | None = Field(
        None, description="Segment gross margin, prior period."
    )
    segment_profit: float | None = Field(None, description="Segment profit, current period.")
    segment_profit_prior_year: float | None = Field(
        None, description="Segment profit, prior period."
    )
    segment_profit_margin_pct: float | None = Field(
        None, description="Segment profit margin, current period."
    )
    yoy_profit_change_pct: float | None = Field(
        None, description="Year-to-year segment profit percent change, as printed."
    )
    annual_recurring_revenue: float | None = Field(
        None,
        description="Annual recurring revenue (ARR) for this segment, if disclosed (common for a software segment).",
    )
    revenue_lines: List[RevenueLineItem] = Field(
        default_factory=list,
        description="Revenue broken down by sub-category/offering within this segment.",
    )

    @field_validator(
        "revenue",
        "revenue_prior_year",
        "yoy_revenue_change_pct",
        "gross_margin_pct",
        "gross_margin_pct_prior_year",
        "segment_profit",
        "segment_profit_prior_year",
        "segment_profit_margin_pct",
        "yoy_profit_change_pct",
        "annual_recurring_revenue",
        mode="before",
    )
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


class Auditor(BaseModel):
    """
    The independent registered public accounting firm. Uniquely identified
    by firm name.
    """

    model_config = ConfigDict(graph_id_fields=["firm_name"], extra="ignore")

    firm_name: str = Field(
        ...,
        description=(
            "The firm's FULL legal name exactly as printed in the audit report "
            "signature — not an abbreviation or nickname (e.g. "
            "'PricewaterhouseCoopers LLP', never 'PwC'). "
            "LOOK FOR: signature block of the 'Report of Independent Registered "
            "Public Accounting Firm'."
        ),
        examples=[
            "PricewaterhouseCoopers LLP",
            "Ernst & Young LLP",
            "Deloitte & Touche LLP",
            "KPMG LLP",
        ],
    )
    opinion: str | None = Field(
        None,
        description="The audit opinion rendered, in a few words.",
        examples=["Unqualified", "Qualified"],
    )
    critical_audit_matters: List[str] = Field(
        default_factory=list,
        description="Short titles of the critical audit matters discussed in the report.",
        examples=[["Income Taxes - Uncertain Tax Positions"]],
    )
    auditor_since_year: int | None = Field(
        None,
        description="Year the firm (or a firm it acquired) began serving as auditor, if stated.",
        examples=[1923],
    )


class Partnership(BaseModel):
    """
    A named strategic/technology/ecosystem partner mentioned in the business
    description. Uniquely identified by partner name.
    """

    model_config = ConfigDict(graph_id_fields=["partner_name"], extra="ignore")

    partner_name: str = Field(
        ...,
        description="Name of the partner organization, as printed.",
        examples=["Adobe", "AWS", "Microsoft", "SAP"],
    )
    partnership_type: str | None = Field(
        None,
        description="Nature of the partnership, if stated.",
        examples=["Technology Alliance", "Reseller", "Strategic Partner"],
    )


class BoardMember(BaseModel):
    """
    ONE seat on the Board of Directors — extract EVERY director from the
    board-of-directors / director-nominees section (each is listed with an
    outside affiliation or committee membership). Executive officers and named
    executives in the compensation tables are NOT board members unless the
    report seats them on the board; the board list and the executive-officer
    list are disjoint except where one person explicitly holds both roles.
    Uniquely identified by the member's name (a per-report role record — there
    is exactly one board seat per named individual in a given report).
    Deliberately an entity rather than a component: dense extraction then
    discovers each seat as its own catalog instance and fills it with scoped
    context, and keeping role fields off the shared Person node avoids the
    attribute collisions described there when the same person also holds an
    executive role.
    """

    model_config = ConfigDict(graph_id_fields=["full_name"], extra="ignore")

    full_name: str = Field(
        ...,
        description="Full name of the board member, matching the linked person's name exactly.",
        examples=["Arvind Krishna", "Thomas Buberl"],
    )
    title: str | None = Field(
        None,
        description=(
            "This person's primary title as printed under their name in the board-of-directors "
            "list (often a title at an external company, not the reporting company)."
        ),
        examples=["Chief Executive Officer", "Retired Chairman and Chief Executive Officer"],
    )
    affiliation: str | None = Field(
        None,
        description="The organization the title belongs to (may be the reporting company itself for executive directors).",
        examples=["AXA S.A.", "Emerson Electric Co.", "IBM"],
    )
    note: str | None = Field(
        None,
        description="Any footnoted qualifier next to the name in the board list (e.g. term start/end date).",
        examples=["Term on the Board begins on March 1, 2026."],
    )
    person: Person | None = edge(
        label="IS_PERSON",
        reference=True,
        description="The shared identity of this board member (reference by full_name only).",
    )

    @model_validator(mode="after")
    def _sync_person_name(self) -> Self:
        """Fall back to the linked person's name if full_name was left empty."""
        if not self.full_name and self.person is not None:
            object.__setattr__(self, "full_name", self.person.full_name)
        return self


class ExecutiveOfficer(BaseModel):
    """
    A management officer from the 10-K "Information about our Executive Officers"
    item. Do NOT pull names from the board-of-directors list or proxy leadership
    table — a director with no management title is a BoardMember, NOT an
    executive officer. Extract EVERY officer in that roster, not just CEO/CFO.
    Uniquely identified by the officer's name (a per-report role record — one
    executive role per named individual). Kept separate from Person like
    BoardMember.
    """

    model_config = ConfigDict(graph_id_fields=["full_name"], extra="ignore")

    full_name: str = Field(
        ...,
        description="Full name of the executive officer, matching the linked person's name exactly.",
        examples=["James J. Kavanaugh", "Arvind Krishna"],
    )
    title: str | None = Field(
        None,
        description="Full title as printed in the senior leadership list.",
        examples=["Senior Vice President, Finance and Operations and Chief Financial Officer"],
    )
    business_area: str | None = Field(
        None,
        description="The business unit or function this officer leads, if distinct from the title.",
        examples=["IBM Consulting", "IBM Infrastructure", "IBM Research"],
    )
    person: Person | None = edge(
        label="IS_PERSON",
        reference=True,
        description="The shared identity of this executive officer (reference by full_name only).",
    )

    @model_validator(mode="after")
    def _sync_person_name(self) -> Self:
        """Fall back to the linked person's name if full_name was left empty."""
        if not self.full_name and self.person is not None:
            object.__setattr__(self, "full_name", self.person.full_name)
        return self


class Acquisition(BaseModel):
    """
    A disclosed acquisition. Uniquely identified by the target company's name.
    Often mentioned briefly in the shareholder letter and MD&A, then again
    with full purchase-price detail in the acquisitions note — all describe
    the same acquisition and should consolidate into one node.
    """

    model_config = ConfigDict(graph_id_fields=["target_name"], extra="ignore")

    target_name: str = Field(
        ...,
        description=(
            "Name of the acquired company. "
            "LOOK FOR: shareholder letter, MD&A, and the acquisitions & divestitures note. "
            "EXTRACT: The company's short/common name, verbatim."
        ),
        examples=["HashiCorp", "StreamSets", "webMethods", "Confluent"],
    )
    status: AcquisitionStatus | None = Field(
        None,
        description=(
            "Lifecycle status. Map 'completed'/'closed' to Completed, an announced-but-not-closed "
            "deal to Announced, and a deal awaiting a known future close to Pending."
        ),
        examples=["Completed", "Announced"],
    )
    announced_date: date | None = Field(
        None, description="Date the deal was announced, normalized to YYYY-MM-DD."
    )
    closed_date: date | None = Field(
        None, description="Date the acquisition closed/completed, normalized to YYYY-MM-DD."
    )
    purchase_price: float | None = Field(
        None, description="Total purchase price/consideration actually paid, if closed."
    )
    enterprise_value: float | None = Field(
        None,
        description="Total enterprise/equity value disclosed for the deal (common for announced, not-yet-closed deals).",
    )
    goodwill_recognized: float | None = Field(
        None, description="Goodwill recognized from this acquisition."
    )
    description: str | None = Field(
        None, description="One to two sentence business rationale, as stated by the company."
    )
    assigned_segment: BusinessSegment | None = edge(
        label="ASSIGNED_TO_SEGMENT",
        reference=True,
        description=(
            "The reportable segment the acquired business was integrated into. "
            "Reference by name only (e.g. {'name': 'Software'}) — full segment "
            "detail belongs in the root segments list; graph assembly merges "
            "this reference into that node by name."
        ),
    )

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, v: Any) -> Any:
        return _normalize_enum(AcquisitionStatus, v)

    @field_validator("purchase_price", "enterprise_value", "goodwill_recognized", mode="before")
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


class Divestiture(BaseModel):
    """
    A disclosed divestiture or asset sale. Uniquely identified by the
    divested business's name.
    """

    model_config = ConfigDict(graph_id_fields=["divested_name"], extra="ignore")

    divested_name: str = Field(
        ...,
        description="Name of the divested business or asset group, verbatim.",
        examples=["The Weather Company", "QRadar SaaS assets"],
    )
    closed_date: date | None = Field(
        None, description="Date the divestiture closed, normalized to YYYY-MM-DD."
    )
    gain_loss_amount: float | None = Field(
        None, description="Gain/(loss) recognized on the divestiture, if stated."
    )
    description: str | None = Field(None, description="One to two sentence description, as stated.")

    @field_validator("gain_loss_amount", mode="before")
    @classmethod
    def _clean(cls, v: Any) -> Any:
        return _coerce_float(v)


# =============================================================================
# ROOT ANNUAL REPORT
# =============================================================================


class AnnualReport(BaseModel):
    """
    Root annual report entity. Uniquely identified by company name and
    fiscal year, so each year's filing for a company becomes its own node
    while shared children (segments, people, the auditor) can accumulate
    across separately-ingested years if the graph is built up over time.
    """

    model_config = ConfigDict(
        graph_id_fields=["company_name", "fiscal_year"], extra="ignore", populate_by_name=True
    )

    # --- Core identity & filing metadata ---

    company_name: str = Field(
        ...,
        description=(
            "Full legal name of the reporting company. "
            "LOOK FOR: cover page, audit report addressee, page footers. "
            "EXTRACT: Full legal name including 'Corporation'/'Inc.'/'plc' suffix."
        ),
        examples=[
            "International Business Machines Corporation",
            "Apple Inc.",
            "Microsoft Corporation",
        ],
    )
    fiscal_year: int = Field(
        ...,
        description=(
            "The fiscal year this report covers (the most recent/current year when "
            "the document shows multi-year comparatives). "
            "LOOK FOR: cover page ('20XX Annual Report'), 'For the year ended December 31' headers."
        ),
        examples=[2025, 2024],
    )
    fiscal_year_end_date: date | None = Field(
        None,
        description="Fiscal year end date, normalized to YYYY-MM-DD.",
        examples=["2025-12-31"],
    )
    filing_type: FilingType | None = Field(
        None,
        description="Format of this annual filing, if identifiable from the document.",
        examples=["Form 10-K", "Annual Report to Shareholders"],
    )
    filing_date: date | None = Field(
        None,
        description="Date the underlying filing was submitted to the securities regulator, normalized to YYYY-MM-DD.",
    )
    annual_meeting_date: date | None = Field(
        None,
        description="Date of the upcoming annual stockholders' meeting, normalized to YYYY-MM-DD.",
    )
    ticker_symbol: str | None = Field(
        None, description="Stock ticker symbol.", examples=["IBM", "AAPL", "MSFT"]
    )
    stock_exchange: str | None = Field(
        None,
        description="Primary exchange the stock is listed on.",
        examples=["New York Stock Exchange", "NASDAQ"],
    )
    reporting_currency: str | None = Field(
        None, description="ISO currency code figures are reported in.", examples=["USD", "EUR"]
    )
    monetary_unit: str | None = Field(
        None,
        description=(
            "The scale monetary figures are reported at, as stated near table headers "
            "(e.g. '$ in millions'). Record the scale; never rescale the numbers yourself."
        ),
        examples=["millions", "thousands", "billions"],
    )
    headquarters: Address | None = edge(
        label="HEADQUARTERED_AT", description="Corporate headquarters address."
    )
    workforce: Workforce | None = edge(
        label="HAS_WORKFORCE",
        description="Employee headcount snapshot from the Human Capital section.",
    )

    # --- Narrative summaries ---

    business_description: str | None = Field(
        None,
        description="A concise (2-4 sentence) summary of what the company does, from the business description section.",
    )
    critical_accounting_estimates: List[str] = Field(
        default_factory=list,
        description=(
            "Topic areas the MD&A names under 'Critical Accounting Estimates' "
            "(the judgment-heavy areas of the financial statements)."
        ),
        examples=[["Income taxes", "Goodwill impairment", "Retirement-related benefits"]],
    )
    market_risk_summary: str | None = Field(
        None,
        description="Brief summary of the market/currency risk discussion in the MD&A, if present.",
    )

    # --- Narrative + financial components ---

    ceo_letter: ExecutiveCommentary | None = edge(
        label="HAS_SHAREHOLDER_LETTER", description="The letter to shareholders."
    )
    income_statement: IncomeStatementSummary | None = edge(
        label="HAS_INCOME_STATEMENT", description="The consolidated income statement."
    )
    balance_sheet: BalanceSheetSummary | None = edge(
        label="HAS_BALANCE_SHEET", description="The consolidated balance sheet."
    )
    cash_flow_statement: CashFlowSummary | None = edge(
        label="HAS_CASH_FLOW_STATEMENT", description="The consolidated statement of cash flows."
    )
    performance_metrics: PerformanceMetrics | None = edge(
        label="HAS_PERFORMANCE_METRICS",
        description="Non-GAAP and operational KPIs the company reports alongside GAAP results.",
    )
    shareholder_returns: ShareholderReturns | None = edge(
        label="HAS_SHAREHOLDER_RETURNS", description="Dividend and share-repurchase activity."
    )
    revenue_by_geography: List[GeographicRevenue] = edge(
        label="HAS_GEOGRAPHIC_REVENUE",
        default_factory=list,
        description="Revenue broken down by geographic region or country.",
    )
    stock_performance: List[IndexComparison] = edge(
        label="HAS_STOCK_PERFORMANCE",
        default_factory=list,
        description="Cumulative total shareholder return vs. benchmark indices from the performance graph.",
    )

    # --- Structural relationships (edges to entities) ---

    segments: List[BusinessSegment] = edge(
        label="HAS_SEGMENT",
        default_factory=list,
        description=(
            "Reportable business segments. "
            "LOOK FOR: 'Segment Details' in Year in Review, and the segments note. "
            "Extract every segment mentioned, whether or not it recurs in both places."
        ),
    )
    acquisitions: List[Acquisition] = edge(
        label="ACQUIRED",
        default_factory=list,
        description="Companies or businesses acquired, announced, or pending, per the acquisitions note.",
    )
    divestitures: List[Divestiture] = edge(
        label="DIVESTED",
        default_factory=list,
        description="Businesses or asset groups divested or sold.",
    )
    auditor: Auditor | None = edge(
        label="AUDITED_BY", description="The independent registered public accounting firm."
    )
    board_of_directors: List[BoardMember] = edge(
        label="HAS_BOARD_MEMBER",
        default_factory=list,
        description=(
            "Members of the board of directors, from the 'Board of Directors' list. If the "
            "same person also appears in executive_officers, list them in both places — the "
            "linked Person node resolves to the same node either way."
        ),
    )
    executive_officers: List[ExecutiveOfficer] = edge(
        label="HAS_EXECUTIVE_OFFICER",
        default_factory=list,
        description="Named senior leadership/executive officers, from the 'Senior Leadership' list.",
    )
    strategic_partners: List[Partnership] = edge(
        label="PARTNERS_WITH",
        default_factory=list,
        description="Named ecosystem/technology partners mentioned in the business description.",
    )
    # --- Validators ---

    @field_validator("fiscal_year", mode="before")
    @classmethod
    def _coerce_fiscal_year(cls, v: Any) -> Any:
        """Accept '2025', 'FY2025', or 'Fiscal 2025' and extract the 4-digit year."""
        if isinstance(v, str):
            match = re.search(r"(\d{4})", v)
            if match:
                return int(match.group(1))
        return v

    @field_validator("filing_type", mode="before")
    @classmethod
    def _normalize_filing_type(cls, v: Any) -> Any:
        return _normalize_enum(FilingType, v)

    @model_validator(mode="after")
    def _dedupe_estimates(self) -> Self:
        """Keep first occurrence per normalized estimate topic (removes chunk-duplicate echoes)."""
        if not self.critical_accounting_estimates:
            return self
        seen: set[str] = set()
        unique: list[str] = []
        for item in self.critical_accounting_estimates:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(item)
        object.__setattr__(self, "critical_accounting_estimates", unique)
        return self
