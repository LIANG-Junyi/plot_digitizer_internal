from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    x: Union[str, float]
    mean: Optional[float] = None
    error_plus: Optional[float] = None
    error_minus: Optional[float] = None


class ChartSeries(BaseModel):
    name: str
    data: List[DataPoint]


class ChartData(BaseModel):
    figure_id: str = "Figure"
    caption: Optional[str] = None
    chart_type: str = "unknown"
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    unit: Optional[str] = None
    series: List[ChartSeries] = Field(default_factory=list)
    notes: Optional[str] = None


class PaperMetadata(BaseModel):
    title: Optional[str] = None
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    research_location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None
    crop_or_species: Optional[str] = None
    experimental_design: Optional[str] = None
    n_replicates: Optional[int] = None
    n_treatments: Optional[int] = None
    treatment_start: Optional[str] = None
    treatment_end: Optional[str] = None
    data_collection_dates: Optional[str] = None
    soil_type: Optional[str] = None
    notes: Optional[str] = None


class PaperResult(BaseModel):
    metadata: PaperMetadata
    charts: List[ChartData]
