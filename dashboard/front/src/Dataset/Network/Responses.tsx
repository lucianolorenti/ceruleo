export interface Point {
    x: number
    y:number
  }
  export interface LineData {
    id: string
    data: Array<Point>
  }

  export interface SeriesData {
      id: string
      data: Array<number>
  }


export interface PlotData {
 id: string
 data: LineData  
 category?: string
}
  
  interface KLDivergenceTableRow {
    feature: string;
    mean_divergence: Number;
  }
  interface BoxPlotDataPoint {
    x:any
    y:number
  }
  export interface BoxPlotData {
      x: any
      min: number
      max: number
      q1: number
      q3: number
      median: number
  }
  export interface BoxPlot {
    boxplot: Array<BoxPlotData>
    outliers: Array<Array<number>>
  }

  interface DataFrameField {
    name: string;
    type: string;
  }
  
  interface DataFrameSchema {
    fields: Array<DataFrameField>;
    primaryKey: Array<string>;
    pandas_version: string;
  }
  
  export interface DataFrameInterface {
    schema: DataFrameSchema;
    data: Array<Object>;
  }