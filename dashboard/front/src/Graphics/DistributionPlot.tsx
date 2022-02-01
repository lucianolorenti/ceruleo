import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'

import ReactApexChart from 'react-apexcharts';
import { ApexOptions } from 'apexcharts';
import { PlotData } from './Types';
import { CircularProgress, Input, InputLabel, MenuItem, Select, SelectChangeEvent } from '@mui/material';
import { Toolbar } from '@mui/material';
import { useFeatureNames } from '../Dataset/Store/FeatureNames';
import ColorScheme from 'color-scheme'


function build_distribution_options(title: string, colors:Array<string>): ApexOptions {
    console.log(colors)
    return {
        colors: colors,
        chart: {
            
            height: 350,
            type: "area",
            stacked: false,
            animations: {
                enabled: false
            }
        },
        dataLabels: {
            enabled: false
        },
        fill: {
            colors: colors
        },
          
        title: {
            text: title,
            align: 'left',
            offsetX: 110
        },
        markers: {
            size: 0
        },
        stroke: {
            curve: 'smooth'
        },
        yaxis: [
            {
                axisTicks: {
                    show: true,
                },
                axisBorder: {
                    show: true,
                    color: '#008FFB'
                },
                labels: {
                    style: {
                        colors: '#008FFB',
                    }
                },
                tooltip: {
                    enabled: true
                }
            }

        ],
        xaxis: {
            tickAmount: 20
        },
        tooltip: {
            fixed: {
                enabled: true,
                position: 'topLeft', // topRight, topLeft, bottomRight, bottomLeft
                offsetY: 30,
                offsetX: 60
            },
        },
        legend: {
            horizontalAlign: 'left',
            offsetX: 40
        }
    }
}

interface DistributionPlotProps {
    title: string
    selectedFeature: string
    fetch_data: (feature: string, life_number: number, nbins: number, colorizedBy:string) => Promise<PlotData>
}

class ColorGenerator {
    color_map: { [k: string] : string; };
    colors: Array<string>
    constructor() {
        var scheme = new ColorScheme;
        this.colors = scheme.from_hue(21)         
            .scheme('triade')    
            .variation('soft')
            .colors(); 
      
        this.color_map = {}
    }
    new_color = () :string => {
        var size = Object.keys(this.color_map).length;
        return '#' + this.colors[size]
    }
    get_color = (category: any) :string => {
        if (!(category  in this.color_map)) {
            
            this.color_map[category] = this.new_color()
         
        }
        return this.color_map[category]

    }
}

export default function DistributionPlot(props: DistributionPlotProps) {
    const [colorizedBy, setColorizedBy] = useState("life")
    const [featureNames, featuresLoading] = useFeatureNames()
    const [loading, setLoading] = useState(false)
    const [distributionData, setDistributionData] = useState<Array<PlotData>>([])
    const [nbins, setNBins] = useState<number>(15)
    const { title, selectedFeature, fetch_data } = props;
    const updateDistributionArray = (elem: PlotData, i: number) => {
        setDistributionData(items => [...items, elem]);
    }
    useEffect(() => {
        if (selectedFeature === null) {
            return
        }
        setLoading(true)
        setDistributionData([])
        let promises = []
        for (let i = 0; i < 10; i++) {
            const p = fetch_data(selectedFeature, i, nbins, colorizedBy).then((e: PlotData) => updateDistributionArray(e, i))
            promises.push(p)
        }
        Promise.all(promises).then(() =>{
            setLoading(false)
        })
        

    }, [selectedFeature, nbins, colorizedBy])

    const handleColorizedByChange = (event: SelectChangeEvent) => {
        setColorizedBy(event.target.value as string);
      };
    const handleNBinsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setNBins(event.target.value === '' ? 2 : Number(event.target.value))
    }
    const cg =new  ColorGenerator()
    let colors = []
    for (let i =0;i<distributionData.length;i++) {
        console.log(distributionData[i].category)
        colors.push(cg.get_color(distributionData[i].category))
    }

    
    return <div>
        <Toolbar>
            <InputLabel shrink>Number of bins</InputLabel> <Input
                value={nbins}
                size="small"
                onChange={handleNBinsChange}
                inputProps={{
                    step: 2,
                    min: 2,
                    max: 100,
                    type: 'number',
                    'aria-labelledby': 'input-slider',
                }}
            />
            <InputLabel shrink>Colorize by</InputLabel>
            <Select
                labelId="demo-simple-select-label"
                id="demo-simple-select"
                value={colorizedBy}
                label="Age"
                onChange={handleColorizedByChange}

            >
                <MenuItem value="life">Life</MenuItem>
              {  featureNames.categoricals.map((elem, idx) => {
                    return <MenuItem  key={idx} value={elem}>{elem}</MenuItem>
                })}
            </Select>

        </Toolbar>
        {loading ? <CircularProgress /> : <ReactApexChart
            options={build_distribution_options(title, colors)}
            series={distributionData}
            height={350} />}
    </div>
}