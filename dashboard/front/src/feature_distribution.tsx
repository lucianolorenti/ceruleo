import { API } from './API'
import { ResponsiveLine } from '@nivo/line'
import React, { useEffect } from 'react'
import { CircularProgress } from '@material-ui/core'
import { isEmpty, zip } from './utils'
interface FeatureDistributionProps {

    api: API;
    features: Array<string>
}
export default function FeatureDistribution(props: FeatureDistributionProps) {
    const [data, setData] = React.useState(null)
    const [loading, setLoading] = React.useState(false)
    const dataReady = (d) => {
        setLoading(false)
        setData(d)
    }
    let histograms = []
    useEffect(() => {
        setData(null)
        setLoading(true)
        props.api.features_histogram(props.features, dataReady)
    }, [props.features]);

    if ((data == null) && (!loading)) {
        return null;
    }
    if ((data == null) && (loading)) {
        return <CircularProgress />
    }
    if (isEmpty(data)) {
        return null;
    }
    console.log(data)
    for (var feature in props.features) {
        if (!(props.features[feature] in data)) {
            continue;
        }
        const feature_data = data[props.features[feature]]

        for (var i = 0; i < feature_data.length; i++) {
            const bins = feature_data[i]['bins']
            const h_data = feature_data[i]['values']
            histograms.push(
                {
                    "id": 'Life ' + i,
                    "data": zip(bins, h_data).map((data, i) => {
                        return {
                            "x": data[0],
                            "y": data[1]
                        }
                    })
                }
            )
        }
    }


    console.log(histograms)
      return (
          <div  style={{height: '600px'}} >
           <ResponsiveLine
              enableArea={true}
               data={histograms}
               margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
               xScale={{ type: 'linear', max:'auto', min:'auto' }}
               yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: false}}
               yFormat=" >-.2f"
               curve='monotoneX'
               enablePoints={false}
               axisTop={null}
               axisRight={null}
               axisBottom={{
                  
                   tickSize: 5,
                   tickPadding: 5,
                   tickRotation: 0,
                   legend: 'transportation',
                   legendOffset: 36,
                   legendPosition: 'middle'
               }}
               axisLeft={{
                   
                   tickSize: 5,
                   tickPadding: 5,
                   tickRotation: 0,
                   legend: 'count',
                   legendOffset: -40,
                   legendPosition: 'middle'
               }}
               pointSize={10}
               pointColor={{ theme: 'background' }}
               pointBorderWidth={2}
               pointBorderColor={{ from: 'serieColor' }}
               pointLabelYOffset={-12}
               useMesh={true}
               legends={[
                {
                    anchor: 'bottom-right',
                    direction: 'column',
                    justify: false,
                    translateX: 100,
                    translateY: 0,
                    itemsSpacing: 0,
                    itemDirection: 'left-to-right',
                    itemWidth: 80,
                    itemHeight: 20,
                    itemOpacity: 0.75,
                    symbolSize: 12,
                    symbolShape: 'circle',
                    symbolBorderColor: 'rgba(0, 0, 0, .5)',
                    effects: [
                        {
                            on: 'hover',
                            style: {
                                itemBackground: 'rgba(0, 0, 0, .03)',
                                itemOpacity: 1
                            }
                        }
                    ]
                }
            ]}

           />
           </div>
       )
}