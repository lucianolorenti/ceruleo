import {API} from './API'

interface FeatureDistributionProps {
    life_proportion:Number;
    api: API;
}
function FeatureDistribution(props: FeatureDistributionProps) {

    const feature = "hola"
    const histogram = props.api.features_histogram(feature)
    return null
}