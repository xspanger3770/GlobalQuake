package gqserver.api.data.earthquake.advanced;

import java.io.Serializable;

public record AdvancedHypocenterData(HypocenterQualityData qualityData,
                                     DepthConfidenceIntervalData depthIntervalData,
                                     LocationConfidenceIntervalData locationConfidenceIntervalData) implements Serializable {

}
