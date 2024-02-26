package gqserver.api.data.earthquake.advanced;

import java.io.Serial;
import java.io.Serializable;
import java.util.List;

public record AdvancedHypocenterData(HypocenterQualityData qualityData,
                                     DepthConfidenceIntervalData depthIntervalData,
                                     LocationConfidenceIntervalData locationConfidenceIntervalData,
                                     StationCountData stationCountData,
                                     List<Float> magsData) implements Serializable {
    @Serial
    private static final long serialVersionUID = 0L;
}
