package gqserver.api.data.earthquake.advanced;

import java.io.Serial;
import java.io.Serializable;
import java.util.List;

public record LocationConfidenceIntervalData(
        List<PolygonConfidenceIntervalData> polygonConfidenceIntervalDataList) implements Serializable {
    @Serial
    private static final long serialVersionUID = 0L;
}
