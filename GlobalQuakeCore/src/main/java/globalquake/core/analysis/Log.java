package globalquake.core.analysis;

public record Log(long time, int rawValue, float filteredV, float ratio, float mediumRatio,
                  float specialRatio) {

}
