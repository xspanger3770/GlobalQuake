package globalquake.core.analysis;

import java.io.Serial;
import java.io.Serializable;

public record Log(long time, int rawValue, float filteredV, float shortAverage, float mediumAverage, float longAverage,
				  float thirdAverage, float specialAverage, AnalysisStatus status) implements Serializable {

	@Serial
	private static final long serialVersionUID = 5578601429266635895L;


	public double getRatio() {
		return shortAverage() / longAverage();
	}

	public double getMediumRatio() {
		return mediumAverage() / longAverage();
	}

	public double getThirdRatio() {
		return thirdAverage() / longAverage();
	}

	public double getSpecialRatio() {
		return specialAverage() / longAverage();
	}

}
