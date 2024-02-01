package globalquake.core.analysis;

import java.io.Serial;
import java.io.Serializable;

public record Log(long time, int rawValue, float filteredV, float shortAverage, float mediumAverage, float longAverage,
				  float specialAverage, byte status) {

	public double getRatio() {
		return shortAverage() / longAverage();
	}

	public double getMediumRatio() {
		return mediumAverage() / longAverage();
	}

	public double getSpecialRatio() {
		return specialAverage() / longAverage();
	}

}
